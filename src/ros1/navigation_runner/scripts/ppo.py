import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world
from lstm_feature_extractor import StaticObstacleLSTMExtractor, StaticObstacleLSTMExtractorV2

"""
定义 PPO 模型封装类 PPO，含特征提取、actor、critic、损失与优化器，供导航节点加载推理/训练。
增强：支持通过LSTM网络处理静态障碍物特征的时间序列信息
"""

class PPO(TensorDictModuleBase):

    def __init__(self, cfg, observation_spec, action_spec, device, enable_static_obstacle_lstm=False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.enable_static_obstacle_lstm = enable_static_obstacle_lstm

        
        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),#三层卷积
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)
        
        # 静态障碍物特征LSTM提取器（可选）
        if enable_static_obstacle_lstm:
            # CNN特征处理
            self.static_obstacle_cnn = nn.Sequential(
                nn.LazyConv2d(out_channels=8, kernel_size=[3, 3], padding=[1, 1]), nn.ReLU(),
                nn.LazyConv2d(out_channels=16, kernel_size=[3, 3], stride=[2, 1], padding=[1, 1]), nn.ReLU(),
                Rearrange("n c w h -> n (c w h)"),
                nn.LazyLinear(64), nn.LayerNorm(64),
            ).to(self.device)
            
            # LSTM处理时间序列
            self.static_obstacle_lstm = StaticObstacleLSTMExtractor(
                static_obs_feature_dim=64,
                lstm_hidden_dim=32,
                lstm_num_layers=1,
                output_dim=64,
                sequence_len=10,
                device=device
            )
            
            static_obstacle_feature_dim = 64
        else:
            self.static_obstacle_cnn = None
            self.static_obstacle_lstm = None
            static_obstacle_feature_dim = 0

        # 根据是否启用LSTM动态构建feature_extractor
        if enable_static_obstacle_lstm:
            # 创建一个包装网络来处理静态障碍物特征
            static_obstacle_wrapper = StaticObstacleFeatureWrapper(
                self.static_obstacle_cnn,
                self.static_obstacle_lstm,
                device
            )
            
            # feature_extractor 串联（包含静态障碍物LSTM处理）
            self.feature_extractor = TensorDictSequential(
                TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
                TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
                TensorDictModule(static_obstacle_wrapper, [("agents", "observation", "static_obstacle")], ["_static_obstacle_lstm_feature"]),
                CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature", "_static_obstacle_lstm_feature"], "_feature", del_keys=False), 
                TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
            ).to(self.device)
        else:
            # feature_extractor 串联（原始版本）
            self.feature_extractor = TensorDictSequential(
                TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
                TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
                CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
                TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
            ).to(self.device)

        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], #输出归一化动作
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)#价值归一化

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict
    
    def reset_static_obstacle_lstm(self, batch_size=1):
        """重置静态障碍物LSTM的隐状态（用于新场景或新任务）"""
        if self.enable_static_obstacle_lstm and self.static_obstacle_lstm is not None:
            self.static_obstacle_lstm.reset_hidden_state(batch_size)
            self.static_obstacle_lstm.reset_feature_buffer(batch_size)


class StaticObstacleFeatureWrapper(nn.Module):
    """
    静态障碍物特征处理包装器
    组合CNN特征提取和LSTM时间序列处理
    """
    
    def __init__(self, cnn_network, lstm_processor, device):
        super(StaticObstacleFeatureWrapper, self).__init__()
        self.cnn_network = cnn_network
        self.lstm_processor = lstm_processor
        self.device = device
    
    def forward(self, static_obstacle_input):
        """
        参数：
            static_obstacle_input: 静态障碍物输入
                - 在在线推理时：[batch_size, channels, H, W]
                - 在训练时可能是序列：[batch_size, seq_len, channels, H, W]
        
        返回：
            lstm_feature: LSTM处理后的特征 [batch_size, output_dim]
        """
        # 检查输入是否为序列
        if len(static_obstacle_input.shape) == 5:
            # 序列输入 [batch_size, seq_len, channels, H, W]
            batch_size, seq_len, channels, h, w = static_obstacle_input.shape
            
            # 重新整形用于CNN处理
            input_reshaped = static_obstacle_input.view(
                batch_size * seq_len, channels, h, w
            )
            
            # CNN特征提取
            cnn_features = self.cnn_network(input_reshaped)
            
            # 重新整形回序列
            cnn_features = cnn_features.view(batch_size, seq_len, -1)
            
            # LSTM处理
            lstm_feature = self.lstm_processor(cnn_features)
        else:
            # 单帧输入 [batch_size, channels, H, W]
            # CNN特征提取
            cnn_feature = self.cnn_network(static_obstacle_input)
            
            # LSTM逐帧处理
            lstm_feature = self.lstm_processor(cnn_feature)
        
        return lstm_feature