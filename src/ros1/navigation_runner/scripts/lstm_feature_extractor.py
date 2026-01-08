"""
LSTM特征提取器模块
用于处理静态障碍物特征的时间序列信息
将静态障碍物特征通过LSTM网络处理，生成更好的潜在状态表示
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import make_mlp


class StaticObstacleLSTMExtractor(nn.Module):
    """
    LSTM特征提取器：处理静态障碍物特征的时间序列
    
    功能：
    1. 维护静态障碍物特征的历史记录（循环缓冲区）
    2. 通过LSTM捕捉时间动态信息
    3. 生成融合时间信息的潜在状态
    """
    
    def __init__(self, 
                 static_obs_feature_dim=128,  # 静态障碍物CNN特征维度
                 lstm_hidden_dim=64,           # LSTM隐层维度
                 lstm_num_layers=1,            # LSTM层数
                 output_dim=128,               # 最终输出维度
                 sequence_len=10,              # 历史序列长度
                 device='cpu'):
        """
        参数：
            static_obs_feature_dim: 输入的静态障碍物特征维度
            lstm_hidden_dim: LSTM隐隐层维度
            lstm_num_layers: LSTM层数
            output_dim: LSTM处理后的输出维度
            sequence_len: 维持的时间序列长度
            device: 计算设备
        """
        super(StaticObstacleLSTMExtractor, self).__init__()
        
        self.static_obs_feature_dim = static_obs_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.output_dim = output_dim
        self.sequence_len = sequence_len
        self.device = device
        
        # LSTM层：处理时间序列
        self.lstm = nn.LSTM(
            input_size=static_obs_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False
        ).to(device)
        
        # 输出处理网络：将LSTM输出映射到目标维度
        # 可以选择只用最后时刻的隐状态，或者对所有时刻进行池化
        self.output_network = nn.Sequential(
            nn.Linear(lstm_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU()
        ).to(device)
        
        # 特征序列缓冲区（循环缓冲）
        self.feature_buffer = None
        self.buffer_index = 0
        
        # 初始化LSTM隐状态
        self.hidden_state = None
        self.cell_state = None
    
    def reset_hidden_state(self, batch_size=1):
        """重置LSTM隐状态"""
        self.hidden_state = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim, 
            device=self.device
        )
        self.cell_state = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim, 
            device=self.device
        )
    
    def reset_feature_buffer(self, batch_size=1):
        """重置特征缓冲区"""
        self.feature_buffer = torch.zeros(
            batch_size, self.sequence_len, self.static_obs_feature_dim,
            device=self.device
        )
        self.buffer_index = 0
    
    def add_feature_to_buffer(self, feature):
        """
        将新特征添加到循环缓冲区
        
        参数：
            feature: 形状为 [batch_size, static_obs_feature_dim] 的特征张量
        
        返回：
            buffer: 当前缓冲区中的所有特征
            valid_len: 实际有效的序列长度（在填满缓冲区之前）
        """
        if self.feature_buffer is None:
            batch_size = feature.shape[0]
            self.reset_feature_buffer(batch_size)
        
        # 添加到缓冲区的当前位置
        self.feature_buffer[:, self.buffer_index, :] = feature
        
        # 循环索引
        self.buffer_index = (self.buffer_index + 1) % self.sequence_len
        
        # 计算有效长度（在缓冲区未满之前）
        valid_len = min(self.buffer_index + 1, self.sequence_len)
        
        return self.feature_buffer, valid_len
    
    def forward(self, static_obs_feature):
        """
        前向传播：处理静态障碍物特征
        
        参数：
            static_obs_feature: 形状为 [batch_size, static_obs_feature_dim] 的特征
        
        返回：
            latent_state: 处理后的潜在状态，形状为 [batch_size, output_dim]
        """
        batch_size = static_obs_feature.shape[0]
        
        # 添加特征到缓冲区
        feature_seq, valid_len = self.add_feature_to_buffer(static_obs_feature)
        
        # 使用有效的序列长度
        # 如果序列还没有填满，只使用前valid_len个时刻
        input_seq = feature_seq[:, :valid_len, :]
        
        # 如果hidden_state未初始化或batch_size不匹配，重新初始化
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            self.reset_hidden_state(batch_size)
        
        # LSTM前向传播
        # output: [batch_size, seq_len, lstm_hidden_dim]
        # h_n: [num_layers, batch_size, lstm_hidden_dim]
        # c_n: [num_layers, batch_size, lstm_hidden_dim]
        lstm_output, (self.hidden_state, self.cell_state) = self.lstm(
            input_seq,
            (self.hidden_state, self.cell_state)
        )
        
        # 取最后一个时刻的输出
        last_output = lstm_output[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # 通过输出网络处理
        latent_state = self.output_network(last_output)  # [batch_size, output_dim]
        
        return latent_state


class StaticObstacleLSTMExtractorV2(nn.Module):
    """
    改进版本：支持完整的时间序列输入
    适用于离线训练或批量处理场景
    """
    
    def __init__(self,
                 static_obs_feature_dim=128,
                 lstm_hidden_dim=64,
                 lstm_num_layers=2,
                 output_dim=128,
                 dropout=0.1,
                 device='cpu'):
        """
        参数与V1相同，但不维护缓冲区
        在forward中接收完整的时间序列
        """
        super(StaticObstacleLSTMExtractorV2, self).__init__()
        
        self.static_obs_feature_dim = static_obs_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.output_dim = output_dim
        self.device = device
        
        # 双向LSTM层以获得更好的时间表示
        self.lstm = nn.LSTM(
            input_size=static_obs_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,  # 双向处理
            dropout=dropout if lstm_num_layers > 1 else 0
        ).to(device)
        
        # 注意力机制（可选）：对LSTM输出的所有时刻进行加权
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),  # 双向所以*2
            nn.Tanh(),
            nn.Linear(lstm_hidden_dim, 1),
            nn.Softmax(dim=1)
        ).to(device)
        
        # 输出处理网络
        self.output_network = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU()
        ).to(device)
    
    def forward(self, feature_sequence, seq_lengths=None):
        """
        前向传播
        
        参数：
            feature_sequence: [batch_size, seq_len, static_obs_feature_dim]
                            完整的时间序列
            seq_lengths: 可选，实际序列长度（用于处理可变长度序列）
        
        返回：
            latent_state: [batch_size, output_dim]
        """
        batch_size, seq_len, _ = feature_sequence.shape
        
        # 处理可变长度序列
        if seq_lengths is not None:
            # 打包处理
            packed_input = pack_padded_sequence(
                feature_sequence, seq_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, _ = self.lstm(feature_sequence)
        
        # lstm_output: [batch_size, seq_len, lstm_hidden_dim*2]
        
        # 注意力加权
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        context = torch.sum(
            lstm_output * attention_weights,  # 加权求和
            dim=1
        )  # [batch_size, lstm_hidden_dim*2]
        
        # 输出处理
        latent_state = self.output_network(context)
        
        return latent_state


class CombinedStaticObstacleExtractor(nn.Module):
    """
    组合提取器：CNN + LSTM
    先用CNN提取单帧特征，再用LSTM处理时间序列
    """
    
    def __init__(self,
                 input_channels=1,
                 cnn_output_dim=128,
                 lstm_hidden_dim=64,
                 lstm_num_layers=1,
                 final_output_dim=128,
                 device='cpu'):
        """
        参数：
            input_channels: 输入通道数（深度图为1，RGB为3）
            cnn_output_dim: CNN特征维度
            lstm_hidden_dim: LSTM隐层维度
            lstm_num_layers: LSTM层数
            final_output_dim: 最终输出维度
        """
        super(CombinedStaticObstacleExtractor, self).__init__()
        
        self.device = device
        
        # CNN特征提取（可根据实际的静态障碍物输入形状修改）
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, cnn_output_dim),
            nn.LayerNorm(cnn_output_dim)
        ).to(device)
        
        # LSTM序列处理
        self.lstm_processor = StaticObstacleLSTMExtractor(
            static_obs_feature_dim=cnn_output_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            output_dim=final_output_dim,
            device=device
        )
    
    def reset(self, batch_size=1):
        """重置状态"""
        self.lstm_processor.reset_hidden_state(batch_size)
        self.lstm_processor.reset_feature_buffer(batch_size)
    
    def forward(self, static_obs_image):
        """
        前向传播
        
        参数：
            static_obs_image: [batch_size, channels, height, width]
        
        返回：
            latent_state: [batch_size, final_output_dim]
        """
        # CNN特征提取
        cnn_feature = self.cnn_extractor(static_obs_image)
        
        # LSTM序列处理
        latent_state = self.lstm_processor(cnn_feature)
        
        return latent_state
