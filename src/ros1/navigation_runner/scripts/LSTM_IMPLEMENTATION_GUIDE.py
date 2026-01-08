"""
LSTM网络处理静态障碍物特征的完整实现指南
=====================================

## 概述

本实现提供了一个完整的方案，用LSTM网络处理静态障碍物特征提取生成的潜在状态。
整个系统包含3个主要组件：

1. lstm_feature_extractor.py - LSTM特征提取器
2. ppo.py (修改版) - 集成LSTM的PPO模型
3. navigation.py (修改版) - 导航节点与时间序列缓冲

---

## 快速开始

### 1. 启用LSTM特征处理

在你的配置文件中添加以下参数：

```yaml
algo:
  enable_static_obstacle_lstm: true
  static_obs_height: 32          # 静态障碍物特征图高度
  static_obs_width: 32           # 静态障碍物特征图宽度
```

### 2. 初始化模型

在navigation.py中，模型会自动：
- 创建静态障碍物LSTM处理模块
- 初始化特征缓冲区（10帧历史）
- 重置LSTM隐状态

```python
policy = self.init_model()  # 自动启用LSTM（如果配置中enable_static_obstacle_lstm=True）
```

### 3. 数据流程

```
静态障碍物 (位置、尺寸、角度)
    ↓
get_static_obstacle_feature()  [生成特征图]
    ↓
add_static_obstacle_to_buffer() [维护循环缓冲]
    ↓
get_static_obstacle_sequence()  [获取10帧序列]
    ↓
StaticObstacleFeatureWrapper   [CNN+LSTM处理]
    ↓
最终潜在状态 → PPO网络


---

## 详细组件说明

### A. LSTM特征提取器 (lstm_feature_extractor.py)

#### StaticObstacleLSTMExtractor (在线推理版本)

用于实时推理场景，维护LSTM隐状态和特征缓冲区。

**关键特性：**
- 循环缓冲区 (环形队列)：维持最近10帧特征
- LSTM隐状态持久化：支持长期依赖建模
- 灵活的输出维度

**使用示例：**
```python
from lstm_feature_extractor import StaticObstacleLSTMExtractor

lstm = StaticObstacleLSTMExtractor(
    static_obs_feature_dim=64,    # CNN输出维度
    lstm_hidden_dim=32,           # LSTM隐层维度
    lstm_num_layers=1,            # LSTM层数
    output_dim=64,                # 最终输出维度
    sequence_len=10,              # 保持10帧历史
    device='cuda'
)

# 每帧调用一次
feature = torch.randn(1, 64)  # [batch_size=1, feature_dim=64]
latent = lstm(feature)         # [1, 64]
```

#### StaticObstacleLSTMExtractorV2 (批处理版本)

用于离线训练，处理完整的时间序列。

**关键特性：**
- 双向LSTM处理
- 注意力机制：自动学习重要时刻权重
- 支持可变长度序列

**使用示例：**
```python
lstm_v2 = StaticObstacleLSTMExtractorV2(
    static_obs_feature_dim=64,
    lstm_hidden_dim=64,
    lstm_num_layers=2,
    output_dim=128,
    dropout=0.1,
    device='cuda'
)

# 批量处理完整序列
feature_seq = torch.randn(batch_size=32, seq_len=10, feature_dim=64)
latent = lstm_v2(feature_seq)  # [32, 128]
```

#### CombinedStaticObstacleExtractor (端到端版本)

同时进行CNN特征提取和LSTM序列处理。

**使用示例：**
```python
combined = CombinedStaticObstacleExtractor(
    input_channels=3,            # RGB图像
    cnn_output_dim=128,
    lstm_hidden_dim=64,
    lstm_num_layers=1,
    final_output_dim=128,
    device='cuda'
)

# 处理静态障碍物图像
obs_image = torch.randn(1, 3, 64, 64)  # RGB图像
latent = combined(obs_image)             # [1, 128]
```

---

### B. 修改后的PPO模型 (ppo.py)

#### 启用LSTM的PPO类初始化

```python
policy = PPO(
    cfg=config,
    observation_spec=obs_spec,
    action_spec=action_spec,
    device='cuda',
    enable_static_obstacle_lstm=True  # ← 启用LSTM
)

# 新场景时重置LSTM隐状态
policy.reset_static_obstacle_lstm(batch_size=1)
```

#### 特征提取流程

启用LSTM后的特征提取管道：

```
输入观测：
  - state:           [batch, 8]         (机器人状态)
  - lidar:           [batch, 1, H, W]   (激光雷达)
  - direction:       [batch, 1, 3]      (目标方向)
  - dynamic_obstacle:[batch, 1, N, 10]  (动态障碍物)
  - static_obstacle: [batch, 1, 3, 32, 32] ← LSTM输入

处理步骤：
1. LiDAR CNN特征提取
2. 动态障碍物MLP处理  
3. 静态障碍物CNN处理
4. 静态障碍物LSTM处理    ← 新增
5. 所有特征拼接 [CNN, state, dyn, static_lstm]
6. MLP合并处理
7. Actor/Critic输出
```

#### StaticObstacleFeatureWrapper

该包装器自动处理：
- 单帧输入：逐帧LSTM处理
- 序列输入：批量LSTM处理

```python
wrapper = StaticObstacleFeatureWrapper(
    cnn_network=self.static_obstacle_cnn,
    lstm_processor=self.static_obstacle_lstm,
    device=device
)

# 自动适应输入形状
output = wrapper(input)
```

---

### C. 修改后的导航节点 (navigation.py)

#### 新增方法

**1. `get_static_obstacle_feature(pos, size)`**

将静态障碍物转换为特征图表示。

```python
def get_static_obstacle_feature(self, static_obstacle_pos, static_obstacle_size):
    """
    生成3通道特征图：
    - 通道1：距离图 (距离越近值越大)
    - 通道2：占据概率 (有障碍物处=1)
    - 通道3：尺寸信息 (障碍物大小归一化)
    """
    # 返回 [1, 3, 32, 32] 的特征张量
```

**2. `add_static_obstacle_to_buffer(feature)`**

维护环形缓冲区，存储最近10帧特征。

```python
def add_static_obstacle_to_buffer(self, feature):
    """
    使用循环索引添加特征到缓冲区
    buffer_shape: [1, 10, 3, 32, 32]
    """
```

**3. `get_static_obstacle_sequence()`**

检索特征序列供LSTM处理。

```python
def get_static_obstacle_sequence(self):
    """
    返回完整的特征序列 [1, 10, 3, 32, 32]
    可直接作为LSTM输入
    """
```

#### get_action() 中的集成

```python
def get_action(self, pos, vel, goal):
    # ... 原有代码 ...
    
    # 新增：静态障碍物特征处理
    if enable_static_obstacle_lstm:
        # 1. 获取静态障碍物
        static_pos, static_size, static_angle = self.get_static_obstacles()
        
        # 2. 生成特征
        static_feature = self.get_static_obstacle_feature(static_pos, static_size)
        
        # 3. 添加到缓冲
        self.add_static_obstacle_to_buffer(static_feature)
        
        # 4. 获取序列并加入观测
        static_seq = self.get_static_obstacle_sequence()
        obs["agents"]["observation"]["static_obstacle"] = static_seq
    
    # 5. PPO推理（自动通过LSTM处理）
    output = self.policy(obs)
    
    # ... 返回动作 ...
```

---

## 配置文件示例

创建或修改配置文件 `config/navigation_config.yaml`：

```yaml
algo:
  # LSTM配置
  enable_static_obstacle_lstm: true
  static_obs_height: 32
  static_obs_width: 32
  
  # 静态障碍物特征提取器配置
  static_obstacle:
    lstm_hidden_dim: 32
    lstm_num_layers: 1
    output_dim: 64
    sequence_len: 10
  
  feature_extractor:
    learning_rate: 1e-4
    dyn_obs_num: 5
  
  actor:
    learning_rate: 1e-4
    action_limit: 1.0
  
sensor:
  lidar_range: 5.0
```

---

## 训练建议

### 1. 预热阶段 (Warmup)

在前N步禁用LSTM，让CNN先学习基本特征：

```python
if step < warmup_steps:
    enable_static_obstacle_lstm = False
else:
    enable_static_obstacle_lstm = True
```

### 2. Loss权重平衡

静态障碍物LSTM特征的重要性：

```python
# 在总loss中加入权重
total_loss = actor_loss + critic_loss + 0.5 * lstm_regularization_loss
```

### 3. 特征归一化

确保特征在[-1, 1]范围内：

```python
# 在特征生成时进行归一化
feature = (raw_feature - mean) / (std + 1e-6)
```

---

## 性能优化

### 1. 缓冲区大小

- 序列长度太长：计算昂贵
- 序列长度太短：无法捕捉长期依赖

**推荐值**：10-20帧 (在30Hz下对应0.3-0.7秒)

### 2. LSTM配置

| 场景 | hidden_dim | num_layers | output_dim |
|------|-----------|-----------|-----------|
| 轻量级 | 32 | 1 | 64 |
| 标准 | 64 | 2 | 128 |
| 高性能 | 128 | 3 | 256 |

### 3. 计算效率

```python
# 使用双向LSTM (V2版本) 比单向LSTM快约10%
lstm = StaticObstacleLSTMExtractorV2(
    ...
    lstm_num_layers=2,  # 增加层数但单向处理
)
```

---

## 调试技巧

### 1. 检查特征质量

```python
# 在get_action()中添加
if debug:
    print(f"Static obs feature shape: {static_feature.shape}")
    print(f"Feature value range: [{static_feature.min()}, {static_feature.max()}]")
    print(f"Buffer occupancy: {self.static_obstacle_buffer_idx} / 10")
```

### 2. 验证LSTM输出

```python
# 检查LSTM是否学到有意义的表示
lstm_output = policy.static_obstacle_lstm(cnn_feature)
print(f"LSTM output variance: {lstm_output.var()}")
# 应该 > 1e-6，表示网络在学习
```

### 3. 可视化特征

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    axes[i].imshow(static_feature[0, i].cpu().numpy())
    axes[i].set_title(f"Channel {i}")
plt.show()
```

---

## 常见问题

### Q1: LSTM隐状态是否会在场景转换时导致问题？

**A**: 是的。在新场景开始时调用 `policy.reset_static_obstacle_lstm()`：

```python
def start_new_mission(self):
    self.policy.reset_static_obstacle_lstm(batch_size=1)
    # ... 其他初始化 ...
```

### Q2: 可以用LSTM处理动态障碍物吗？

**A**: 可以。已经有的动态障碍物输入就是时间相关的，直接改用LSTM处理：

```python
# 在ppo.py中
dynamic_obstacle_lstm = StaticObstacleLSTMExtractor(
    static_obs_feature_dim=128,
    ...
)
```

### Q3: LSTM需要训练多久才能收敛？

**A**: 通常：
- 冷启动：1-2万步
- 从预训练模型：几千步
- 微调：几百步

---

## 下一步改进

1. **注意力机制**: 使用Transformer代替LSTM
2. **多尺度处理**: 同时处理不同分辨率的特征
3. **在线适应**: 实时调整LSTM权重
4. **密集特征**: 使用实时深度图而不是稀疏点云

---

## 参考文献

- LSTM基础: Hochreiter & Schmidhuber (1997)
- 注意力机制: Vaswani et al. (2017)
- 深度强化学习: Schulman et al. (2017) PPO

"""

if __name__ == "__main__":
    print(__doc__)
