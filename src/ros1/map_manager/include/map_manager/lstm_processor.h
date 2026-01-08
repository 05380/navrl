#ifndef LSTM_PROCESSOR_H
#define LSTM_PROCESSOR_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <Eigen/Dense>

class ObstacleLSTM : public torch::nn::Module {
public:
    ObstacleLSTM(int input_size, int hidden_size, int num_layers, int output_size);
    
    torch::Tensor forward(torch::Tensor input);
    
private:
    torch::nn::LSTM lstm;
    torch::nn::Linear linear;
};

class LSTMProcessor {
public:
    LSTMProcessor();
    ~LSTMProcessor();
    
    // 处理障碍物特征序列，返回潜在状态
    Eigen::VectorXd process(const std::vector<std::vector<Eigen::Vector3d>>& obstacle_sequence);
    
    // 更新特征序列
    void updateFeatureSequence(const std::vector<Eigen::Vector3d>& current_features);
    
    // 获取当前潜在状态
    Eigen::VectorXd getLatentState() const { return latent_state_; }

private:
    std::shared_ptr<ObstacleLSTM> lstm_model_;
    std::vector<std::vector<Eigen::Vector3d>> feature_history_;
    int max_sequence_length_;
    Eigen::VectorXd latent_state_;
    bool model_loaded_;
};

#endif