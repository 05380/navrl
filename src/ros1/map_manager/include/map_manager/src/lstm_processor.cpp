#include "map_manager/lstm_processor.h"
#include <iostream>

ObstacleLSTM::ObstacleLSTM(int input_size, int hidden_size, int num_layers, int output_size) 
    : lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers)))),
      linear(register_module("linear", torch::nn::Linear(hidden_size, output_size))) {
}

torch::Tensor ObstacleLSTM::forward(torch::Tensor input) {
    auto lstm_out = lstm(input);
    auto output = linear(std::get<0>(lstm_out));
    return output;
}

LSTMProcessor::LSTMProcessor() : max_sequence_length_(10), model_loaded_(false) {
    try {
        // 初始化LSTM模型
        lstm_model_ = std::make_shared<ObstacleLSTM>(7, 64, 2, 32); // 输入7维，隐藏64维，2层，输出32维
        model_loaded_ = true;
        std::cout << "[LSTMProcessor]: Model initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[LSTMProcessor]: Failed to initialize model: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

LSTMProcessor::~LSTMProcessor() {}

void LSTMProcessor::updateFeatureSequence(const std::vector<Eigen::Vector3d>& current_features) {
    // 构建特征向量
    std::vector<Eigen::Vector3d> current_feature_vec;
    for (const auto& feat : current_features) {
        Eigen::Vector3d feature;
        // 提取障碍物的特征：位置、尺寸、角度
        // 这里需要根据bboxVertex的实际结构调整
        feature(0) = feat(0);  // x位置
        feature(1) = feat(1);  // y位置
        feature(2) = feat(2);  // z位置
        // 假设其他维度是尺寸信息
        current_feature_vec.push_back(feature);
    }
    
    // 将当前特征添加到历史序列中
    feature_history_.push_back(current_feature_vec);
    
    // 保持序列长度在限制范围内
    if (feature_history_.size() > max_sequence_length_) {
        feature_history_.erase(feature_history_.begin());
    }
}

Eigen::VectorXd LSTMProcessor::process(const std::vector<std::vector<Eigen::Vector3d>>& obstacle_sequence) {
    if (!model_loaded_ || obstacle_sequence.empty()) {
        // 返回默认零向量
        return Eigen::VectorXd::Zero(32);
    }
    
    try {
        // 准备LSTM输入
        std::vector<torch::Tensor> sequence_tensors;
        for (const auto& features : obstacle_sequence) {
            if (!features.empty()) {
                // 将特征转换为一维张量
                std::vector<float> flat_features;
                for (const auto& feat : features) {
                    for (int i = 0; i < 3 && i < feat.size(); ++i) {
                        flat_features.push_back(feat(i));
                    }
                    // 如果特征不足7维，补零
                    while (flat_features.size() % 7 != 0) {
                        flat_features.push_back(0.0f);
                    }
                }
                
                if (!flat_features.empty()) {
                    // 确保至少有7个元素
                    if (flat_features.size() < 7) {
                        flat_features.resize(7, 0.0f);
                    }
                    
                    torch::Tensor tensor = torch::from_blob(
                        flat_features.data(), 
                        {1, static_cast<long>(flat_features.size())}, 
                        torch::kFloat
                    ).clone(); // 使用clone避免内存问题
                    sequence_tensors.push_back(tensor);
                }
            }
        }
        
        if (sequence_tensors.empty()) {
            return Eigen::VectorXd::Zero(32);
        }
        
        // 组合序列
        torch::Tensor input_tensor = torch::stack(sequence_tensors, 0); // [seq_len, batch_size, feature_size]
        
        // 前向传播
        torch::NoGradGuard no_grad;
        torch::Tensor output = lstm_model_->forward(input_tensor);
        
        // 提取最终的潜在状态
        torch::Tensor final_output = output[output.size(0)-1][0]; // [batch, feature]
        std::vector<float> output_vec(final_output.data_ptr<float>(), 
                                      final_output.data_ptr<float>() + final_output.numel());
        
        Eigen::VectorXd result(output_vec.size());
        for (size_t i = 0; i < output_vec.size(); ++i) {
            result(i) = output_vec[i];
        }
        
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[LSTMProcessor]: Error during processing: " << e.what() << std::endl;
        return Eigen::VectorXd::Zero(32);
    }
}