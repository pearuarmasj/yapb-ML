//
// Neural Network - LibTorch C++ Wrapper
// Real-time inference and training for zombie survival AI
//

#pragma once

#include <yapb.h>
#include <vector>
#include <memory>
#include <string>

// Forward declarations to avoid LibTorch include spam
namespace torch {
    namespace jit {
        struct Module;
    }
    class Tensor;
}

namespace ZombieAI {
    namespace ML {
        
        // Neural network input/output structures
        struct NetworkInput {
            std::vector<float> gameState;      // Position, health, etc.
            std::vector<float> hazardDistances; // Distance to environmental dangers
            std::vector<float> enemyInfo;      // Enemy positions and states
            std::vector<float> teamInfo;       // Teammate information
        };
        
        struct NetworkOutput {
            float moveForward;    // -1.0 to 1.0
            float moveRight;      // -1.0 to 1.0  
            float turnYaw;        // -1.0 to 1.0
            float turnPitch;      // -1.0 to 1.0
            bool jump;
            bool duck;
            bool attackPrimary;
            bool attackSecondary;
            float confidence;     // How confident the network is about this decision
        };
        
        class NeuralNetwork {
        private:
            std::unique_ptr<torch::jit::Module> m_model;
            bool m_isGPUAvailable;
            bool m_isInitialized;
            
            // Performance tracking
            float m_avgInferenceTime;
            int m_inferenceCount;
            
        public:
            NeuralNetwork();
            ~NeuralNetwork();
            
            // Model management
            bool loadModel(const String& modelPath);
            bool saveModel(const String& modelPath);
            bool isInitialized() const { return m_isInitialized; }
            
            // Inference
            NetworkOutput predict(const NetworkInput& input);
            std::vector<NetworkOutput> predictBatch(const std::vector<NetworkInput>& inputs);
            
            // Training (for online learning)
            void updateWeights(const std::vector<float>& gradients);
            void setLearningRate(float lr);
            
            // Performance monitoring
            float getAverageInferenceTime() const { return m_avgInferenceTime; }
            bool isGPUAccelerated() const { return m_isGPUAvailable; }
            
            // Utility
            static std::vector<float> encodeInput(const NetworkInput& input);
            static NetworkOutput decodeOutput(const std::vector<float>& rawOutput);
        };
        
    } // namespace ML
} // namespace ZombieAI
