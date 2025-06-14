//
// Online Trainer - Real-Time Neural Network Training
// Manages training loop during gameplay at 240 FPS
//

#pragma once

#include <yapb.h>
#include <ml/neural_network.h>
#include <ml/experience_buffer.h>
#include <thread>
#include <atomic>
#include <chrono>

namespace ZombieAI {
    namespace ML {
        
        // Training configuration
        struct TrainingConfig {
            float learningRate = 0.0001f;
            int batchSize = 32;
            int trainingFrequency = 60;        // Train every N frames (4 times/sec at 240 FPS)
            int targetUpdateFrequency = 1800;  // Update target network every N frames
            float discountFactor = 0.99f;
            bool useGPU = true;
            bool usePrioritizedReplay = false;
        };
        
        // Training statistics
        struct TrainingStats {
            size_t totalTrainingSteps;
            float avgLoss;
            float avgReward;
            float learningRate;
            float trainingTime;
            size_t experiencesProcessed;
            bool isTraining;
        };
        
        class OnlineTrainer {
        private:
            NeuralNetwork* m_network;
            ExperienceBuffer* m_buffer;
            TrainingConfig m_config;
            TrainingStats m_stats;
            
            // Threading
            std::thread m_trainingThread;
            std::atomic<bool> m_isRunning;
            std::atomic<bool> m_shouldTrain;
            
            // Timing
            std::chrono::high_resolution_clock::time_point m_lastTrainingTime;
            std::chrono::high_resolution_clock::time_point m_startTime;
            
            // Frame counting for training frequency
            std::atomic<int> m_frameCounter;
            
        public:
            OnlineTrainer(NeuralNetwork* network, ExperienceBuffer* buffer);
            ~OnlineTrainer();
            
            // Training control
            bool startTraining(const TrainingConfig& config);
            void stopTraining();
            void pauseTraining();
            void resumeTraining();
            
            // Frame-based training trigger (called every frame)
            void onFrame();
            
            // Manual training trigger
            void trainStep();
            bool shouldTrainThisFrame() const;
            
            // Configuration
            void updateConfig(const TrainingConfig& config);
            TrainingConfig getConfig() const { return m_config; }
            
            // Statistics and monitoring
            TrainingStats getStats() const;
            float getTrainingFPS() const;
            bool isTraining() const { return m_isRunning.load(); }
            
            // Performance optimization
            void setTrainingFrequency(int frames);
            void setLearningRate(float lr);
            void setBatchSize(int size);
            
        private:
            void trainingLoop();
            void performTrainingStep();
            float calculateLoss(const ExperienceBatch& batch);
            void updateStatistics(float loss, float trainingTime);
        };
        
    } // namespace ML
} // namespace ZombieAI
