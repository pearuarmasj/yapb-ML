//
// Experience Buffer - Replay Buffer for Online Learning
// Stores training experiences for neural network updates
//

#pragma once

#include <yapb.h>
#include <ml/neural_network.h>
#include <vector>
#include <deque>
#include <mutex>
#include <random>

namespace ZombieAI {
    namespace ML {
        
        // Single training experience
        struct Experience {
            NetworkInput state;           // Game state at time t
            NetworkOutput action;         // Action taken at time t
            float reward;                 // Reward received
            NetworkInput nextState;       // Game state at time t+1
            bool isTerminal;             // Did episode end (death/round end)?
            float timestamp;             // When this experience occurred
            int botId;                   // Which bot generated this experience
        };
        
        // Batch of experiences for training
        struct ExperienceBatch {
            std::vector<Experience> experiences;
            float avgReward;
            int batchSize;
            float timestamp;
        };
        
        class ExperienceBuffer {
        private:
            std::deque<Experience> m_buffer;
            mutable std::mutex m_bufferMutex;
            std::mt19937 m_randomGenerator;
            
            // Configuration
            size_t m_maxSize;
            size_t m_minSizeForSampling;
            
            // Statistics
            size_t m_totalExperiences;
            float m_avgReward;
            size_t m_environmentalDeaths;
            
        public:
            ExperienceBuffer(size_t maxSize = 50000, size_t minSize = 1000);
            ~ExperienceBuffer();
            
            // Experience management
            void addExperience(const Experience& exp);
            void addExperiences(const std::vector<Experience>& experiences);
            
            // Sampling for training
            ExperienceBatch sampleBatch(size_t batchSize);
            ExperienceBatch samplePrioritized(size_t batchSize, float alpha = 0.6f);
            bool canSample() const;
            
            // Buffer management
            void clear();
            size_t size() const;
            bool isFull() const;
            float getCapacityRatio() const;
            
            // Statistics and monitoring
            float getAverageReward() const { return m_avgReward; }
            size_t getTotalExperiences() const { return m_totalExperiences; }
            size_t getEnvironmentalDeaths() const { return m_environmentalDeaths; }
            
            // Analysis
            std::vector<Experience> getRecentExperiences(size_t count) const;
            float getRewardTrend(size_t windowSize = 1000) const;
            
            // Configuration
            void setMaxSize(size_t maxSize);
            void setMinSizeForSampling(size_t minSize);
        };
        
    } // namespace ML
} // namespace ZombieAI
