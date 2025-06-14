//
// Zombie AI - Main Integration Header
// Clean interface for integrating ML and survival systems with YaPB
//

#pragma once

#include <yapb.h>

// ML Module
#include <ml/neural_network.h>
#include <ml/experience_buffer.h>
#include <ml/online_trainer.h>
#include <ml/state_encoder.h>

// Survival Module  
#include <survival/hazard_detector.h>
#include <survival/survival_rewards.h>
#include <survival/death_analyzer.h>

namespace ZombieAI {
    
    // AI system state
    enum class AIMode {
        DISABLED = 0,          // No AI, use original YaPB logic
        DATA_COLLECTION,       // Collect training data only
        NEURAL_INFERENCE,      // Use neural network for decisions
        ONLINE_LEARNING        // Real-time learning during gameplay
    };
    
    // Performance monitoring
    struct AIPerformance {
        float avgInferenceTime;
        float avgTrainingTime;
        float frameRate;
        size_t memoryUsage;
        bool isGPUActive;
    };
    
    // Main AI system coordinator
    class ZombieAISystem {
    private:
        // Core components
        ML::NeuralNetwork* m_neuralNetwork;
        ML::ExperienceBuffer* m_experienceBuffer;
        ML::OnlineTrainer* m_trainer;
        ML::StateEncoder* m_stateEncoder;
        
        Survival::HazardDetector* m_hazardDetector;
        Survival::SurvivalRewards* m_survivalRewards;
        Survival::DeathAnalyzer* m_deathAnalyzer;
        
        // System state
        AIMode m_currentMode;
        bool m_isInitialized;
        String m_currentMap;
        
        // Performance tracking
        AIPerformance m_performance;
        std::chrono::high_resolution_clock::time_point m_lastUpdate;
        
    public:
        ZombieAISystem();
        ~ZombieAISystem();
        
        // System lifecycle
        bool initialize();
        void shutdown();
        bool isInitialized() const { return m_isInitialized; }
        
        // Mode control
        void setMode(AIMode mode);
        AIMode getMode() const { return m_currentMode; }
        
        // Map management
        void onMapChange(const String& mapName);
        void onRoundStart();
        void onRoundEnd();
        
        // Main AI decision making (called from Bot::think())
        ML::NetworkOutput makeDecision(const Bot* bot);
        void recordExperience(const Bot* bot, const ML::NetworkOutput& action, float reward);
        void onBotDeath(const Bot* bot, const String& killerName = "");
        
        // Frame-by-frame updates
        void onFrame();  // Called every frame for training and updates
        
        // Component access (for advanced usage)
        ML::NeuralNetwork* getNeuralNetwork() { return m_neuralNetwork; }
        ML::ExperienceBuffer* getExperienceBuffer() { return m_experienceBuffer; }
        ML::OnlineTrainer* getTrainer() { return m_trainer; }
        Survival::HazardDetector* getHazardDetector() { return m_hazardDetector; }
        Survival::SurvivalRewards* getSurvivalRewards() { return m_survivalRewards; }
        Survival::DeathAnalyzer* getDeathAnalyzer() { return m_deathAnalyzer; }
        
        // Performance monitoring
        AIPerformance getPerformance() const { return m_performance; }
        void updatePerformanceStats();
        
        // Configuration and debugging
        void printSystemStatus() const;
        void printBotStats(int botId) const;
        void enableDebugOutput(bool enabled);
        
        // Static instance for global access
        static ZombieAISystem& getInstance();
        
    private:
        void initializeComponents();
        void cleanupComponents();
        void updatePerformanceMetrics();
        
        // Component initialization helpers
        bool initializeNeuralNetwork();
        bool initializeExperienceBuffer();
        bool initializeTrainer();
        bool initializeStateEncoder();
        bool initializeHazardDetector();
        bool initializeSurvivalRewards();
        bool initializeDeathAnalyzer();
    };
    
    // Global convenience functions
    inline ZombieAISystem& GetAI() { return ZombieAISystem::getInstance(); }
    inline bool IsAIEnabled() { return GetAI().getMode() != AIMode::DISABLED; }
    inline bool IsLearning() { return GetAI().getMode() == AIMode::ONLINE_LEARNING; }
    
} // namespace ZombieAI

// Integration macros for easy use in existing YaPB code
#define ZOMBIE_AI ZombieAI::GetAI()
#define ZOMBIE_AI_DECISION(bot) ZombieAI::GetAI().makeDecision(bot)
#define ZOMBIE_AI_RECORD(bot, action, reward) ZombieAI::GetAI().recordExperience(bot, action, reward)
#define ZOMBIE_AI_DEATH(bot, killer) ZombieAI::GetAI().onBotDeath(bot, killer)
