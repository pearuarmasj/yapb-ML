//
// Survival Rewards - Survival-Focused Reward Calculation
// Calculates rewards and penalties for zombie survival scenarios
//

#pragma once

#include <yapb.h>
#include <survival/hazard_detector.h>
#include <survival/death_analyzer.h>

namespace ZombieAI {
    namespace Survival {
        
        // Reward categories and weights
        enum class RewardType {
            SURVIVAL_TIME = 0,     // Staying alive
            HAZARD_AVOIDANCE,      // Avoiding environmental dangers
            MOVEMENT_EFFICIENCY,   // Smart movement patterns
            COMBAT_PERFORMANCE,    // Zombie hunting/human survival
            TEAM_COORDINATION,     // Working with teammates
            OBJECTIVE_COMPLETION,  // Map-specific objectives
            LEARNING_PROGRESS      // Improvement over time
        };
        
        // Individual reward calculation result
        struct RewardComponent {
            RewardType type;
            float value;           // Raw reward value
            float weight;          // Importance multiplier
            float finalValue;      // value * weight
            String reason;         // Why this reward was given
        };
        
        // Complete reward calculation for one frame
        struct SurvivalReward {
            std::vector<RewardComponent> components;
            float totalReward;     // Sum of all weighted components
            float immediateReward; // This frame only
            float cumulativeReward; // Running total
            bool wasTerminal;      // Did episode end?
            String summary;        // Human-readable summary
        };
        
        // Reward calculation configuration
        struct RewardConfig {
            // Survival rewards
            float survivalBonus = 1.0f;              // +1 per second alive
            float environmentalDeathPenalty = -1000.0f; // Massive penalty for environmental death
            float combatDeathPenalty = -100.0f;      // Smaller penalty for combat death
            
            // Hazard avoidance
            float hazardProximityPenalty = -10.0f;   // Per unit distance to hazard
            float safeMovementBonus = 5.0f;          // Moving away from danger
            float repeatMistakePenalty = -50.0f;     // Repeating same environmental mistake
            
            // Movement and positioning
            float efficientMovementBonus = 2.0f;     // Good pathfinding
            float stuckPenalty = -5.0f;              // Being stuck/immobile
            float explorationBonus = 1.0f;           // Exploring new areas safely
            
            // Combat and objectives (zombie mode specific)
            float killReward = 50.0f;                // Successful kill
            float damageReward = 1.0f;               // Damage dealt per point
            float infectionBonus = 100.0f;           // Successful infection (zombies)
            float rescueBonus = 200.0f;              // Saving teammate (humans)
            
            // Learning and adaptation
            float improvementBonus = 10.0f;          // Performance improvement over time
            float consistencyBonus = 5.0f;           // Consistent good performance
        };
        
        class SurvivalRewards {
        private:
            HazardDetector* m_hazardDetector;
            DeathAnalyzer* m_deathAnalyzer;
            RewardConfig m_config;
            
            // Per-bot tracking for learning rewards
            std::map<int, float> m_botCumulativeRewards;
            std::map<int, std::vector<float>> m_botRewardHistory;
            std::map<int, int> m_botEnvironmentalDeaths;
            
            // Performance tracking
            std::map<int, Vector3> m_botLastPositions;
            std::map<int, float> m_botLastHazardDistances;
            std::map<int, float> m_botSurvivalTimes;
            
        public:
            SurvivalRewards(HazardDetector* hazardDetector = nullptr, DeathAnalyzer* deathAnalyzer = nullptr);
            ~SurvivalRewards();
            
            // Main reward calculation
            SurvivalReward calculateReward(const Bot* bot);
            SurvivalReward calculateDeathReward(const Bot* bot, DeathCause cause);
            
            // Individual reward components
            RewardComponent calculateSurvivalReward(const Bot* bot);
            RewardComponent calculateHazardAvoidanceReward(const Bot* bot);
            RewardComponent calculateMovementReward(const Bot* bot);
            RewardComponent calculateCombatReward(const Bot* bot);
            RewardComponent calculateLearningReward(const Bot* bot);
            
            // Reward shaping and balancing
            float shapeReward(float rawReward, RewardType type);
            void updateRewardWeights(const std::map<RewardType, float>& weights);
            
            // Configuration
            void setConfig(const RewardConfig& config) { m_config = config; }
            RewardConfig getConfig() const { return m_config; }
            void setHazardDetector(HazardDetector* detector) { m_hazardDetector = detector; }
            void setDeathAnalyzer(DeathAnalyzer* analyzer) { m_deathAnalyzer = analyzer; }
            
            // Statistics and analysis
            float getBotCumulativeReward(int botId) const;
            std::vector<float> getBotRewardHistory(int botId) const;
            float getBotAverageReward(int botId, int windowSize = 100) const;
            int getBotEnvironmentalDeaths(int botId) const;
            
            // Reset and cleanup
            void resetBotStats(int botId);
            void clearAllStats();
            
            // Debug and monitoring
            void printRewardBreakdown(const SurvivalReward& reward) const;
            String getRewardSummary(int botId) const;
        };
        
    } // namespace Survival
} // namespace ZombieAI
