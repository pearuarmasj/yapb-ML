//
// State Encoder - Convert Game State to Neural Network Input
// Transforms CS 1.6 game data into normalized neural network features
//

#pragma once

#include <yapb.h>
#include <ml/neural_network.h>
#include <survival/hazard_detector.h>
#include <vector>

namespace ZombieAI {
    namespace ML {
        
        // Input feature categories
        enum class FeatureCategory {
            POSITION = 0,      // Bot position, velocity, angles
            HEALTH,            // Health, armor, status effects
            ENVIRONMENT,       // Hazards, map features, ladders, water
            ENEMIES,           // Enemy positions, visibility, distances
            TEAMMATES,         // Team information, coordination
            WEAPONS,           // Current weapon, ammo, equipment
            GAME_STATE,        // Round time, game mode, objectives
            HISTORY            // Recent actions, performance metrics
        };
        
        // Feature normalization ranges
        struct FeatureRange {
            float minValue;
            float maxValue;
            bool isNormalized;
        };
        
        class StateEncoder {
        private:
            // Feature configuration
            std::vector<FeatureRange> m_featureRanges;
            std::vector<FeatureCategory> m_featureCategories;
            
            // Dependencies
            Survival::HazardDetector* m_hazardDetector;
            
            // Feature counts
            static constexpr int POSITION_FEATURES = 9;     // x,y,z, vel_x,vel_y,vel_z, yaw,pitch,roll
            static constexpr int HEALTH_FEATURES = 4;       // health, armor, stuck, status
            static constexpr int ENVIRONMENT_FEATURES = 8;  // hazard distances, ladder, water, etc.
            static constexpr int ENEMY_FEATURES = 12;       // enemy pos, health, visibility, etc.
            static constexpr int TEAMMATE_FEATURES = 6;     // teammate count, positions, etc.
            static constexpr int WEAPON_FEATURES = 4;       // weapon type, ammo, reload state
            static constexpr int GAME_STATE_FEATURES = 5;   // time, mode, objectives
            static constexpr int HISTORY_FEATURES = 8;      // recent performance metrics
            
        public:
            static constexpr int TOTAL_FEATURES = POSITION_FEATURES + HEALTH_FEATURES + 
                                                ENVIRONMENT_FEATURES + ENEMY_FEATURES + 
                                                TEAMMATE_FEATURES + WEAPON_FEATURES +
                                                GAME_STATE_FEATURES + HISTORY_FEATURES;
            
            StateEncoder(Survival::HazardDetector* hazardDetector = nullptr);
            ~StateEncoder();
            
            // Main encoding function
            NetworkInput encodeGameState(const Bot* bot);
            
            // Individual feature encoders
            std::vector<float> encodePosition(const Bot* bot);
            std::vector<float> encodeHealth(const Bot* bot);
            std::vector<float> encodeEnvironment(const Bot* bot);
            std::vector<float> encodeEnemies(const Bot* bot);
            std::vector<float> encodeTeammates(const Bot* bot);
            std::vector<float> encodeWeapons(const Bot* bot);
            std::vector<float> encodeGameState(const Bot* bot);
            std::vector<float> encodeHistory(const Bot* bot);
            
            // Feature normalization
            std::vector<float> normalizeFeatures(const std::vector<float>& features, FeatureCategory category);
            float normalizeValue(float value, float minVal, float maxVal);
            
            // Configuration
            void setHazardDetector(Survival::HazardDetector* detector);
            void updateFeatureRanges(const std::vector<FeatureRange>& ranges);
            
            // Analysis and debugging
            std::vector<String> getFeatureNames() const;
            void printFeatureDebug(const NetworkInput& input, const Bot* bot);
            
            // Validation
            bool validateInput(const NetworkInput& input) const;
            NetworkInput clampInput(const NetworkInput& input) const;
        };
        
    } // namespace ML
} // namespace ZombieAI
