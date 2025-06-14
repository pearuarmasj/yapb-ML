//
// Death Analyzer - Track and Analyze Bot Death Causes
// Monitors how bots die to improve reward engineering and learning
//

#pragma once

#include <yapb.h>
#include <vector>
#include <map>
#include <chrono>

namespace ZombieAI {
    namespace Survival {
        
        // Detailed death cause classification
        enum class DeathCause {
            // Environmental deaths (highest priority to prevent)
            DROWNING = 0,
            FALL_DAMAGE,
            CRUSH_DAMAGE,
            FIRE_DAMAGE,
            TOXIC_DAMAGE,
            OUT_OF_BOUNDS,
            
            // Combat deaths
            ENEMY_GUNFIRE,
            ENEMY_KNIFE,
            ENEMY_GRENADE,
            ENEMY_INFECTION,
            
            // Other causes
            SUICIDE,
            TEAM_KILL,
            UNKNOWN,
            DISCONNECT
        };
        
        // Individual death event record
        struct DeathEvent {
            int botId;
            DeathCause cause;
            Vector3 deathPosition;
            Vector3 lastSafePosition;
            float survivalTime;           // How long bot survived this round
            float distanceToHazard;       // Distance to nearest hazard at death
            String killerName;            // Who/what killed the bot
            String mapName;
            std::chrono::system_clock::time_point timestamp;
            
            // Context information
            int health;                   // Health at time of death
            int armor;                    // Armor at time of death
            String weaponUsed;            // Weapon that caused death
            bool wasStuck;               // Was bot stuck when died?
            bool wasInCombat;            // Was bot in active combat?
            std::vector<String> nearbyPlayers; // Other players nearby
        };
        
        // Death statistics for analysis
        struct DeathStats {
            std::map<DeathCause, int> deathCounts;
            std::map<DeathCause, float> avgSurvivalTime;
            std::map<DeathCause, std::vector<Vector3>> deathLocations;
            
            int totalDeaths;
            float avgSurvivalTime;
            float environmentalDeathRate; // Percentage of deaths that are environmental
            
            // Learning progress indicators
            bool isImprovingOverTime;
            float recentDeathRate;        // Deaths per minute in recent games
            std::vector<String> mostProblematicAreas; // Map areas with most deaths
        };
        
        class DeathAnalyzer {
        private:
            std::vector<DeathEvent> m_deathHistory;
            std::map<int, DeathStats> m_botStats;
            std::map<String, std::vector<DeathEvent>> m_mapDeaths; // Deaths per map
            
            // Analysis configuration
            size_t m_maxHistorySize;
            float m_recentTimeWindow;     // Seconds to consider "recent"
            
            // Real-time tracking
            std::map<int, Vector3> m_botLastSafePositions;
            std::map<int, std::chrono::system_clock::time_point> m_botRoundStartTimes;
            
        public:
            DeathAnalyzer(size_t maxHistory = 10000, float recentWindow = 300.0f);
            ~DeathAnalyzer();
            
            // Death event recording
            void recordDeath(const Bot* bot, DeathCause cause, const String& killerName = "");
            void recordEnvironmentalDeath(const Bot* bot, DeathCause cause, const Vector3& hazardLocation);
            
            // Real-time tracking updates
            void updateBotPosition(const Bot* bot);
            void onRoundStart(const Bot* bot);
            void onMapChange(const String& mapName);
            
            // Death cause detection (automatic classification)
            DeathCause classifyDeath(const Bot* bot, const String& killerName);
            DeathCause detectEnvironmentalCause(const Bot* bot);
            
            // Statistics and analysis
            DeathStats getBotStats(int botId) const;
            DeathStats getOverallStats() const;
            DeathStats getMapStats(const String& mapName) const;
            
            // Learning progress analysis
            bool isBotImproving(int botId, int recentGames = 10) const;
            float getEnvironmentalDeathTrend(int botId, int windowSize = 50) const;
            std::vector<Vector3> getProblematicLocations(const String& mapName, DeathCause cause) const;
            
            // Query functions
            std::vector<DeathEvent> getRecentDeaths(int botId, float timeWindow = 60.0f) const;
            std::vector<DeathEvent> getDeathsByType(int botId, DeathCause cause) const;
            int getDeathCount(int botId, DeathCause cause) const;
            
            // Reward integration
            float getRepeatDeathPenalty(int botId, const Vector3& position, float radius = 50.0f) const;
            bool isRepeatMistake(int botId, const Vector3& position, DeathCause cause) const;
            
            // Configuration
            void setMaxHistorySize(size_t maxSize);
            void setRecentTimeWindow(float seconds);
            void clearHistory();
            void clearBotHistory(int botId);
            
            // Debug and reporting
            void printDeathSummary(int botId) const;
            void printMapDeathAnalysis(const String& mapName) const;
            String getDeathCauseString(DeathCause cause) const;
            
            // Export for external analysis
            std::vector<DeathEvent> exportDeathHistory() const;
            void saveDeathReport(const String& filename) const;
        };
        
    } // namespace Survival
} // namespace ZombieAI
