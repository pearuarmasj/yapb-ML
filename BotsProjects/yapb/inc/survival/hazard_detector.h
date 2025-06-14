//
// Hazard Detector - Environmental Danger Detection for de_survivor
// Detects and manages environmental hazards that cause instant death
//

#pragma once

#include <yapb.h>
#include <vector>
#include <map>

namespace ZombieAI {
    namespace Survival {
        
        // Types of environmental hazards
        enum class HazardType {
            WATER_DROWNING = 0,    // Frozen river areas
            FALL_DAMAGE,           // Chasm, breakable bridges  
            CRUSH_DAMAGE,          // Falling objects, closing doors
            FIRE_DAMAGE,           // Fire/lava areas
            TOXIC_DAMAGE,          // Poison/radiation zones
            OUT_OF_BOUNDS          // Map boundary violations
        };
        
        // Individual hazard zone definition
        struct HazardZone {
            HazardType type;
            Vector3 center;           // Center point of hazard
            float radius;             // Danger radius
            float severity;           // How dangerous (0.0-1.0)
            String description;       // Human-readable description
            bool isActive;           // Can be disabled/enabled
            
            // Optional bounding box for complex shapes
            Vector3 minBounds;
            Vector3 maxBounds;
            bool usesBoundingBox;
        };
        
        // Distance and danger calculation result
        struct HazardProximity {
            float distance;           // Distance to nearest hazard edge
            float dangerLevel;        // Calculated danger level (0.0-1.0)  
            HazardType nearestType;   // Type of nearest hazard
            Vector3 safeDirection;    // Direction to move to get safer
            bool isInDanger;         // Currently in immediate danger
        };
        
        class HazardDetector {
        private:
            std::vector<HazardZone> m_hazards;
            std::map<String, std::vector<HazardZone>> m_mapHazards; // Per-map hazard definitions
            String m_currentMap;
            
            // Performance optimization
            mutable std::vector<float> m_distanceCache;
            mutable int m_cacheFrameCount;
            static constexpr int CACHE_LIFETIME = 5; // Frames to cache calculations
            
        public:
            HazardDetector();
            ~HazardDetector();
            
            // Map-specific hazard loading
            bool loadMapHazards(const String& mapName);
            void setCurrentMap(const String& mapName);
            
            // Hazard zone management
            void addHazard(const HazardZone& hazard);
            void removeHazard(int index);
            void clearHazards();
            size_t getHazardCount() const { return m_hazards.size(); }
            
            // Danger detection
            HazardProximity checkPosition(const Vector3& position) const;
            float getDistanceToNearestHazard(const Vector3& position) const;
            std::vector<float> getDistancesToAllHazards(const Vector3& position) const;
            
            // Safety analysis  
            bool isPositionSafe(const Vector3& position, float safetyMargin = 50.0f) const;
            Vector3 findSafeDirection(const Vector3& position, const Vector3& currentVelocity) const;
            Vector3 findNearestSafePosition(const Vector3& position, float searchRadius = 200.0f) const;
            
            // Pathfinding integration
            float getHazardPenalty(const Vector3& position) const;
            bool isPathSafe(const Vector3& start, const Vector3& end, int checkPoints = 10) const;
            
            // Configuration and debugging
            const std::vector<HazardZone>& getHazards() const { return m_hazards; }
            void enableHazard(int index, bool enabled);
            void setHazardSeverity(int index, float severity);
            
            // Debug visualization (for development)
            void drawDebugHazards() const;
            void printHazardInfo() const;
            
        private:
            void initializeDefaultHazards();
            void loadDeSurvivorHazards();
            float calculateDangerLevel(const Vector3& position, const HazardZone& hazard) const;
            bool isPointInHazard(const Vector3& position, const HazardZone& hazard) const;
        };
        
    } // namespace Survival
} // namespace ZombieAI
