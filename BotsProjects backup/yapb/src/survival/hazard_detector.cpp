//
// Smart Hazard Detection for de_survivor
// Learns dangerous patterns rather than just avoiding death zones
//

#include <yapb.h>
#include <survival/hazard_detector_simple.h>
#include <vector>
#include <map>

namespace ZombieAI {
    namespace Survival {
          // de_survivor specific hazard zones
        struct DeSurvivorHazards {
            // Bridge gap coordinates (EXACT user-provided coordinates)
            Vector bridgeGapSide1 = Vector(374.0f, 833.0f, 85.0f);
            Vector bridgeGapSide2 = Vector(386.0f, 892.0f, 85.0f);
            
            // Frozen river zone (AVOID entirely) - exact coordinates
            Vector frozenRiverSide1 = Vector(226.0f, 1917.0f, 85.0f);
            Vector frozenRiverSide2 = Vector(-1017.0f, 1923.0f, 85.0f);
        };
        
        class SmartHazardDetector {
        private:
            DeSurvivorHazards m_hazards;
            
            // Death learning - track WHERE bots die
            std::vector<Vector> m_deathLocations;
            std::map<int, Vector> m_botLastSafePositions; // Per-bot safe position tracking
            
            // Environmental detection
            std::map<int, bool> m_botNearWater;
            std::map<int, bool> m_botNearLedge;
            
        public:
            SmartHazardDetector() {}
            ~SmartHazardDetector() {}
              // MAIN HAZARD ANALYSIS - combines all methods
            float analyzeHazards(const Bot* bot) {
                float totalPenalty = 0.0f;
                Vector botPos = bot->pev->origin;
                
                // Method 1: Bridge gap detection - LEARN TO JUMP
                float bridgePenalty = checkBridgeGap(botPos, bot->pev->velocity);
                totalPenalty += bridgePenalty;
                
                // Method 2: Frozen river - AVOID ENTIRELY  
                float riverPenalty = checkFrozenRiver(botPos);
                totalPenalty += riverPenalty;
                
                // Method 3: Death zone learning - AVOID PAST DEATH SPOTS
                float deathZonePenalty = checkDeathZones(botPos);
                totalPenalty += deathZonePenalty;
                
                // Method 4: Real-time environmental danger
                float environmentPenalty = checkEnvironmentalDanger(bot);
                totalPenalty += environmentPenalty;
                
                return totalPenalty;
            }
            
            // Bridge gap handling - SMART JUMPING
            float checkBridgeGap(const Vector& botPos, const Vector& velocity) {
                float distToGap = getDistanceToBridgeGap(botPos);
                
                // If approaching bridge gap
                if (distToGap < 100.0f) {
                    Vector gapCenter = (m_hazards.bridgeGapSide1 + m_hazards.bridgeGapSide2) * 0.5f;
                    Vector directionToGap = (gapCenter - botPos).normalize();
                    Vector velocityDirection = velocity.normalize();
                    
                    float movingTowardGap = directionToGap.dot(velocityDirection);
                    
                    // If moving toward gap and close
                    if (movingTowardGap > 0.7f && distToGap < 50.0f) {
                        // POSITIVE reward for jumping when approaching gap
                        // This teaches: "when near bridge gap + moving toward it = JUMP"
                        return 20.0f; // Positive reward for good timing
                    }
                    
                    // Small negative if just standing near gap doing nothing
                    if (velocity.length() < 10.0f) {
                        return -5.0f; // "Don't just stand near the gap"
                    }
                }
                
                return 0.0f;
            }
            
            // Frozen river - TOTAL AVOIDANCE
            float checkFrozenRiver(const Vector& botPos) {
                Vector riverCenter = (m_hazards.frozenRiverSide1 + m_hazards.frozenRiverSide2) * 0.5f;
                float riverWidth = m_hazards.frozenRiverSide1.distance(m_hazards.frozenRiverSide2);
                
                float distToRiver = botPos.distance(riverCenter);
                float dangerRadius = riverWidth * 0.6f; // 60% of river width
                
                if (distToRiver < dangerRadius) {
                    // MASSIVE penalty - get the fuck away from frozen river
                    float proximityFactor = 1.0f - (distToRiver / dangerRadius);
                    return -100.0f * proximityFactor; // Up to -100 points
                }
                
                return 0.0f;
            }
            
            // Death zone learning - avoid where bots previously died
            float checkDeathZones(const Vector& botPos) {
                float penalty = 0.0f;
                
                for (const Vector& deathLoc : m_deathLocations) {
                    float distToDeath = botPos.distance(deathLoc);
                    
                    // Small penalty for being near previous death locations
                    if (distToDeath < 80.0f) {
                        penalty -= 10.0f * (1.0f - distToDeath / 80.0f);
                    }
                }
                
                return penalty;
            }
            
            // Real-time environmental danger detection
            float checkEnvironmentalDanger(const Bot* bot) {
                float penalty = 0.0f;
                
                // Check if bot is stuck (environmental problem)
                if (bot->m_isStuck) {
                    penalty -= 20.0f; // "Being stuck is bad"
                }
                
                // Check if bot is in water (could be drowning)
                if (bot->pev->waterlevel > 1) {
                    penalty -= 50.0f; // "Deep water is very bad"
                }
                
                // Check if bot is moving very slowly (might be stuck on geometry)
                float speed = bot->pev->velocity.length();
                if (speed < 5.0f && bot->m_moveSpeed > 100.0f) {
                    penalty -= 15.0f; // "Trying to move but can't"
                }
                
                return penalty;
            }
            
            // DEATH LEARNING - record when bot dies for future avoidance
            void recordBotDeath(const Bot* bot, const String& cause) {
                Vector deathPos = bot->pev->origin;
                m_deathLocations.push_back(deathPos);
                
                // Keep only recent deaths (last 100)
                if (m_deathLocations.size() > 100) {
                    m_deathLocations.erase(m_deathLocations.begin());
                }
                
                game.print("💀 Bot %s died at (%.0f, %.0f, %.0f) - cause: %s", 
                          bot->pev->netname.chars(), deathPos.x, deathPos.y, deathPos.z, cause.chars());
            }
            
            // SAFE POSITION TRACKING - remember last safe position
            void updateBotPosition(const Bot* bot) {
                Vector currentPos = bot->pev->origin;
                int botId = bot->index();
                
                // If bot is safe (not near hazards), update safe position
                if (analyzeHazards(bot) >= -5.0f) { // Only small penalties or positive
                    m_botLastSafePositions[botId] = currentPos;
                }
            }
            
            // UTILITY FUNCTIONS
            float getDistanceToBridgeGap(const Vector& pos) {
                Vector gapCenter = (m_hazards.bridgeGapSide1 + m_hazards.bridgeGapSide2) * 0.5f;
                return pos.distance(gapCenter);
            }
            
            bool isNearBridgeGap(const Vector& pos, float radius = 100.0f) {
                return getDistanceToBridgeGap(pos) < radius;
            }
            
            bool isNearFrozenRiver(const Vector& pos, float radius = 200.0f) {
                Vector riverCenter = (m_hazards.frozenRiverSide1 + m_hazards.frozenRiverSide2) * 0.5f;
                return pos.distance(riverCenter) < radius;
            }
            
            // DEBUG OUTPUT
            void printHazardDebug(const Bot* bot) {
                Vector pos = bot->pev->origin;                game.print("🔍 Bot %s hazard analysis:", bot->pev->netname.chars());
                game.print("  Bridge gap distance: %.1f", getDistanceToBridgeGap(pos));
                game.print("  Near frozen river: %s", isNearFrozenRiver(pos) ? "YES" : "no");
                game.print("  Total penalty: %.1f", analyzeHazards(bot));
            }
        };
        
        // Global instance definition
        SmartHazardDetector g_smartHazardDetector;
        
    } // namespace Survival
} // namespace ZombieAI
