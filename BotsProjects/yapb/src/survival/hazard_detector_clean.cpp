//
// Smart Hazard Detection Implementation for de_survivor
// Learns dangerous patterns rather than just avoiding death zones
//

#include <yapb.h>
#include <survival/hazard_detector_simple.h>
#include <vector>
#include <map>

namespace ZombieAI {
    namespace Survival {
        
        // de_survivor specific hazard zones - static data
        static struct DeSurvivorHazards {
            Vector bridgeGapSide1 = Vector(374.0f, 833.0f, 85.0f);
            Vector bridgeGapSide2 = Vector(386.0f, 892.0f, 85.0f);
            Vector frozenRiverSide1 = Vector(226.0f, 1917.0f, 85.0f);
            Vector frozenRiverSide2 = Vector(-1017.0f, 1923.0f, 85.0f);
        } g_hazards;
          // MAIN HAZARD ANALYSIS IMPLEMENTATION
        float SmartHazardDetector::analyzeHazards(const Bot* bot) {
            float totalPenalty = 0.0f;
            Vector botPos = bot->pev->origin;
            
            // Method 1: Bridge gap detection - LEARN TO JUMP
            Vector gapCenter = (g_hazards.bridgeGapSide1 + g_hazards.bridgeGapSide2) * 0.5f;
            float distToGap = (botPos - gapCenter).length();
            
            if (distToGap < 100.0f) {
                // If very close to gap - check if bot is jumping
                if (distToGap < 30.0f) {
                    if (bot->pev->button & IN_JUMP) {
                        totalPenalty += 50.0f; // Reward for jumping at bridge
                    } else {
                        totalPenalty -= 200.0f; // Massive penalty for not jumping
                    }
                }
            }
            
            // Method 2: Frozen river - AVOID ENTIRELY  
            Vector riverCenter = (g_hazards.frozenRiverSide1 + g_hazards.frozenRiverSide2) * 0.5f;
            float riverWidth = g_hazards.frozenRiverSide1.distance(g_hazards.frozenRiverSide2);
            float distToRiver = botPos.distance(riverCenter);
            
            if (distToRiver < riverWidth * 0.6f) {
                // Massive penalty for approaching frozen river
                float proximityPenalty = (1.0f - (distToRiver / (riverWidth * 0.6f))) * -500.0f;
                totalPenalty += proximityPenalty;
            }
            
            return totalPenalty;
        }
        
        // Global instance definition
        SmartHazardDetector g_smartHazardDetector;
        
    } // namespace Survival
} // namespace ZombieAI
