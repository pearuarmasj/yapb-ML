//
// Simple Hazard Detector - Environmental Danger Detection for de_survivor
// Detects and manages environmental hazards that cause instant death
//

#pragma once

#include <yapb.h>

namespace ZombieAI {
    namespace Survival {
        
        class SmartHazardDetector {
        public:
            SmartHazardDetector() {}
            ~SmartHazardDetector() {}
            
            // MAIN HAZARD ANALYSIS - combines all methods
            float analyzeHazards(const Bot* bot);
        };
        
        // Global instance declaration
        extern SmartHazardDetector g_smartHazardDetector;
        
    } // namespace Survival
} // namespace ZombieAI
