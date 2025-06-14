//
// PURE NEURAL AI - NO YAPB INTELLIGENCE! 
// Bots start as complete idiots and learn through pain and exploration
//

#include <yapb.h>
#include <pure_neural/pure_neural_bot.h>

// ConVar declarations for pure neural system
ConVar cv_pure_neural_enabled ("pure_neural_enabled", "1", "Enable pure neural learning system (no YaPB intelligence)");
ConVar cv_pure_neural_debug ("pure_neural_debug", "1", "Show neural learning debug output");
ConVar cv_neural_use_for_decisions ("neural_use_for_decisions", "1", "Use neural network for decisions (always true for pure neural system).");
ConVar cv_neural_data_collection ("neural_data_collection", "0", "LEGACY: Run neural + YaPB AI together (NOT for pure neural system - keep at 0!)");

// Global neural system state
static bool g_neural_system_initialized = false;

// ================================
// PURE NEURAL SYSTEM INITIALIZATION
// ================================

static void initializeNeuralSystem() {
    if (!g_neural_system_initialized && cv_pure_neural_enabled.as<bool>()) {
        PureNeural::initializePureNeuralSystem();
        g_neural_system_initialized = true;
        
        game.print("🧠 PURE NEURAL AI SYSTEM ACTIVE!");
        game.print("🤖 Bots will start as complete fucking idiots!");
        game.print("💀 They will learn through pain, death, and exploration!");
        game.print("🎯 NO YaPB pathfinding or intelligence assistance!");
    }
}

static void shutdownNeuralSystem() {
    if (g_neural_system_initialized) {
        PureNeural::shutdownPureNeuralSystem();
        g_neural_system_initialized = false;
    }
}

// ================================
// BOT NEURAL LOGIC - REPLACES ALL YAPB INTELLIGENCE!
// ================================

void Bot::neuralZombieLogic() {
    // CHECK IF PURE NEURAL SYSTEM IS ENABLED!
    if (!cv_pure_neural_enabled.as<bool>()) {
        return; // Pure neural system disabled - do nothing
    }
    
    // Initialize system if needed
    if (!g_neural_system_initialized) {
        initializeNeuralSystem("", "");  // Empty paths - pure neural doesn't need external data
    }
    
    // Check if this bot has a neural brain
    PureNeural::PureNeuralBot* neural_bot = nullptr;
    for (auto* bot : PureNeural::g_neural_bots) {
        if (bot->getYaPBBot() == this) {
            neural_bot = bot;
            break;
        }
    }
    
    // If no neural brain, create one (bot becomes complete idiot)
    if (!neural_bot) {
        PureNeural::addPureNeuralBot(this);
        game.print("🤖 Bot '%s' now has a neural brain - Starting as complete idiot!", 
                  pev->netname.chars());
        return;
    }
    
    // RUN PURE NEURAL FRAME - NO YAPB INTELLIGENCE!
    neural_bot->runNeuralFrame();
    
    // Debug output (rate limited)
    if (cv_pure_neural_debug.as<bool>()) {
        static float last_global_debug = 0.0f;
        if (game.time() - last_global_debug > 10.0f) {
            game.print("🧠 PURE NEURAL SYSTEM: %d bots learning through pain!", 
                      (int)PureNeural::g_neural_bots.size());
            last_global_debug = game.time();
        }
    }
}

// ================================
// DUMMY FUNCTIONS - NOT USED IN PURE NEURAL!
// ================================

ZombieTrainingState Bot::getCurrentState() {
    // Not used in pure neural system
    ZombieTrainingState state;
    return state;
}

ZombieTrainingReward Bot::calculateReward() {
    // Not used in pure neural system  
    ZombieTrainingReward reward;
    return reward;
}

void Bot::executeNeuralAction(const ZombieTrainingAction &action) {
    // Not used in pure neural system
}

ZombieTrainingAction Bot::predictNeuralAction() {
    // Not used in pure neural system
    ZombieTrainingAction action;
    return action;
}

ZombieTrainingAction Bot::getCurrentAction() {
    // Not used in pure neural system
    ZombieTrainingAction action;
    return action;
}

bool Bot::shouldUseNeuralNetwork() {
    // Always true for pure neural system
    return true;
}

void Bot::printNeuralStats() {
    // Not used in pure neural system
}

float Bot::getNeuralConfidence() const {
    // Always 1.0 for pure neural system
    return 1.0f;
}

void Bot::executeNeuralAction(const Vector &action) {
    // Not used in pure neural system - we use PureNeural system
}

// Static neural system management functions
void Bot::shutdownNeuralSystem() {
    // Cleanup pure neural system
    PureNeural::clearAllNeuralBots();
    g_neural_system_initialized = false;
}

void Bot::startDataCollection() {
    // Pure neural system doesn't need external data collection
}

void Bot::stopDataCollection() {
    // Pure neural system doesn't need external data collection
}

bool Bot::isNeuralSystemEnabled() {
    return g_neural_system_initialized;
}

// Initialize neural system (static function)
void Bot::initializeNeuralSystem(const String &dataDir, const String &modelPath) {
    if (g_neural_system_initialized) {
        return; // Already initialized
    }
    
    // Initialize pure neural system
    PureNeural::initializePureNeuralSystem();
    g_neural_system_initialized = true;
    
    game.print("🧠 Pure Neural AI System initialized - Bots will start as complete idiots!");
}
