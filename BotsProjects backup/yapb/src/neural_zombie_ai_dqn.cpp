//
// DQN AI INTEGRATION - REPLACES neural_zombie_ai.cpp
// Real Deep Q-Learning system for YaPB
//

#include <yapb.h>
#include <dqn/deep_q_network.h>

// ConVars for compatibility with existing YaPB code
ConVar cv_neural_data_collection ("neural_data_collection", "1", "DQN learning system - always enabled");
ConVar cv_neural_use_for_decisions ("neural_use_for_decisions", "1", "DQN decision making - always enabled");

// Forward declarations for DQN system
extern "C" {
    void dqn_initialize();
    void dqn_shutdown();
    void dqn_run_frame();
    void dqn_add_bot(Bot* bot);
    void dqn_remove_bot(Bot* bot);
    void dqn_bot_spawn(Bot* bot);
    void dqn_bot_death(Bot* bot);
    void dqn_print_stats();
}

// Add missing Bot methods for compatibility
void Bot::neuralZombieLogic() {
    // This replaces the old broken neural logic with DQN calls
    // Call the DQN system for this bot if it's in the DQN system
    for (auto* dqn_bot : DQN::g_dqn_bots) {
        if (dqn_bot->m_yapb_bot == this) {
            dqn_bot->runDQNFrame();
            return;
        }
    }
    
    // If bot not in DQN system yet, add it
    dqn_add_bot(this);
}

void Bot::shutdownNeuralSystem() {
    dqn_shutdown();
}

// Replace all the broken neural AI with REAL DQN
void initializeNeuralZombieAI() {
    dqn_initialize();
    game.print("🔥 REAL DQN AI REPLACING BROKEN NEURAL SYSTEM!");
}

void shutdownNeuralZombieAI() {
    dqn_shutdown();
}

void runNeuralZombieAI() {
    dqn_run_frame();
}

void addBotToNeuralAI(Bot* bot) {
    dqn_add_bot(bot);
}

void removeBotFromNeuralAI(Bot* bot) {
    dqn_remove_bot(bot);
}

void onNeuralBotSpawn(Bot* bot) {
    dqn_bot_spawn(bot);
}

void onNeuralBotDeath(Bot* bot) {
    dqn_bot_death(bot);
}

void printNeuralAIStats() {
    dqn_print_stats();
}

// Compatibility functions for old neural AI interface
void trainNeuralAI() {
    game.print("🧠 DQN training is always active - no manual training needed!");
}

void saveNeuralModel() {
    game.print("💾 DQN model saving not yet implemented");
}

void loadNeuralModel() {
    game.print("📂 DQN model loading not yet implemented");
}
