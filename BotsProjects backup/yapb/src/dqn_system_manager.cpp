//
// DQN SYSTEM MANAGER - REPLACES BROKEN NEURAL AI
// Real Deep Q-Learning for YaPB bots
//

#include <yapb.h>
#include <dqn/deep_q_network.h>

namespace DQN {

    // DQN System Integration with YaPB
    class DQNSystemManager {
    private:
        bool m_system_initialized = false;
        bool m_training_enabled = true;
        float m_last_stats_time = 0.0f;
        
    public:
        void initialize() {
            if (!m_system_initialized) {
                initializeDQNSystem();
                m_system_initialized = true;
                
                game.print("🔥 DQN System Manager initialized!");
                game.print("🧠 Ready to create ACTUALLY LEARNING bots!");
            }
        }
        
        void shutdown() {
            if (m_system_initialized) {
                shutdownDQNSystem();
                m_system_initialized = false;
            }
        }
        
        void runFrame() {
            if (!m_system_initialized) return;
            
            // Run DQN learning for all bots
            runDQNSystemFrame();
            
            // Print stats occasionally
            if (game.time() - m_last_stats_time > 30.0f) {  // Every 30 seconds
                printDQNSystemStats();
                m_last_stats_time = game.time();
            }
        }
        
        void onBotAdd(Bot* bot) {
            if (m_system_initialized && m_training_enabled) {
                addDQNBot(bot);
            }
        }
        
        void onBotRemove(Bot* bot) {
            if (m_system_initialized) {
                removeDQNBot(bot);
            }
        }
        
        void setTrainingEnabled(bool enabled) {
            m_training_enabled = enabled;
            game.print("🎯 DQN Training %s", enabled ? "ENABLED" : "DISABLED");
        }
        
        bool isInitialized() const { return m_system_initialized; }
    };

    // Global DQN system manager
    static DQNSystemManager g_dqn_manager;

} // namespace DQN

// ================================
// REPLACEMENT FOR neural_zombie_ai.cpp
// ================================

void initializeDQNAI() {
    DQN::g_dqn_manager.initialize();
    
    game.print("🚀 REAL DQN AI SYSTEM LOADED!");
    game.print("🧠 Bots will now ACTUALLY LEARN through Deep Q-Learning!");
    game.print("📚 Features: Experience replay, target networks, epsilon-greedy exploration");
}

void shutdownDQNAI() {
    DQN::g_dqn_manager.shutdown();
    game.print("💀 DQN AI System shutdown");
}

void runDQNAIFrame() {
    DQN::g_dqn_manager.runFrame();
}

void addBotToDQN(Bot* bot) {
    DQN::g_dqn_manager.onBotAdd(bot);
}

void removeBotFromDQN(Bot* bot) {
    DQN::g_dqn_manager.onBotRemove(bot);
}

void setDQNTraining(bool enabled) {
    DQN::g_dqn_manager.setTrainingEnabled(enabled);
}

// Handle bot lifecycle events
void onDQNBotSpawn(Bot* bot) {
    // Find the DQN bot and notify it of spawn
    for (auto* dqn_bot : DQN::g_dqn_bots) {
        if (dqn_bot->m_yapb_bot == bot) {
            dqn_bot->onBotSpawn();
            break;
        }
    }
}

void onDQNBotDeath(Bot* bot) {
    // Find the DQN bot and notify it of death
    for (auto* dqn_bot : DQN::g_dqn_bots) {
        if (dqn_bot->m_yapb_bot == bot) {
            dqn_bot->onBotDeath();
            break;
        }
    }
}

// Console commands for DQN control
void printDQNStats() {
    DQN::printDQNSystemStats();
}

// Integration points for YaPB
extern "C" {
    // These functions should be called from YaPB's main loop
    void dqn_initialize() { initializeDQNAI(); }
    void dqn_shutdown() { shutdownDQNAI(); }
    void dqn_run_frame() { runDQNAIFrame(); }
    void dqn_add_bot(Bot* bot) { addBotToDQN(bot); }
    void dqn_remove_bot(Bot* bot) { removeBotFromDQN(bot); }
    void dqn_bot_spawn(Bot* bot) { onDQNBotSpawn(bot); }
    void dqn_bot_death(Bot* bot) { onDQNBotDeath(bot); }
    void dqn_print_stats() { printDQNStats(); }
}
