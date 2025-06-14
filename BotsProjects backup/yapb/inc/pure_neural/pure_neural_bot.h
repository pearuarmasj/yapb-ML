//
// PURE NEURAL LEARNING BOT - NO YAPB INTELLIGENCE!
// Bots start as complete idiots and learn through pain
//

#pragma once

#include <yapb.h>
#include <vector>
#include <array>

namespace PureNeural {

    // RAW GAME STATE - NO PATHFINDING HELP!
    struct GameState {
        // Position & orientation (normalized)
        float pos_x, pos_y, pos_z;
        float angle_yaw, angle_pitch;
        
        // Health & survival
        float health;           // 0.0 - 1.0
        float alive_time;       // Seconds since spawn
        
        // Environmental sensors (NO cheating with YaPB data!)
        float wall_forward;     // Distance to wall ahead (0.0 - 1.0)
        float wall_left;        // Distance to wall left
        float wall_right;       // Distance to wall right  
        float ground_below;     // Distance to ground below
        
        // Vision rays - basic obstacle detection (8 directions)
        std::array<float, 8> vision_rays;  // 0° 45° 90° 135° 180° 225° 270° 315°
        
        // Movement state
        float velocity_x, velocity_y, velocity_z;
        bool on_ground;
        bool in_water;
    };

    // RAW ACTIONS - PURE NEURAL OUTPUT
    struct ActionOutput {
        float move_forward;     // -1.0 to +1.0 (back/forward)
        float move_right;       // -1.0 to +1.0 (left/right)  
        float turn_yaw;         // -1.0 to +1.0 (turn left/right)
        float turn_pitch;       // -1.0 to +1.0 (look down/up)
        float jump;             // 0.0 to 1.0 (jump probability)
        float crouch;           // 0.0 to 1.0 (crouch probability)
    };    // LEARNING FEEDBACK - PAIN AND REWARD
    struct LearningReward {
        float total_reward = 0.0f;
        
        // Death penalties (MASSIVE PAIN!)
        float death_penalty = 0.0f;        // -1000.0f when bot dies
        float fall_damage_penalty = 0.0f;  // -100.0f for falling damage
        float stuck_penalty = 0.0f;        // -10.0f for not moving
        float wall_collision_penalty = 0.0f; // -20.0f for hitting walls
        
        // Anti-spinning penalties
        float spinning_penalty = 0.0f;     // -5.0f for spinning without moving
        
        // Movement encouragement
        float movement_attempt_reward = 0.0f;   // +2.0f for successful movement
        float movement_attempt_penalty = 0.0f;  // -1.0f for failed movement attempts
        
        // Exploration rewards (small encouragement)
        float movement_reward = 0.0f;      // +1.0f for moving
        float new_area_reward = 0.0f;      // +50.0f for discovering new areas
        float survival_reward = 0.0f;      // +0.1f per second alive
        float height_exploration_reward = 0.0f; // +10.0f for reaching higher ground
        
        // Learning stats
        bool just_died = false;
        bool discovered_new_area = false;
        bool made_progress = false;
        float exploration_distance = 0.0f;
    };

    // PURE NEURAL NETWORK (Simple Feed-Forward)
    class SimpleNeuralNetwork {
    private:
        static const int INPUT_SIZE = 20;   // GameState inputs
        static const int HIDDEN_SIZE = 64;  // Hidden layer neurons
        static const int OUTPUT_SIZE = 6;   // ActionOutput values
        
        // Network weights (randomly initialized = complete idiot)
        std::vector<std::vector<float>> weights_input_hidden;
        std::vector<std::vector<float>> weights_hidden_output;
        std::vector<float> bias_hidden;
        std::vector<float> bias_output;
          // Learning parameters - MUCH MORE AGGRESSIVE!
        float learning_rate = 0.1f;  // 100x larger for faster learning
        float momentum = 0.9f;       // Memory to build consistent behavior
        float exploration_rate = 0.3f;  // 30% random actions to break local minima
        
        // Memory for learning and momentum
        std::vector<float> last_input;
        std::vector<float> last_hidden;
        std::vector<float> last_output;
        std::vector<std::vector<float>> velocity_input_hidden;    // Momentum vectors
        std::vector<std::vector<float>> velocity_hidden_output;
        
    public:
        SimpleNeuralNetwork();
        ~SimpleNeuralNetwork() {}
        
        // Forward pass - convert game state to actions
        ActionOutput processGameState(const GameState& state);
        
        // Backward pass - learn from pain/reward
        void learnFromReward(const LearningReward& reward);
        
        // Network management
        void randomizeWeights();  // Make bot completely stupid again
        void saveWeights(const char* filename);
        void loadWeights(const char* filename);
        
        // Mutation for evolution
        void mutateWeights(float mutation_rate = 0.1f);
    };    // PURE NEURAL BOT - REPLACES YAPB INTELLIGENCE
    class PureNeuralBot {
    public:
        Bot* m_yapb_bot;  // Only for game interface, NOT intelligence!
        
    private:
        SimpleNeuralNetwork m_brain;
        
        // Learning state
        GameState m_current_state;
        ActionOutput m_current_actions;
        LearningReward m_current_reward;        // Exploration tracking
        std::vector<Vector> m_visited_positions;
        Vector m_last_position;
        Vector m_last_angles;              // Track angle changes for anti-spinning
        float m_stuck_timer = 0.0f;
        float m_last_movement_time = 0.0f; // When did they last move horizontally
        float m_last_health = 100.0f;
        float m_spawn_time = 0.0f;
        
        // Performance stats
        int m_death_count = 0;
        float m_total_alive_time = 0.0f;
        float m_total_exploration_distance = 0.0f;
        int m_areas_discovered = 0;
        
    public:
        PureNeuralBot(Bot* yapb_bot);
        ~PureNeuralBot() {}
        
        // Access helper
        Bot* getYaPBBot() const { return m_yapb_bot; }
        
        // MAIN NEURAL LOOP - REPLACES ALL YAPB LOGIC!
        void runNeuralFrame();
        
        // State processing
        GameState extractGameState();
        void applyNeuralActions(const ActionOutput& actions);
        LearningReward calculateLearningReward();
        
        // Learning management
        void onBotDeath();
        void onBotSpawn(); 
        void saveProgress(const char* filename);
        void loadProgress(const char* filename);
        
        // Debug and monitoring
        void printLearningStats();
        bool isLearningProgress();
    };

    // GLOBAL PURE NEURAL SYSTEM
    extern std::vector<PureNeuralBot*> g_neural_bots;
      // System management
    void initializePureNeuralSystem();
    void shutdownPureNeuralSystem();
    void addPureNeuralBot(Bot* yapb_bot);
    void removePureNeuralBot(Bot* yapb_bot);
    void clearAllNeuralBots();
    
} // namespace PureNeural
