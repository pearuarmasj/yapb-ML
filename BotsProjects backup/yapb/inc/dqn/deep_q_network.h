//
// DEEP Q-NETWORK (DQN) IMPLEMENTATION - REAL REINFORCEMENT LEARNING!
// Proper Q-learning with experience replay and target networks
//

#pragma once

#include <yapb.h>
#include <vector>
#include <deque>
#include <random>
#include <array>

namespace DQN {

    // Experience tuple for replay buffer
    struct Experience {
        std::vector<float> state;           // Current state
        int action;                         // Action taken
        float reward;                       // Reward received
        std::vector<float> next_state;      // Next state
        bool done;                          // Episode terminated
        
        Experience(const std::vector<float>& s, int a, float r, const std::vector<float>& ns, bool d)
            : state(s), action(a), reward(r), next_state(ns), done(d) {}
    };

    // Experience replay buffer - STORES ACTUAL MEMORIES!
    class ExperienceReplayBuffer {
    private:
        std::deque<Experience> buffer;
        size_t max_size;
        std::mt19937 rng;
        
    public:
        ExperienceReplayBuffer(size_t capacity = 100000);
        
        void addExperience(const std::vector<float>& state, int action, float reward, 
                          const std::vector<float>& next_state, bool done);
        
        std::vector<Experience> sampleBatch(size_t batch_size);
        
        size_t size() const { return buffer.size(); }
        bool canSample(size_t batch_size) const { return buffer.size() >= batch_size; }
    };

    // Deep Q-Network - ACTUAL NEURAL NETWORK FOR Q-LEARNING!
    class DeepQNetwork {
    private:
        static const int STATE_SIZE = 25;       // Game state features
        static const int HIDDEN_SIZE = 256;     // Hidden layer neurons
        static const int ACTION_SIZE = 13;      // Discrete action space
        
        // Network weights
        std::vector<std::vector<float>> weights_input_hidden;
        std::vector<std::vector<float>> weights_hidden_hidden;
        std::vector<std::vector<float>> weights_hidden_output;
        std::vector<float> bias_hidden1;
        std::vector<float> bias_hidden2;
        std::vector<float> bias_output;
        
        // Target network (for stable learning)
        std::vector<std::vector<float>> target_weights_input_hidden;
        std::vector<std::vector<float>> target_weights_hidden_hidden;
        std::vector<std::vector<float>> target_weights_hidden_output;
        std::vector<float> target_bias_hidden1;
        std::vector<float> target_bias_hidden2;
        std::vector<float> target_bias_output;          // Learning parameters - VERY SLOW DECAY FOR CONTINUED LEARNING
        float learning_rate = 0.001f;
        float discount_factor = 0.99f;      // Gamma for future rewards
        float epsilon = 1.0f;               // Start with 100% exploration
        float epsilon_decay = 0.99995f;     // EXTREMELY SLOW decay - takes ~14k steps to reach 50%
        float epsilon_min = 0.05f;          // Lower minimum - always 5% exploration
        
        int target_update_frequency = 1000; // Update target network every N steps
        int step_count = 0;
        
        std::mt19937 rng;
        
    public:
        DeepQNetwork();
        
        // Forward pass through main network
        std::vector<float> forward(const std::vector<float>& state);
        
        // Forward pass through target network
        std::vector<float> forwardTarget(const std::vector<float>& state);
        
        // Select action using epsilon-greedy policy
        int selectAction(const std::vector<float>& state);
        
        // Train the network using a batch of experiences
        void trainBatch(const std::vector<Experience>& batch);
        
        // Update target network weights
        void updateTargetNetwork();
          // Get current exploration rate
        float getEpsilon() const { return epsilon; }
        
        // Decay exploration rate
        void decayEpsilon();
        
        // Reset exploration rate (for new episodes)
        void resetEpsilon() { epsilon = 1.0f; }
        
    private:
        void randomizeWeights();
        float relu(float x) { return std::max(0.0f, x); }
        std::vector<float> softmax(const std::vector<float>& values);
    };

    // Game state representation for DQN
    struct GameStateVector {
        // Position & orientation (6 features)
        float pos_x, pos_y, pos_z;
        float angle_yaw, angle_pitch, angle_roll;
        
        // Movement & physics (6 features)
        float velocity_x, velocity_y, velocity_z;
        float speed, on_ground, in_water;
        
        // Health & status (3 features)
        float health, armor, alive_time;
        
        // Environmental sensors (10 features)
        float wall_distances[8];    // 8 directional rays
        float ground_distance;
        float ceiling_distance;
        
        std::vector<float> toVector() const;
    };

    // Discrete action space for Q-learning
    enum class BotAction {
        MOVE_FORWARD = 0,
        MOVE_BACK = 1,
        MOVE_LEFT = 2,
        MOVE_RIGHT = 3,
        TURN_LEFT = 4,
        TURN_RIGHT = 5,
        TURN_UP = 6,
        TURN_DOWN = 7,
        JUMP = 8,
        CROUCH = 9,
        STOP = 10,
        RANDOM_TURN = 11,
        COMBO_FORWARD_JUMP = 12,
        ACTION_COUNT = 13
    };    // DQN Bot - REPLACES THE BROKEN NEURAL BOT!
    class DQNBot {
    public:  // Make members public for system manager access
        Bot* m_yapb_bot;
        
    private:  // Move other members to private
        DeepQNetwork m_dqn;
        ExperienceReplayBuffer m_replay_buffer;
        
        // Experience tracking
        std::vector<float> m_last_state;
        int m_last_action;
        float m_last_reward;
        bool m_has_last_state;
        
        // Learning parameters
        static const size_t BATCH_SIZE = 32;
        static const size_t MIN_REPLAY_SIZE = 1000;
        
        // Reward tracking
        float m_total_reward = 0.0f;
        float m_episode_reward = 0.0f;
        int m_episode_count = 0;
        int m_step_count = 0;
        
        // Performance tracking
        float m_spawn_time = 0.0f;
        Vector m_last_position;
        float m_last_health = 100.0f;
        float m_exploration_distance = 0.0f;
        int m_death_count = 0;
        
    public:
        DQNBot(Bot* yapb_bot);
        
        // Main DQN learning loop
        void runDQNFrame();
        
        // Convert game state to feature vector
        GameStateVector extractGameState();
        
        // Apply selected action to bot
        void applyAction(BotAction action);
        
        // Calculate reward signal
        float calculateReward(const GameStateVector& current_state);
        
        // Handle bot spawn/death
        void onBotSpawn();
        void onBotDeath();
        
        // Performance reporting
        void printLearningStats();
        
        // Get learning statistics
        float getTotalReward() const { return m_total_reward; }
        float getEpsilon() const { return m_dqn.getEpsilon(); }
        int getEpisodeCount() const { return m_episode_count; }
    };

    // Global DQN system management
    void initializeDQNSystem();
    void shutdownDQNSystem();
    void addDQNBot(Bot* yapb_bot);
    void removeDQNBot(Bot* yapb_bot);
    void runDQNSystemFrame();
    void printDQNSystemStats();

    // Global bot registry
    extern std::vector<DQNBot*> g_dqn_bots;
}
