//
// PURE NEURAL BOT IMPLEMENTATION - IDIOTS LEARNING THROUGH PAIN!
// No YaPB intelligence - pure neural network from scratch
//

#include <yapb.h>
#include <pure_neural/pure_neural_bot.h>
#include <random>
#include <cmath>
#include <algorithm>

namespace PureNeural {    // Global neural bot registry
    std::vector<PureNeuralBot*> g_neural_bots;
    
    // Random number generator for neural network
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    static std::uniform_real_distribution<float> action_dist(0.0f, 1.0f);

    // ================================
    // SIMPLE NEURAL NETWORK - STARTS STUPID!
    // ================================
      SimpleNeuralNetwork::SimpleNeuralNetwork() {
        // Initialize weight matrices with random values (complete idiot brain!)
        weights_input_hidden.resize(INPUT_SIZE, std::vector<float>(HIDDEN_SIZE));
        weights_hidden_output.resize(HIDDEN_SIZE, std::vector<float>(OUTPUT_SIZE));
        bias_hidden.resize(HIDDEN_SIZE);
        bias_output.resize(OUTPUT_SIZE);
        
        // Initialize momentum vectors (for memory)
        velocity_input_hidden.resize(INPUT_SIZE, std::vector<float>(HIDDEN_SIZE, 0.0f));
        velocity_hidden_output.resize(HIDDEN_SIZE, std::vector<float>(OUTPUT_SIZE, 0.0f));
        
        randomizeWeights();
        
        game.print("🧠 AGGRESSIVE Neural Network initialized - Learning Rate: %.3f, Momentum: %.2f!", 
                  learning_rate, momentum);
    }
    
    void SimpleNeuralNetwork::randomizeWeights() {
        // Make the bot completely stupid with random weights
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weights_input_hidden[i][j] = weight_dist(gen);
            }
        }
        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                weights_hidden_output[i][j] = weight_dist(gen);
            }
            bias_hidden[i] = weight_dist(gen);
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            bias_output[i] = weight_dist(gen);
        }
    }
    
    ActionOutput SimpleNeuralNetwork::processGameState(const GameState& state) {
        // Convert game state to neural network input
        std::vector<float> input = {
            state.pos_x, state.pos_y, state.pos_z,
            state.angle_yaw, state.angle_pitch,
            state.health, state.alive_time,
            state.wall_forward, state.wall_left, state.wall_right, state.ground_below,
            state.vision_rays[0], state.vision_rays[1], state.vision_rays[2], state.vision_rays[3],
            state.vision_rays[4], state.vision_rays[5], state.vision_rays[6], state.vision_rays[7],
            state.velocity_x, state.velocity_y
        };
        
        // Forward pass - Hidden layer
        std::vector<float> hidden(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = bias_hidden[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                sum += input[j] * weights_input_hidden[j][i];
            }
            hidden[i] = std::tanh(sum);  // Activation function
        }
        
        // Forward pass - Output layer  
        std::vector<float> output(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float sum = bias_output[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += hidden[j] * weights_hidden_output[j][i];
            }
            output[i] = std::tanh(sum);
        }
        
        // Store for learning
        last_input = input;
        last_hidden = hidden;
        last_output = output;
          // Convert to action output WITH EXPLORATION!
        ActionOutput actions;
        
        // Add random exploration to break stuck behavior and local minima
        bool use_random = (action_dist(gen) < exploration_rate);
        
        if (use_random) {
            // Random exploration action - completely random to escape stuck states
            actions.move_forward = weight_dist(gen);   
            actions.move_right = weight_dist(gen);     
            actions.turn_yaw = weight_dist(gen);       
            actions.turn_pitch = weight_dist(gen);     
            actions.jump = action_dist(gen);    
            actions.crouch = action_dist(gen);  
        } else {
            // Neural network action
            actions.move_forward = output[0];   // -1 to +1
            actions.move_right = output[1];     // -1 to +1  
            actions.turn_yaw = output[2];       // -1 to +1
            actions.turn_pitch = output[3];     // -1 to +1
            actions.jump = (output[4] + 1.0f) * 0.5f;    // 0 to 1
            actions.crouch = (output[5] + 1.0f) * 0.5f;  // 0 to 1
        }
        
        return actions;
    }
      void SimpleNeuralNetwork::learnFromReward(const LearningReward& reward) {
        // MUCH MORE AGGRESSIVE REINFORCEMENT LEARNING!
        // Scale reward and use momentum for consistent learning
        
        float scaled_reward = reward.total_reward * learning_rate;
        
        // Debug learning occasionally
        static float last_learn_debug = 0.0f;
        if (game.time() - last_learn_debug > 5.0f && std::abs(scaled_reward) > 0.1f) {
            game.print("🧠 LEARNING: Reward=%.1f, Scaled=%.3f, LR=%.3f", 
                      reward.total_reward, scaled_reward, learning_rate);
            last_learn_debug = game.time();
        }
        
        // Proper gradient descent with momentum for hidden->output weights
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                // Calculate gradient (simplified RL gradient)
                float gradient = scaled_reward * last_hidden[i] * (1.0f - last_output[j] * last_output[j]); // tanh derivative
                
                // Apply momentum (builds consistent behavior patterns)
                velocity_hidden_output[i][j] = momentum * velocity_hidden_output[i][j] + gradient;
                
                // Update weight with momentum
                weights_hidden_output[i][j] += velocity_hidden_output[i][j];
            }
        }
        
        // Proper gradient descent with momentum for input->hidden weights  
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                // Calculate gradient
                float gradient = scaled_reward * last_input[i] * (1.0f - last_hidden[j] * last_hidden[j]) * 0.1f; // tanh derivative
                
                // Apply momentum
                velocity_input_hidden[i][j] = momentum * velocity_input_hidden[i][j] + gradient;
                
                // Update weight with momentum
                weights_input_hidden[i][j] += velocity_input_hidden[i][j];
            }
        }
        
        // Clamp weights to prevent explosion (wider range for more expressive learning)
        for (auto& layer : weights_input_hidden) {
            for (auto& weight : layer) {
                weight = std::clamp(weight, -50.0f, 50.0f);  // Much wider range
            }
        }
        
        for (auto& layer : weights_hidden_output) {
            for (auto& weight : layer) {
                weight = std::clamp(weight, -50.0f, 50.0f);  // Much wider range
            }
        }
        
        // Adaptive exploration - reduce randomness as bot gets better
        if (reward.total_reward > 10.0f) {
            exploration_rate = std::max(0.1f, exploration_rate * 0.999f);  // Slowly reduce exploration when doing well
        } else if (reward.total_reward < -100.0f) {
            exploration_rate = std::min(0.5f, exploration_rate * 1.001f);  // Increase exploration when doing badly
        }
    }
    
    // ================================    // PURE NEURAL BOT - NO YAPB INTELLIGENCE!
    // ================================
    
    PureNeuralBot::PureNeuralBot(Bot* yapb_bot) : m_yapb_bot(yapb_bot) {
        m_spawn_time = game.time();
        m_last_movement_time = game.time();
        m_last_position = yapb_bot->pev->origin;
        m_last_angles = yapb_bot->pev->angles;
        m_last_health = yapb_bot->pev->health;
        
        game.print("🤖 Pure Neural Bot '%s' initialized - Starting as complete idiot!", 
                  yapb_bot->pev->netname.chars());
    }
    
    void PureNeuralBot::runNeuralFrame() {
        // MAIN NEURAL LOOP - REPLACES ALL YAPB LOGIC!
        
        // Extract current game state (NO YaPB help!)
        m_current_state = extractGameState();
        
        // Run neural network to get actions (starts random/stupid)
        m_current_actions = m_brain.processGameState(m_current_state);
        
        // Apply actions to game world
        applyNeuralActions(m_current_actions);
        
        // Calculate learning reward (pain/pleasure feedback)
        m_current_reward = calculateLearningReward();
          // Learn from experience (gradually get less stupid)
        m_brain.learnFromReward(m_current_reward);
        
        // Debug rewards occasionally
        static float last_reward_debug = 0.0f;
        if (game.time() - last_reward_debug > 4.0f && m_current_reward.total_reward != 0.0f) {
            game.print("💰 Reward '%s': Total=%.1f (Death=%.1f, Stuck=%.1f, Move=%.1f)", 
                      m_yapb_bot->pev->netname.chars(),
                      m_current_reward.total_reward,
                      m_current_reward.death_penalty,
                      m_current_reward.stuck_penalty, 
                      m_current_reward.movement_reward);
            last_reward_debug = game.time();
        }
        
        // Debug output (occasionally)
        static float last_debug = 0.0f;
        if (game.time() - last_debug > 5.0f) {
            printLearningStats();
            last_debug = game.time();
        }
    }
    
    GameState PureNeuralBot::extractGameState() {
        GameState state;
        Vector pos = m_yapb_bot->pev->origin;
        Vector angles = m_yapb_bot->pev->angles;
        
        // Normalize position (rough world bounds)
        state.pos_x = pos.x / 4096.0f;
        state.pos_y = pos.y / 4096.0f; 
        state.pos_z = pos.z / 1024.0f;
        
        // Normalize angles
        state.angle_yaw = angles.x / 360.0f;
        state.angle_pitch = angles.y / 90.0f;
        
        // Health and survival
        state.health = m_yapb_bot->pev->health / 100.0f;
        state.alive_time = game.time() - m_spawn_time;
          // Environmental sensing (NO YaPB pathfinding!)
        // Simple ray traces for wall detection
        TraceResult tr;        Vector forward = Vector(1, 0, 0);  // Simple forward direction
        Vector right = Vector(0, 1, 0);    // Simple right direction  
        Vector up = Vector(0, 0, 1);       // Simple up direction        // Wall distances using YaPB's game.testLine
        game.testLine(pos, pos + forward * 200.0f, TraceIgnore::Everything, m_yapb_bot->ent(), &tr);
        state.wall_forward = tr.flFraction;
        
        game.testLine(pos, pos + right * 200.0f, TraceIgnore::Everything, m_yapb_bot->ent(), &tr);
        state.wall_right = tr.flFraction;
        
        game.testLine(pos, pos - right * 200.0f, TraceIgnore::Everything, m_yapb_bot->ent(), &tr);
        state.wall_left = tr.flFraction;
        
        game.testLine(pos, pos - up * 200.0f, TraceIgnore::Everything, m_yapb_bot->ent(), &tr);
        state.ground_below = tr.flFraction;
        
        // Vision rays in 8 directions (basic obstacle detection)
        for (int i = 0; i < 8; i++) {
            float angle = (i * 45.0f) * (M_PI / 180.0f);
            Vector ray_dir = Vector(std::cos(angle), std::sin(angle), 0.0f);
            
            game.testLine(pos, pos + ray_dir * 300.0f, TraceIgnore::Everything, m_yapb_bot->ent(), &tr);
            state.vision_rays[i] = tr.flFraction;
        }
        
        // Movement state
        Vector velocity = m_yapb_bot->pev->velocity;
        state.velocity_x = velocity.x / 320.0f;  // Normalize to max run speed
        state.velocity_y = velocity.y / 320.0f;
        state.velocity_z = velocity.z / 320.0f;
        
        state.on_ground = (m_yapb_bot->pev->flags & FL_ONGROUND) != 0;
        state.in_water = (m_yapb_bot->pev->waterlevel > 0);
        
        return state;
    }
      void PureNeuralBot::applyNeuralActions(const ActionOutput& actions) {
        // Convert neural output to actual game inputs
        // COMPLETELY BYPASS YaPB INTELLIGENCE!
        
        // Clear all YaPB inputs first
        m_yapb_bot->pev->button = 0;
        
        // Debug output occasionally
        static float last_action_debug = 0.0f;
        if (game.time() - last_action_debug > 2.0f) {
            game.print("🤖 Neural Actions '%s': Move=%.2f, Strafe=%.2f, Turn=%.2f", 
                      m_yapb_bot->pev->netname.chars(),
                      actions.move_forward, actions.move_right, actions.turn_yaw);
            last_action_debug = game.time();
        }
        
        // Movement (forward/back) - LOWER THRESHOLD!
        if (actions.move_forward > 0.05f) {
            m_yapb_bot->pev->button |= IN_FORWARD;
        } else if (actions.move_forward < -0.05f) {
            m_yapb_bot->pev->button |= IN_BACK;
        }
        
        // Strafing (left/right) - LOWER THRESHOLD!
        if (actions.move_right > 0.05f) {
            m_yapb_bot->pev->button |= IN_MOVERIGHT;
        } else if (actions.move_right < -0.05f) {
            m_yapb_bot->pev->button |= IN_MOVELEFT;
        }
        
        // Jumping (probabilistic)
        if (action_dist(gen) < actions.jump) {
            m_yapb_bot->pev->button |= IN_JUMP;
        }
        
        // Crouching (probabilistic)
        if (action_dist(gen) < actions.crouch) {
            m_yapb_bot->pev->button |= IN_DUCK;
        }
          // Turning (modify view angles directly) - FIXED AXES!
        Vector current_angles = m_yapb_bot->pev->angles;
        current_angles.y += actions.turn_yaw * 10.0f;   // YAW = horizontal turning (left/right)
        current_angles.x += actions.turn_pitch * 5.0f;  // PITCH = vertical looking (up/down)
        
        // Clamp angles
        current_angles.y = std::fmod(current_angles.y, 360.0f);  // YAW wraps around
        current_angles.x = std::clamp(current_angles.x, -89.0f, 89.0f);  // PITCH clamped
        
        m_yapb_bot->pev->angles = current_angles;
        m_yapb_bot->pev->v_angle = current_angles;
    }
    
    LearningReward PureNeuralBot::calculateLearningReward() {
        LearningReward reward;
        Vector current_pos = m_yapb_bot->pev->origin;
        float current_health = m_yapb_bot->pev->health;
        
        // DEATH PENALTY (MASSIVE PAIN!)
        if (current_health <= 0.0f && m_last_health > 0.0f) {
            reward.death_penalty = -1000.0f;
            reward.just_died = true;
            m_death_count++;
            game.print("💀 Bot '%s' DIED! Death #%d - MASSIVE PENALTY!", 
                      m_yapb_bot->pev->netname.chars(), m_death_count);
        }        // Damage penalty
        if (current_health < m_last_health) {
            reward.fall_damage_penalty = (m_last_health - current_health) * -2.0f;
        }
        
        // IMPROVED MOVEMENT REWARDS - HORIZONTAL VELOCITY ONLY!
        Vector velocity = m_yapb_bot->pev->velocity;
        float horizontal_speed = std::sqrt(velocity.x * velocity.x + velocity.z * velocity.z);  // X/Z only - NO JUMPING!
        
        // Debug movement occasionally - HORIZONTAL SPEED ONLY!
        static float last_movement_debug = 0.0f;
        if (game.time() - last_movement_debug > 3.0f) {
            game.print("🏃 Bot '%s': HorizontalSpeed=%.2f, Exploration=%.1fm", 
                      m_yapb_bot->pev->netname.chars(), 
                      horizontal_speed, m_total_exploration_distance);
            last_movement_debug = game.time();
        }
        
        // Reward actual HORIZONTAL velocity (not jumping in place)
        if (horizontal_speed > 10.0f) {
            reward.movement_reward = horizontal_speed * 0.05f;  // Reward based on horizontal speed only
        }
        
        // EXPLORATION DISTANCE - HORIZONTAL MOVEMENT ONLY! (NO JUMPING BULLSHIT!)
        if (horizontal_speed > 20.0f) {
            m_total_exploration_distance += horizontal_speed * 0.1f;  // Distance = horizontal_speed * time
        }// BRUTAL STUCK PUNISHMENT - TRACK X/Z COORDINATES ONLY!
        float horizontal_distance = std::sqrt(
            std::pow(current_pos.x - m_last_position.x, 2) + 
            std::pow(current_pos.z - m_last_position.z, 2)  // X/Z only - ignore Y (vertical look)
        );
        
        // PROPER TIME-BASED STUCK TRACKING (PER BOT!) - MUCH MORE SENSITIVE!
        if (horizontal_distance > 1.0f) {  // LOWER THRESHOLD - any tiny movement counts
            // They moved! Reset the stuck timer
            m_last_movement_time = game.time();
            m_stuck_timer = 0.0f;
        } else {
            // They're stuck! Calculate how long
            m_stuck_timer = game.time() - m_last_movement_time;
        }
          // INSANELY BRUTAL ESCALATING PENALTIES - IMMEDIATE PUNISHMENT!
        reward.stuck_penalty = 0.0f;  // Reset first
        if (m_stuck_timer > 0.2f) {
            reward.stuck_penalty = -100.0f;   // 0.2 seconds = immediate severe pain
        }
        if (m_stuck_timer > 0.5f) {
            reward.stuck_penalty = -500.0f;  // 0.5 second = agony
        }
        if (m_stuck_timer > 1.0f) {
            reward.stuck_penalty = -2000.0f; // 1 second = massive torture
        }
        if (m_stuck_timer > 2.0f) {
            reward.stuck_penalty = -10000.0f; // 2+ seconds = ABSOLUTE HELL
        }
          // DEBUG STUCK TIMER CONSTANTLY AND FORCE RESET IF HOPELESS
        static float last_stuck_debug = 0.0f;
        if (game.time() - last_stuck_debug > 1.0f && m_stuck_timer > 0.2f) {
            game.print("⚡ NUCLEAR WHIPPING '%s': Stuck %.1fs, HorizontalDist=%.3f, HorizSpeed=%.1f, Penalty=%.0f", 
                      m_yapb_bot->pev->netname.chars(), m_stuck_timer, horizontal_distance, horizontal_speed, reward.stuck_penalty);
            last_stuck_debug = game.time();
        }
        
        // FORCE RESET BOT POSITION IF HOPELESS (stuck for 3+ seconds)
        if (m_stuck_timer > 3.0f) {
            // Teleport bot to a random nearby position to break free
            Vector random_pos = m_yapb_bot->pev->origin;
            random_pos.x += weight_dist(gen) * 200.0f;  // Random offset
            random_pos.z += weight_dist(gen) * 200.0f;
            
            // Try to teleport (might not work, but worth trying)
            m_yapb_bot->pev->origin = random_pos;
            m_yapb_bot->pev->angles.y += weight_dist(gen) * 180.0f;  // Random turn
            
            // Reset stuck timer and apply massive penalty
            m_stuck_timer = 0.0f;
            m_last_movement_time = game.time();
            reward.stuck_penalty = -20000.0f;  // EVEN MORE MASSIVE PENALTY
            
            game.print("🚁 EMERGENCY TELEPORT '%s' - was stuck %.1fs!", 
                      m_yapb_bot->pev->netname.chars(), m_stuck_timer);
        }// MOVEMENT BUTTON REWARDS - Encourage trying to move HORIZONTALLY
        if (m_yapb_bot->pev->button & (IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT)) {
            if (horizontal_speed > 5.0f) {
                reward.movement_attempt_reward = 2.0f;  // Reward successful HORIZONTAL movement attempts
            } else {
                reward.movement_attempt_penalty = -1.0f;  // Small penalty for trying to move but failing
            }
        }
        
        // Wall collision detection - use HORIZONTAL speed
        if (horizontal_speed < 5.0f && (m_yapb_bot->pev->button & IN_FORWARD)) {
            reward.wall_collision_penalty = -20.0f;
        }
        
        // Survival reward (small but consistent)
        reward.survival_reward = 0.1f;
        
        // New area discovery - VELOCITY BASED ONLY!
        bool found_new_area = true;
        for (const Vector& visited : m_visited_positions) {
            if (current_pos.distance(visited) < 200.0f) {  // Increased from 100 to 200
                found_new_area = false;
                break;
            }
        }
          // Only count as new area if actually moving with REAL HORIZONTAL VELOCITY
        if (found_new_area && horizontal_speed > 50.0f) {  // Must have significant HORIZONTAL velocity
            m_visited_positions.push_back(current_pos);
            reward.new_area_reward = 50.0f;
            reward.discovered_new_area = true;
            m_areas_discovered++;
            game.print("🗺️ Bot '%s' discovered new area! Total areas: %d (horizontal speed: %.1f)", 
                      m_yapb_bot->pev->netname.chars(), m_areas_discovered, horizontal_speed);
        }
        
        // Height exploration (encourage vertical movement)
        static float max_z = current_pos.z;
        if (current_pos.z > max_z + 50.0f) {
            reward.height_exploration_reward = 10.0f;
            max_z = current_pos.z;
        }
          // Calculate total reward
        reward.total_reward = reward.death_penalty + reward.fall_damage_penalty + 
                             reward.stuck_penalty + reward.wall_collision_penalty +
                             reward.spinning_penalty + reward.movement_attempt_reward +
                             reward.movement_attempt_penalty + reward.movement_reward + 
                             reward.new_area_reward + reward.survival_reward + 
                             reward.height_exploration_reward;
          // Update tracking
        m_last_position = current_pos;
        m_last_angles = m_yapb_bot->pev->angles;  // Track angle changes
        m_last_health = current_health;
        
        return reward;
    }
    
    void PureNeuralBot::onBotDeath() {
        m_total_alive_time += (game.time() - m_spawn_time);
        
        game.print("📊 Bot '%s' death stats: Lived %.1fs, Explored %.1fm, Areas: %d", 
                  m_yapb_bot->pev->netname.chars(),
                  game.time() - m_spawn_time,
                  m_total_exploration_distance,
                  m_areas_discovered);
    }
    
    void PureNeuralBot::onBotSpawn() {
        m_spawn_time = game.time();
        m_last_position = m_yapb_bot->pev->origin;
        m_last_health = 100.0f;
        m_stuck_timer = 0.0f;
        
        game.print("🎯 Bot '%s' spawned - Back to being an idiot!", 
                  m_yapb_bot->pev->netname.chars());
    }
    
    void PureNeuralBot::printLearningStats() {
        float avg_alive_time = (m_death_count > 0) ? (m_total_alive_time / m_death_count) : (game.time() - m_spawn_time);
        
        game.print("🧠 Neural Stats '%s': Deaths=%d, AvgLife=%.1fs, Exploration=%.1fm, Areas=%d", 
                  m_yapb_bot->pev->netname.chars(),
                  m_death_count,
                  avg_alive_time,
                  m_total_exploration_distance,
                  m_areas_discovered);
    }
    
    // ================================
    // GLOBAL SYSTEM MANAGEMENT
    // ================================
    
    void initializePureNeuralSystem() {
        game.print("🚀 PURE NEURAL SYSTEM INITIALIZED - NO YAPB INTELLIGENCE!");
        game.print("🤖 Bots will start as complete fucking idiots and learn through pain!");
    }
    
    void shutdownPureNeuralSystem() {
        for (auto* bot : g_neural_bots) {
            delete bot;
        }
        g_neural_bots.clear();
        
        game.print("💀 Pure Neural System shutdown");
    }
    
    void addPureNeuralBot(Bot* yapb_bot) {
        PureNeuralBot* neural_bot = new PureNeuralBot(yapb_bot);
        g_neural_bots.push_back(neural_bot);
        
        game.print("➕ Added Pure Neural Bot: %s (Total: %d)", 
                  yapb_bot->pev->netname.chars(), 
                  (int)g_neural_bots.size());
    }
      void removePureNeuralBot(Bot* yapb_bot) {
        for (auto it = g_neural_bots.begin(); it != g_neural_bots.end(); ++it) {
            if ((*it)->m_yapb_bot == yapb_bot) {
                delete *it;
                g_neural_bots.erase(it);
                game.print("➖ Removed Pure Neural Bot: %s", yapb_bot->pev->netname.chars());
                break;
            }
        }
    }
    
    void clearAllNeuralBots() {
        for (auto* bot : g_neural_bots) {
            delete bot;
        }
        g_neural_bots.clear();
        game.print("🧹 Cleared all neural bots");
    }

} // namespace PureNeural
