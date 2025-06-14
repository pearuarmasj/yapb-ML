//
// YaPB Neural Network Integration - Minimal Working Version
// This header contains the basic structures needed for neural network integration
//

#pragma once

// Forward declarations to avoid circular dependencies
class Bot;
class ConVar;

// Simple training data structures (will be expanded later)
struct ZombieTrainingState {
   // Position and movement
   float posX, posY, posZ;
   float velX, velY, velZ;
   float viewYaw, viewPitch;
   
   // Health and status
   float health;
   float armor;
   bool isStuck;
   bool isOnLadder;
   bool isInWater;
   
   // Enemy information
   bool hasEnemy;
   float enemyPosX, enemyPosY, enemyPosZ;
   float enemyDistance;
   float enemyHealth;
   bool enemyVisible;
   float timeSinceEnemySeen;
   
   // Team and environment info
   int teamId;
   int numTeammatesNearby;
   int numEnemiesNearby;
   int currentTask;
   
   // Zombie-specific
   float huntRange;
   float aggressionLevel;
   bool targetEntityPresent;
   
   // Timestamp
   float gameTime;
};

struct ZombieTrainingAction {
   // Movement actions
   float moveForward;    // -1.0 to 1.0
   float moveRight;      // -1.0 to 1.0
   float turnYaw;        // -180.0 to 180.0
   float turnPitch;      // -90.0 to 90.0
   
   // Combat actions
   bool attackPrimary;
   bool attackSecondary;
   
   // Task switching
   int taskSwitch;       // 0=no change, 1=hunt, 2=attack, 3=seek cover
   
   // Movement modifiers
   bool jump;
   bool duck;
   bool walk;
   
   // Decision metadata
   float confidence;     // 0.0 to 1.0
   // Note: String removed to avoid complex dependencies
};

struct ZombieTrainingReward {
   float immediateReward;
   float enemyKillReward;
   float damageDealtReward;
   float proximityReward;
   float survivalReward;
   float teamCoordReward;
   float totalReward;
   
   // Event flags
   bool enemyKilled;
   float damageDealt;
   bool taskCompleted;
   float survivalTime;
};

// Global neural network configuration
extern ConVar cv_neural_training_enabled;
extern ConVar cv_neural_data_collection;
extern ConVar cv_neural_use_for_decisions;
extern ConVar cv_neural_debug_output;
extern ConVar cv_neural_verbose_output;
extern ConVar cv_neural_force_usage;