//
// YaPB Neural Network Integration
// Collects training data from existing zombie AI behavior for machine learning
//

#pragma once

#include <yapb.h>
#include <vector>
#include <fstream>
#include <memory>
#include <chrono>

// Forward declarations
class ZombieStateCollector;
class NeuralNetworkInterface;

// Training data structures
struct ZombieTrainingState {
   // Position and movement
   Vector position;
   Vector velocity;
   Vector viewAngles;
   
   // Health and status
   float health;
   float armor;
   bool isStuck;
   bool isOnLadder;
   bool isInWater;
   
   // Enemy information
   bool hasEnemy;
   Vector enemyPosition;
   float enemyDistance;
   float enemyHealth;
   bool enemyVisible;
   float timeSinceEnemySeen;
   
   // Team and environment
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
   // Movement commands
   float moveForward;   // -1.0 to 1.0
   float moveRight;     // -1.0 to 1.0
   
   // View changes
   float turnYaw;       // angle change in degrees
   float turnPitch;     // angle change in degrees
   
   // Combat actions
   bool attackPrimary;
   bool attackSecondary;
   
   // Task decisions
   int taskSwitch;      // 0=no change, 1=hunt, 2=attack, 3=seek_cover
   
   // Movement modifiers
   bool jump;
   bool duck;
   bool walk;
   
   // Decision reasoning (for analysis)
   String decisionReason;
   float confidence;    // 0.0 to 1.0
};

struct ZombieTrainingReward {
   float immediateReward;
   float enemyKillReward;
   float damageDealtReward;
   float proximityReward;
   float survivalReward;
   float teamCoordReward;
   float totalReward;
   
   // Game events that triggered rewards
   bool enemyKilled;
   float damageDealt;
   bool taskCompleted;
   float survivalTime;
};

// Training data collection interface
class ZombieDataCollector {
private:
   Array <ZombieTrainingState> m_stateHistory;
   Array <ZombieTrainingAction> m_actionHistory;
   Array <ZombieTrainingReward> m_rewardHistory;
   
   std::ofstream m_dataFile;
   std::ofstream m_logFile;
   
   float m_lastCollectionTime;
   int m_collectionInterval;  // frames between data collection
   int m_maxHistorySize;
   
   bool m_isCollecting;
   bool m_writeToFile;
   String m_outputDirectory;
   
public:
   ZombieDataCollector ();
   ~ZombieDataCollector ();
   
   void initialize (const String &outputDir);
   void startCollection ();
   void stopCollection ();
   
   void collectState (Bot *zombie);
   void collectAction (Bot *zombie, const ZombieTrainingAction &action);
   void collectReward (Bot *zombie, const ZombieTrainingReward &reward);
   
   void saveToFile ();
   void clearHistory ();
   
   // Statistics
   int getStateCount () const { return m_stateHistory.length (); }
   int getActionCount () const { return m_actionHistory.length (); }
   int getRewardCount () const { return m_rewardHistory.length (); }
   
   // Configuration
   void setCollectionInterval (int frames) { m_collectionInterval = frames; }
   void setMaxHistorySize (int size) { m_maxHistorySize = size; }
   void setWriteToFile (bool enable) { m_writeToFile = enable; }
};

// Neural network inference interface (for trained models)
class NeuralZombieAI {
private:
   bool m_isInitialized;
   bool m_useNeuralNetwork;
   String m_modelPath;
   float m_confidence;
   
   // Fallback to rule-based AI
   bool m_useFallback;
   float m_fallbackThreshold;
   
public:
   NeuralZombieAI ();
   ~NeuralZombieAI ();
   
   bool initialize (const String &modelPath);
   void shutdown ();
   
   ZombieTrainingAction predictAction (const ZombieTrainingState &state);
   float getConfidence () const { return m_confidence; }
   
   bool isUsingNeuralNetwork () const { return m_useNeuralNetwork; }
   void setUseFallback (bool enable, float threshold = 0.7f);
   
   // Model management
   bool loadModel (const String &modelPath);
   bool isModelLoaded () const { return m_isInitialized; }
};

// Enhanced Bot class methods for neural network integration
class Bot {
   // Add these members to existing Bot class
private:
   static ZombieDataCollector *s_dataCollector;
   static NeuralZombieAI *s_neuralAI;
   
   ZombieTrainingState m_lastState;
   ZombieTrainingAction m_lastAction;
   float m_lastRewardCalculation;
   
   // Neural network decision tracking
   bool m_useNeuralDecisions;
   float m_neuralConfidenceThreshold;
   int m_neuralDecisionCount;
   int m_ruleBasedDecisionCount;
   
public:
   // Neural network integration methods
   void collectTrainingData ();
   void updateNeuralDecision ();
   ZombieTrainingState getCurrentState ();
   ZombieTrainingAction getCurrentAction ();
   ZombieTrainingReward calculateReward ();
   
   // Enhanced zombie AI with neural network support
   void neuralZombieLogic ();
   bool shouldUseNeuralNetwork ();
   void executeNeuralAction (const ZombieTrainingAction &action);
   
   // Statistics and debugging
   void printNeuralStats ();
   float getNeuralConfidence () const;
   
   // Static methods for global neural network management
   static void initializeNeuralSystem (const String &dataDir, const String &modelPath);
   static void shutdownNeuralSystem ();
   static void startDataCollection ();
   static void stopDataCollection ();
   static bool isNeuralSystemEnabled ();
};

// Global neural network configuration
extern ConVar cv_neural_training_enabled;
extern ConVar cv_neural_data_collection;
extern ConVar cv_neural_model_path;
extern ConVar cv_neural_confidence_threshold;
extern ConVar cv_neural_collection_interval;
extern ConVar cv_neural_use_for_decisions;
extern ConVar cv_neural_fallback_enabled;
extern ConVar cv_neural_debug_output;

// Utility functions for neural network integration
namespace NeuralUtils {
   // State conversion functions
   ZombieTrainingState botToTrainingState (Bot *bot);
   void trainingStateToBot (Bot *bot, const ZombieTrainingState &state);
   
   // Action conversion functions
   ZombieTrainingAction extractActionFromBot (Bot *bot);
   void applyActionToBot (Bot *bot, const ZombieTrainingAction &action);
   
   // Reward calculation functions
   float calculateProximityReward (const Vector &zombiePos, const Vector &enemyPos);
   float calculateSurvivalReward (float healthBefore, float healthAfter, float timeDelta);
   float calculateTeamCoordinationReward (Bot *zombie, int teammatesNearby);
   float calculateTaskCompletionReward (Task::TaskType completedTask);
   
   // Data normalization
   float normalizePosition (float value, float max = 4096.0f);
   float normalizeHealth (float health);
   float normalizeDistance (float distance, float max = 4096.0f);
   float normalizeAngle (float angle);
   
   // File I/O utilities
   bool saveTrainingData (const String &filename, 
                         const Array <ZombieTrainingState> &states,
                         const Array <ZombieTrainingAction> &actions,
                         const Array <ZombieTrainingReward> &rewards);
   
   bool loadTrainingData (const String &filename,
                         Array <ZombieTrainingState> &states,
                         Array <ZombieTrainingAction> &actions,
                         Array <ZombieTrainingReward> &rewards);
   
   // Neural network communication
   bool callPythonTrainer (const String &scriptPath, const String &dataPath);
   bool loadNeuralModel (const String &modelPath);
   
   // Performance monitoring
   void logNeuralPerformance (const String &logFile, 
                             float neuralAccuracy, 
                             float ruleBasedAccuracy,
                             int totalDecisions);
};

// Training session management
class ZombieTrainingSession {
private:
   String m_sessionId;
   float m_startTime;
   float m_endTime;
   int m_totalStates;
   int m_totalActions;
   int m_totalRewards;
   
   String m_outputDirectory;
   bool m_isActive;
   
public:
   ZombieTrainingSession (const String &sessionId);
   ~ZombieTrainingSession ();
   
   void startSession ();
   void endSession ();
   
   void recordState (const ZombieTrainingState &state);
   void recordAction (const ZombieTrainingAction &action);
   void recordReward (const ZombieTrainingReward &reward);
   
   void generateReport ();
   void saveSession ();
   
   bool isActive () const { return m_isActive; }
   String getSessionId () const { return m_sessionId; }
   float getDuration () const { return m_endTime - m_startTime; }
   
   // Statistics
   int getTotalStates () const { return m_totalStates; }
   int getTotalActions () const { return m_totalActions; }
   int getTotalRewards () const { return m_totalRewards; }
};

// Neural network evaluation metrics
struct NeuralPerformanceMetrics {
   float accuracy;              // % of correct decisions
   float averageReward;         // average reward per episode
   float episodeLength;         // average episode length
   float convergenceRate;       // training convergence rate
   float explorationRate;       // exploration vs exploitation ratio
   
   int totalDecisions;
   int correctDecisions;
   int neuralDecisions;
   int ruleBasedDecisions;
   
   float trainingTime;
   float inferenceTime;
   
   void reset () {
      accuracy = 0.0f;
      averageReward = 0.0f;
      episodeLength = 0.0f;
      convergenceRate = 0.0f;
      explorationRate = 0.0f;
      totalDecisions = 0;
      correctDecisions = 0;
      neuralDecisions = 0;
      ruleBasedDecisions = 0;
      trainingTime = 0.0f;
      inferenceTime = 0.0f;
   }
   
   void calculate () {
      if (totalDecisions > 0) {
         accuracy = static_cast <float> (correctDecisions) / totalDecisions;
      }
   }
   
   void print () const {
      game.print ("Neural AI Performance Metrics:");
      game.print ("  Accuracy: %.2f%% (%d/%d)", accuracy * 100.0f, correctDecisions, totalDecisions);
      game.print ("  Average Reward: %.2f", averageReward);
      game.print ("  Episode Length: %.2f", episodeLength);
      game.print ("  Neural Decisions: %d (%.1f%%)", neuralDecisions, 
                 (neuralDecisions * 100.0f) / totalDecisions);
      game.print ("  Rule-based Decisions: %d (%.1f%%)", ruleBasedDecisions,
                 (ruleBasedDecisions * 100.0f) / totalDecisions);
      game.print ("  Training Time: %.2fs", trainingTime);
      game.print ("  Inference Time: %.4fs", inferenceTime);
   }
};

// Global performance tracking
extern NeuralPerformanceMetrics g_neuralMetrics;
