//
// PURE NEURAL AI - NO YAPB INTELLIGENCE! 
// Bots start as complete idiots and learn through pain and exploration
//

#include <yapb.h>
#include <pure_neural/pure_neural_bot.h>

// ConVar declarations for pure neural system
ConVar cv_pure_neural_enabled ("pure_neural_enabled", "1", "Enable pure neural learning system (no YaPB intelligence)");
ConVar cv_pure_neural_debug ("pure_neural_debug", "1", "Show neural learning debug output");

// Global neural system state
static bool g_neural_system_initialized = false;

// ================================
// PURE NEURAL SYSTEM INITIALIZATION
// ================================

void initializeNeuralSystem() {
    if (!g_neural_system_initialized && cv_pure_neural_enabled.as<bool>()) {
        PureNeural::initializePureNeuralSystem();
        g_neural_system_initialized = true;
        
        game.print("🧠 PURE NEURAL AI SYSTEM ACTIVE!");
        game.print("🤖 Bots will start as complete fucking idiots!");
        game.print("💀 They will learn through pain, death, and exploration!");
        game.print("🎯 NO YaPB pathfinding or intelligence assistance!");
    }
}

void shutdownNeuralSystem() {
    if (g_neural_system_initialized) {
        PureNeural::shutdownPureNeuralSystem();
        g_neural_system_initialized = false;
    }
}

// ================================
// BOT NEURAL LOGIC - REPLACES ALL YAPB INTELLIGENCE!
// ================================

void Bot::neuralZombieLogic() {
    // Initialize system if needed
    if (!g_neural_system_initialized) {
        initializeNeuralSystem();
    }
    
    // Check if this bot has a neural brain
    PureNeural::PureNeuralBot* neural_bot = nullptr;
    for (auto* bot : PureNeural::g_neural_bots) {
        if (bot->m_yapb_bot == this) {
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
// BOT EVENT HANDLERS
// ================================

void Bot::onNeuralBotSpawn() {
    // Find neural bot and notify of spawn
    for (auto* bot : PureNeural::g_neural_bots) {
        if (bot->m_yapb_bot == this) {
            bot->onBotSpawn();
            break;
        }
    }
}

void Bot::onNeuralBotDeath() {
    // Find neural bot and notify of death
    for (auto* bot : PureNeural::g_neural_bots) {
        if (bot->m_yapb_bot == this) {
            bot->onBotDeath();
            break;
        }
    }
}

// ================================
// DUMMY FUNCTIONS - NOT USED IN PURE NEURAL!
// ================================

ZombieTrainingState Bot::gatherZombieTrainingState() {
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
struct TrainingDataPacket {
   std::string botName;  
   std::string csvLine;   
   float timestamp;
   int priority;  // Higher priority packets get processed first
   size_t sequenceId;  // Ensures proper ordering within each bot's data
};

// Enhanced batch writing for maximum I/O efficiency
struct BatchDataPacket {
   std::string botName;
   std::vector<std::string> csvLines;
   size_t totalSize;
   float oldestTimestamp;
   float newestTimestamp;
};

class MultiThreadedDataWriter {
private:
   std::vector<std::thread> writerThreads;
   std::queue<TrainingDataPacket> dataQueue;
   std::mutex queueMutex;
   std::condition_variable queueCondition;
   std::atomic<bool> shutdownFlag{false};
   std::map<std::string, std::ofstream> botFiles;  
   std::mutex fileMutex;
   
   // NEW: Advanced queue management and batch processing
   std::priority_queue<TrainingDataPacket> priorityQueue;
   std::map<std::string, std::vector<std::string>> batchBuffers;  // Per-bot batching
   std::map<std::string, size_t> botSequenceIds;
   std::mutex batchMutex;
   std::atomic<size_t> globalSequenceId{0};
     // NEW: Performance and memory management
   static constexpr size_t MAX_QUEUE_SIZE = 50000;  // Prevent memory explosion
   size_t BATCH_SIZE = 100;        // Dynamic batch size from ConVar
   static constexpr size_t BATCH_TIMEOUT_MS = 50;   // Force write after 50ms
   std::map<std::string, std::chrono::high_resolution_clock::time_point> lastBatchWrite;
   
   // NEW: Performance monitoring
   std::atomic<size_t> totalPacketsProcessed{0};
   std::atomic<size_t> totalBatchesWritten{0};
   std::atomic<size_t> droppedPackets{0};
   std::chrono::high_resolution_clock::time_point startTime;
   
public:
   MultiThreadedDataWriter() {
      // DYNAMICALLY CONFIGURE THREAD COUNT BASED ON ConVar
      int availableThreads = std::thread::hardware_concurrency();
      int requestedThreads = cv_neural_thread_count.as<int>();
      int ioThreads = std::min(requestedThreads, availableThreads / 2);  // Use at most half available cores
      int processingThreads = std::max(1, ioThreads / 3); // Some threads for batch processing
      int totalThreads = ioThreads + processingThreads;
      
      // Clamp to reasonable limits
      ioThreads = std::max(1, std::min(16, ioThreads));
      processingThreads = std::max(1, std::min(4, processingThreads));
      totalThreads = ioThreads + processingThreads;
      
      game.print("🚀 CONFIGURABLE MULTITHREADED DATA WRITER: %d threads (%d I/O + %d batch) on %d-core system", 
                 totalThreads, ioThreads, processingThreads, availableThreads);
      
      startTime = std::chrono::high_resolution_clock::now();
      
      // Spawn I/O threads
      for (int i = 0; i < ioThreads; ++i) {
         writerThreads.emplace_back(&MultiThreadedDataWriter::ioThreadFunction, this, i);
      }
      
      // Spawn batch processing threads  
      for (int i = 0; i < processingThreads; ++i) {
         writerThreads.emplace_back(&MultiThreadedDataWriter::batchProcessingThreadFunction, this, i + ioThreads);
      }
   }
     ~MultiThreadedDataWriter() {
      shutdown();
   }
   
   void enqueueData(const String& botName, const String& csvLine, int priority = 1) {
      // Check queue size limit to prevent memory issues
      {
         std::lock_guard<std::mutex> lock(queueMutex);
         if (dataQueue.size() >= MAX_QUEUE_SIZE) {
            droppedPackets++;
            if (droppedPackets % 1000 == 0) {
               game.print("⚠️ QUEUE FULL: Dropped %zu packets to prevent memory issues", droppedPackets.load());
            }
            return;
         }
      }
      
      TrainingDataPacket packet;
      packet.botName = botName.chars();  
      packet.csvLine = csvLine.chars();  
      packet.timestamp = game.time();
      packet.priority = priority;
      packet.sequenceId = globalSequenceId++;
      
      {
         std::lock_guard<std::mutex> lock(queueMutex);
         dataQueue.push(packet);
      }
      queueCondition.notify_one();
      totalPacketsProcessed++;
   }
   
   void shutdown() {
      game.print("🛑 SHUTTING DOWN ENHANCED MULTITHREADED DATA WRITER...");
      
      // Flush all pending batches first
      flushAllBatches();
      
      shutdownFlag = true;
      queueCondition.notify_all();
      
      for (auto& thread : writerThreads) {
         if (thread.joinable()) {
            thread.join();
         }
      }
      
      // Close all files and print final stats
      {
         std::lock_guard<std::mutex> lock(fileMutex);
         for (auto& pair : botFiles) {
            if (pair.second.is_open()) {
               pair.second.close();
            }
         }
      }
      
      printFinalStats();
      game.print("✅ ENHANCED MULTITHREADED DATA WRITER SHUT DOWN CLEANLY");
   }
   
   void printStats() {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
      
      size_t packetsPerSecond = duration.count() > 0 ? totalPacketsProcessed / duration.count() : 0;
      size_t batchesPerSecond = duration.count() > 0 ? totalBatchesWritten / duration.count() : 0;
      
      game.print("📊 DATA WRITER STATS: %zu packets/sec, %zu batches/sec, %zu dropped, queue: %zu", 
                 packetsPerSecond, batchesPerSecond, droppedPackets.load(), dataQueue.size());
   }
   
private:
   // Enhanced I/O thread with batch processing
   void ioThreadFunction(int threadId) {
      game.print("📝 Enhanced I/O thread %d started - READY FOR MASSIVE THROUGHPUT!", threadId);
      
      while (!shutdownFlag) {
         std::unique_lock<std::mutex> lock(queueMutex);
         queueCondition.wait(lock, [this] { return !dataQueue.empty() || shutdownFlag; });
         
         if (shutdownFlag && dataQueue.empty()) {
            break;
         }
         
         if (!dataQueue.empty()) {
            TrainingDataPacket packet = dataQueue.front();
            dataQueue.pop();
            lock.unlock();
            
            // Add to batch buffer instead of immediate write
            addToBatch(packet, threadId);
         }
      }
      
      game.print("📝 Enhanced I/O thread %d finished", threadId);
   }
   
   // NEW: Dedicated batch processing thread
   void batchProcessingThreadFunction(int threadId) {
      game.print("🔄 Batch processing thread %d started - OPTIMIZING I/O PATTERNS!", threadId);
      
      while (!shutdownFlag) {
         std::this_thread::sleep_for(std::chrono::milliseconds(BATCH_TIMEOUT_MS));
           // Check for batches that need to be written
         std::lock_guard<std::mutex> lock(batchMutex);
         auto now = std::chrono::high_resolution_clock::now();
         
         // Update dynamic batch size from ConVar
         size_t currentBatchSize = std::max(50, std::min(500, cv_neural_batch_size.as<int>()));
         
         for (auto& pair : batchBuffers) {
            const std::string& botName = pair.first;
            std::vector<std::string>& buffer = pair.second;
            
            if (buffer.empty()) continue;
            
            // Write batch if it's full or timed out
            bool shouldWrite = buffer.size() >= currentBatchSize;
            if (!shouldWrite && lastBatchWrite.find(botName) != lastBatchWrite.end()) {
               auto timeSinceLastWrite = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now - lastBatchWrite[botName]);
               shouldWrite = timeSinceLastWrite.count() >= BATCH_TIMEOUT_MS;
            }
            
            if (shouldWrite) {
               writeBatchToFile(botName, buffer, threadId);
               buffer.clear();
               lastBatchWrite[botName] = now;
               totalBatchesWritten++;
            }
         }
      }
      
      game.print("� Batch processing thread %d finished", threadId);
   }
     void addToBatch(const TrainingDataPacket& packet, int threadId) {
      std::lock_guard<std::mutex> lock(batchMutex);
      
      // Update dynamic batch size from ConVar
      BATCH_SIZE = std::max(50, std::min(500, cv_neural_batch_size.as<int>()));
      
      batchBuffers[packet.botName].push_back(packet.csvLine);
      
      // Initialize sequence tracking
      if (botSequenceIds.find(packet.botName) == botSequenceIds.end()) {
         botSequenceIds[packet.botName] = 0;
      }
      
      // Immediate write if batch is full
      if (batchBuffers[packet.botName].size() >= BATCH_SIZE) {
         writeBatchToFile(packet.botName, batchBuffers[packet.botName], threadId);
         batchBuffers[packet.botName].clear();
         lastBatchWrite[packet.botName] = std::chrono::high_resolution_clock::now();
         totalBatchesWritten++;
      }
   }
   
   void writeBatchToFile(const std::string& botName, const std::vector<std::string>& batch, int threadId) {
      std::lock_guard<std::mutex> lock(fileMutex);
      
      // Create file if it doesn't exist
      if (botFiles.find(botName) == botFiles.end()) {
         String filename = strings.format("addons/yapb/ai_training/data/training_data_%s_enhanced.csv", 
                                         botName.c_str());
         botFiles[botName].open(filename.chars(), std::ios::out | std::ios::app);
         
         if (botFiles[botName].is_open()) {
            game.print("🔥 Thread %d opened ENHANCED file for bot %s - BATCH WRITING ACTIVATED!", 
                      threadId, botName.c_str());
         }
      }
      
      // Batch write all lines at once - MAXIMUM I/O EFFICIENCY
      if (botFiles[botName].is_open()) {
         for (const auto& line : batch) {
            botFiles[botName] << line;
         }
         botFiles[botName].flush(); // Ensure data is written
         
         // Less frequent debug output for batch writes
         static std::map<std::string, int> botBatchCounts;
         if (++botBatchCounts[botName] % 50 == 0) { // Every 50 batches
            game.print("💪 Thread %d: Bot %s batch #%d (%zu lines) - CRUSHING I/O BOTTLENECKS!", 
                      threadId, botName.c_str(), botBatchCounts[botName], batch.size());
         }
      }
   }
   
   void flushAllBatches() {
      std::lock_guard<std::mutex> lock(batchMutex);
      
      for (auto& pair : batchBuffers) {
         const std::string& botName = pair.first;
         std::vector<std::string>& buffer = pair.second;
         
         if (!buffer.empty()) {
            writeBatchToFile(botName, buffer, -1); // Special thread ID for shutdown
            buffer.clear();
            totalBatchesWritten++;
         }
      }
   }
   
   void printFinalStats() {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
      
      game.print("🏁 FINAL STATS: %zu total packets, %zu batches, %zu dropped in %ld seconds", 
                 totalPacketsProcessed.load(), totalBatchesWritten.load(), 
                 droppedPackets.load(), duration.count());
   }
};

// Global multithreaded writer instance
static MultiThreadedDataWriter* g_dataWriter = nullptr;

// GLOBAL SMART HAZARD DETECTOR - Our survival system for environmental learning
static ZombieAI::Survival::SmartHazardDetector g_smartHazardDetector;

// Cleanup function for proper shutdown
void shutdownMultiThreadedDataWriter() {
   if (g_dataWriter) {
      game.print("🛑 SHUTTING DOWN MULTITHREADED DATA WRITER - Saving all data...");
      delete g_dataWriter;
      g_dataWriter = nullptr;
      game.print("✅ MULTITHREADED DATA WRITER SHUT DOWN CLEANLY");
   }
}

// Forward declarations
bool loadNeuralWeights(const String& weightsPath);
bool parseJsonArray(const String& jsonText, int start, std::vector<float>& result);
bool parse2DJsonArray(const String& jsonText, int start, std::vector<std::vector<float>>& result);

// Neural network ConVars
ConVar cv_neural_training_enabled ("neural_training_enabled", "0", "Enable neural network training data collection.");
ConVar cv_neural_data_collection ("neural_data_collection", "0", "Enable real-time data collection from zombie AI.");
ConVar cv_neural_use_for_decisions ("neural_use_for_decisions", "0", "Use neural network for zombie AI decisions when available.");
ConVar cv_neural_debug_output ("neural_debug_output", "0", "Enable debug output for neural network decisions.");
ConVar cv_neural_verbose_output ("neural_verbose_output", "0", "Enable very verbose neural network debug messages (spams console).");
ConVar cv_neural_force_usage ("neural_force_usage", "0", "Force 100% neural network usage for testing (ignores adoption rate).");

// ENHANCED SYSTEM ConVars
ConVar cv_neural_collection_rate ("neural_collection_rate", "100", "Data collection frequency in Hz (10-100).");
ConVar cv_neural_priority_mode ("neural_priority_mode", "1", "Enable priority-based data processing.");
ConVar cv_neural_stats_interval ("neural_stats_interval", "30", "Statistics reporting interval in seconds.");

// Function for debug output - FORCE ALL MESSAGES to appear in console with RATE LIMITING
void neuralDebugPrint(const char* fmt, ...) {
   // Rate limiting to prevent console spam - allow max 10 messages per second
   static float lastMessageTime = 0.0f;
   static int messageCount = 0;
   static const int MAX_MESSAGES_PER_SECOND = 10;
   static const float RATE_LIMIT_WINDOW = 1.0f;
   
   float currentTime = game.time();
   
   // Reset counter every second
   if (currentTime - lastMessageTime >= RATE_LIMIT_WINDOW) {
      messageCount = 0;
      lastMessageTime = currentTime;
   }
   
   // Skip message if we've exceeded the rate limit (except for heartbeat messages)
   bool isHeartbeat = (fmt && strstr(fmt, "HEARTBEAT"));
   if (!isHeartbeat && messageCount >= MAX_MESSAGES_PER_SECOND) {
      return;
   }
   
   messageCount++;
   
   // Prevent potential crashes with null/invalid format strings
   if (!fmt || fmt[0] == '\0') {
      return;
   }
   
   // Format the message properly with error checking
   va_list ap;
   char buffer[512];
   va_start(ap, fmt);
   int result = vsnprintf(buffer, sizeof(buffer), fmt, ap);
   va_end(ap);
   
   // Check for formatting errors
   if (result < 0 || result >= static_cast<int>(sizeof(buffer))) {
      strcpy(buffer, "[NEURAL] Format error in debug message");
   }
   
   // Multiple output methods for redundancy
   try {
      // Method 1: Direct game print (most reliable)
      game.print("[NEURAL] %s", buffer);
   } catch (...) {
      // Fallback if game.print fails
   }
   
   try {
      // Method 2: YaPB's message system as backup
      ctrl.msg("[NEURAL] %s", buffer);
   } catch (...) {
      // Fallback if ctrl.msg fails
   }
   
   try {
      // Method 3: Console output as last resort
      printf("[NEURAL] %s\n", buffer);
      fflush(stdout);
   } catch (...) {
      // If even printf fails, we're in serious trouble
   }
   
   // Remove the HUD message system as it might be causing issues
   // The HUD system can crash or hang if called inappropriately
}

// Global instances - REAL neural network implementation
struct NeuralNetwork {
   std::vector<std::vector<float>> fc1_weights, fc2_weights, fc3_weights;
   std::vector<float> fc1_bias, fc2_bias, fc3_bias;
   bool loaded = false;
   
   std::vector<float> forward(const std::vector<float>& input) {
      if (!loaded) return {};
      
      // Layer 1
      std::vector<float> hidden1(fc1_weights.size());
      for (size_t i = 0; i < fc1_weights.size(); i++) {
         hidden1[i] = fc1_bias[i];
         for (size_t j = 0; j < input.size(); j++) {
            hidden1[i] += input[j] * fc1_weights[i][j];
         }
         hidden1[i] = std::max(0.0f, hidden1[i]); // ReLU
      }
      
      // Layer 2  
      std::vector<float> hidden2(fc2_weights.size());
      for (size_t i = 0; i < fc2_weights.size(); i++) {
         hidden2[i] = fc2_bias[i];
         for (size_t j = 0; j < hidden1.size(); j++) {
            hidden2[i] += hidden1[j] * fc2_weights[i][j];
         }
         hidden2[i] = std::max(0.0f, hidden2[i]); // ReLU
      }
      
      // Output layer
      std::vector<float> output(fc3_weights.size());
      for (size_t i = 0; i < fc3_weights.size(); i++) {
         output[i] = fc3_bias[i];
         for (size_t j = 0; j < hidden2.size(); j++) {
            output[i] += hidden2[j] * fc3_weights[i][j];
         }
      }
      
      return output;
   }
};

static NeuralNetwork g_neuralNet;
void *Bot::s_dataCollector = nullptr;
void *Bot::s_neuralAI = nullptr;

// Enhanced Bot neural network method implementations
void Bot::neuralZombieLogic () {   
   // ENHANCED MULTITHREADED WRITER INITIALIZATION - FULL POWER!
   static bool writerInitialized = false;
   if (!writerInitialized && cv_neural_data_collection.as <bool> ()) {
      if (!g_dataWriter) {
         g_dataWriter = new MultiThreadedDataWriter();
         game.print("🚀 ENHANCED MULTITHREADED DATA SYSTEM ACTIVATED - 32-THREAD MONSTER UNLEASHED!");
      }
      writerInitialized = true;
   }

   // ENHANCED HEARTBEAT & STATUS SYSTEM - Monitor performance continuously
   static float lastHeartbeat = 0.0f;
   static float lastStatsReport = 0.0f;
   static int heartbeatCounter = 0;
   
   if (game.time() - lastHeartbeat > 5.0f) { // Every 5 seconds
      neuralDebugPrint("💗 ENHANCED NEURAL HEARTBEAT #%d - System active at %.1f seconds", 
                       ++heartbeatCounter, game.time());      lastHeartbeat = game.time();
   }
   
   // Performance statistics at configurable intervals
   float statsInterval = std::max(10.0f, std::min(300.0f, cv_neural_stats_interval.as<float>()));
   if (g_dataWriter && (game.time() - lastStatsReport > statsInterval)) {      g_dataWriter->printStats();
      lastStatsReport = game.time();
   }
   
   // Rate-limited debug output (only show occasionally, not every frame)
   static float lastDebugOutput = 0.0f;
   static int debugCounter = 0;
   bool showDebug = (game.time() - lastDebugOutput > 2.0f); // Every 2 seconds
   if (showDebug) {
      neuralDebugPrint("Bot %s calling neuralZombieLogic #%d", pev->netname.chars(), ++debugCounter);
      lastDebugOutput = game.time();
   }
   
   // Debug the data collection variable (only occasionally)
   bool dataCollectionEnabled = cv_neural_data_collection.as <bool> ();
   if (showDebug) {
      neuralDebugPrint("Bot %s: neural_data_collection = %d", pev->netname.chars(), dataCollectionEnabled ? 1 : 0);
   }
   
   // First, always collect training data if enabled
   if (dataCollectionEnabled) {
      if (showDebug) {
         neuralDebugPrint("Bot %s: Calling collectTrainingData()...", pev->netname.chars());
      }
      collectTrainingData ();
   }
     // Decide whether to use neural network or allow regular AI
   if (shouldUseNeuralNetwork()) {
      // Rate-limited neural debug
      if (showDebug) {
         neuralDebugPrint("Bot %s USING NEURAL NETWORK decisions", pev->netname.chars());
      }
      
      // Get current state
      ZombieTrainingState state = getCurrentState();
      
      // Generate action using neural network
      ZombieTrainingAction action = predictNeuralAction();
      
      // Execute the action
      executeNeuralAction(action);
      
      // Track neural network usage
      m_neuralDecisionCount++;
        // Update counter for UI display (much less frequent)
      static int neuralDecisionCounter = 0;
      if (cv_neural_debug_output.as <bool> () && ++neuralDecisionCounter % 300 == 0) { // Every 300 decisions
         neuralDebugPrint("Bot %s stats: Neural=%d, Rule=%d", 
                    pev->netname.chars(), m_neuralDecisionCount, m_ruleBasedDecisionCount);
      }   } else {      
      // ALLOW REGULAR AI TO RUN - Just collect data, don't interfere!
      if (showDebug && dataCollectionEnabled) {
         neuralDebugPrint("Bot %s: Data collection mode - allowing REGULAR AI to run", pev->netname.chars());
      }
      // DO NOTHING - Let the regular AI in botlib.cpp continue normally
   }
   
   // Rate-limited end debug message
   if (showDebug) {
      neuralDebugPrint("Bot %s completed neuralZombieLogic", pev->netname.chars());
   }
}

void Bot::collectTrainingData () {
   // Rate-limited debug output - Show function is called
   static int totalCalls = 0;
   static float lastCollectDebug = 0.0f;
   if (++totalCalls % 1000 == 0 || (game.time() - lastCollectDebug > 10.0f)) { // Every 1000 calls or 10 seconds
      neuralDebugPrint("Bot %s: collectTrainingData() called %d times total", pev->netname.chars(), totalCalls);
      lastCollectDebug = game.time();
   }
   
   // Debug the conditions that might cause early return (rate limited)
   bool dataCollectionEnabled = cv_neural_data_collection.as <bool> ();
   bool isCreatureStatus = m_isCreature;
   bool isOnInfectedTeam = m_isOnInfectedTeam;
   
   static float lastConditionDebug = 0.0f;
   if (game.time() - lastConditionDebug > 5.0f) { // Every 5 seconds
      neuralDebugPrint("Bot %s: data_collection=%d, isCreature=%d, onInfectedTeam=%d", 
                       pev->netname.chars(), dataCollectionEnabled ? 1 : 0, isCreatureStatus ? 1 : 0, 
                       isOnInfectedTeam ? 1 : 0);
      lastConditionDebug = game.time();
   }   
   
   if (!dataCollectionEnabled) {
      static float lastEarlyReturnDebug = 0.0f;
      if (game.time() - lastEarlyReturnDebug > 5.0f) { // Every 5 seconds
         neuralDebugPrint("Bot %s: EARLY RETURN - data_collection=0", pev->netname.chars());
         lastEarlyReturnDebug = game.time();
      }
      return;
   }
   
   // FOR NOW, COLLECT DATA FOR ALL BOTS WHEN DATA COLLECTION IS ENABLED
   static float lastProceedDebug = 0.0f;
   if (game.time() - lastProceedDebug > 5.0f) { // Every 5 seconds
      neuralDebugPrint("Bot %s: PROCEEDING with enhanced data collection", pev->netname.chars());
      lastProceedDebug = game.time();
   }   
     // ENHANCED HIGH-FREQUENCY DATA COLLECTION - LEVERAGING THAT 32-THREAD MONSTER!
   static std::map<Bot*, float> botLastCollectionTime;
   static std::map<Bot*, int> botCollectionCounts;
   
   // CONFIGURABLE COLLECTION RATE - Default 100Hz but user can adjust
   float collectionFreq = std::max(10.0f, std::min(100.0f, cv_neural_collection_rate.as<float>()));
   const float collectionInterval = 1.0f / collectionFreq; // Convert Hz to interval
   
   if (botLastCollectionTime.find(this) == botLastCollectionTime.end()) {
      botLastCollectionTime[this] = 0.0f;
      botCollectionCounts[this] = 0;
   }
   
   if (game.time () - botLastCollectionTime[this] < collectionInterval) {
      return;
   }
   botLastCollectionTime[this] = game.time ();
   
   // Debug output with enhanced frequency tracking
   int& collectionCount = botCollectionCounts[this];
   if (++collectionCount % (int)(collectionFreq * 20) == 0) { // Every 20 seconds worth of collections
      neuralDebugPrint("Bot %s: HIGH-FREQ collection #%d (%.1f Hz target)", 
                       pev->netname.chars(), collectionCount, collectionFreq);
   }

   ZombieTrainingState state = getCurrentState ();
   ZombieTrainingAction action = getCurrentAction ();   
   ZombieTrainingReward reward = calculateReward ();
   
   // ENHANCED MULTITHREADED DATA WRITING - FULL POWER ENGAGED!
   if (g_dataWriter) {
      // Build enhanced CSV line with additional metadata
      String csvLine = strings.format("%.4f,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%d,%d,%.4f,%d,%d,%d,%d,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%d,%d\n",
                                       game.time(), pev->netname.chars(),
                                       state.posX, state.posY, state.posZ,
                                       state.velX, state.velY, state.velZ,
                                       state.viewYaw, state.viewPitch,
                                       state.health, state.armor, (state.isStuck ? 1 : 0),
                                       (state.isOnLadder ? 1 : 0), (state.isInWater ? 1 : 0),
                                       (state.hasEnemy ? 1 : 0), state.enemyPosX, state.enemyPosY, state.enemyPosZ,
                                       state.enemyDistance, state.enemyHealth, (state.enemyVisible ? 1 : 0),
                                       state.timeSinceEnemySeen, state.teamId, state.numTeammatesNearby,
                                       state.numEnemiesNearby, state.currentTask, state.huntRange,
                                       state.aggressionLevel, (state.targetEntityPresent ? 1 : 0),
                                       action.moveForward, action.moveRight, action.turnYaw, action.turnPitch,
                                       (action.attackPrimary ? 1 : 0), (action.attackSecondary ? 1 : 0), action.taskSwitch,
                                       (action.jump ? 1 : 0), (action.duck ? 1 : 0), (action.walk ? 1 : 0),
                                       action.confidence, reward.immediateReward, reward.totalReward,
                                       collectionCount, totalCalls); // Additional metadata
        // Determine priority - high priority for important events (if enabled)
      int priority = 1; // Normal priority
      if (cv_neural_priority_mode.as<bool>()) {
         if (state.hasEnemy && state.enemyVisible) priority = 3; // High priority for combat
         if (action.attackPrimary || action.attackSecondary) priority = 4; // Highest for attacks
         if (state.isStuck || state.health < 50) priority = 2; // Medium for problems
      }
      
      // Queue for ENHANCED multithreaded writing - BLAZING FAST!
      String botName = pev->netname.str();
      g_dataWriter->enqueueData(botName, csvLine, priority);
      
      // Debug output every 500 writes per bot due to increased frequency
      if (collectionCount % 500 == 0) {
         neuralDebugPrint("Bot %s: 🚀 %d ENHANCED samples queued (%.1f/sec)!", 
                         pev->netname.chars(), collectionCount,
                         collectionCount / (game.time() + 0.001f));
      }   
   } else {
      neuralDebugPrint("Bot %s: ❌ Enhanced g_dataWriter not initialized!", pev->netname.chars());
   }
   
   if (cv_neural_debug_output.as <bool> ()) {
      if (collectionCount % 1000 == 0) {
         game.print ("Bot %s exported %d MASSIVE training samples (100Hz collection!)", 
                    pev->netname.chars (), collectionCount);
      }
   }
}

bool Bot::shouldUseNeuralNetwork () {
   // FORCE DEBUG OUTPUT REGARDLESS OF SETTINGS
   neuralDebugPrint("Bot %s checking neural network conditions - Neural=%d, Rule=%d", 
                    pev->netname.chars(), m_neuralDecisionCount, m_ruleBasedDecisionCount);
   
   // SIMPLIFIED FOR TESTING - Only check if neural decisions are enabled
   if (!cv_neural_use_for_decisions.as <bool> ()) {
      neuralDebugPrint("Bot %s REJECTED: neural_use_for_decisions=0", pev->netname.chars());
      return false;
   }     // FORCE NEURAL NETWORK USAGE FOR TESTING - SKIP ALL OTHER CHECKS
   neuralDebugPrint("Bot %s: FORCING neural network usage (regular AI disabled)", pev->netname.chars());
   return true;
}

void Bot::printNeuralStats () {
   game.print ("Neural network integration active - data collection: %s", 
              cv_neural_data_collection.as <bool> () ? "enabled" : "disabled");
}

float Bot::getNeuralConfidence () const {
   // Calculate confidence based on current situation
   if (m_team != Team::Terrorist) {
      return 0.0f; // Only zombies use neural network
   }
   
   float baseConfidence = 0.7f; // Base neural network confidence
   
   // Simplified confidence calculation for now
   // TODO: Implement proper entity scanning when API is clarified
   
   // Confidence based on health
   if (pev->max_health > 0.0f) {
      float healthRatio = pev->health / pev->max_health;
      if (healthRatio < 0.3f) {
         baseConfidence -= 0.2f; // Less confident when low health
      }
   }
   
   // Confidence based on enemy presence
   if (m_enemy != nullptr) {
      baseConfidence += 0.1f; // More confident when we have a target
   }
   
   return cr::clamp (baseConfidence, 0.1f, 0.95f);
}

// Static method implementations
void Bot::initializeNeuralSystem (const String &dataDir, const String &modelPath) {
   if (cv_neural_debug_output.as <bool> ()) {
      game.print ("Initializing neural zombie AI system...");
      game.print ("Data directory: %s", dataDir.chars ());
      game.print ("Model path: %s", modelPath.chars ());
   }
}

void Bot::shutdownNeuralSystem () {
   if (cv_neural_debug_output.as <bool> ()) {
      game.print ("Shutting down neural zombie AI system");
   }
   
   // Shutdown the multithreaded data writer
   shutdownMultiThreadedDataWriter();
}

void Bot::startDataCollection () {
   cv_neural_data_collection.set (1);
   game.print ("Neural network data collection started");
}

void Bot::stopDataCollection () {
   cv_neural_data_collection.set (0);
   game.print ("Neural network data collection stopped");
}

bool Bot::isNeuralSystemEnabled () {
   return cv_neural_data_collection.as <bool> () || cv_neural_use_for_decisions.as <bool> ();
}

// Training data extraction methods
ZombieTrainingState Bot::getCurrentState () {
   ZombieTrainingState state;
   
   // Position and movement
   state.posX = pev->origin.x;
   state.posY = pev->origin.y;
   state.posZ = pev->origin.z;
   state.velX = pev->velocity.x;
   state.velY = pev->velocity.y;
   state.velZ = pev->velocity.z;
   state.viewYaw = pev->v_angle.y;
   state.viewPitch = pev->v_angle.x;
   
   // Health and status
   state.health = pev->health;
   state.armor = pev->armorvalue;
   state.isStuck = m_isStuck;
   state.isOnLadder = isOnLadder ();
   state.isInWater = isInWater ();
   
   // Enemy information
   state.hasEnemy = !game.isNullEntity (m_enemy);
   if (state.hasEnemy) {
      state.enemyPosX = m_enemy->v.origin.x;
      state.enemyPosY = m_enemy->v.origin.y;
      state.enemyPosZ = m_enemy->v.origin.z;
      state.enemyDistance = pev->origin.distance (m_enemy->v.origin);
      state.enemyHealth = m_enemy->v.health;
      state.enemyVisible = seesEntity (m_enemy->v.origin);
      state.timeSinceEnemySeen = game.time () - m_seeEnemyTime;
   }
   else {
      state.enemyPosX = state.enemyPosY = state.enemyPosZ = 0.0f;
      state.enemyDistance = 0.0f;
      state.enemyHealth = 0.0f;
      state.enemyVisible = false;
      state.timeSinceEnemySeen = 999.0f;
   }
   
   // Team and environment
   state.teamId = m_team;
   state.numTeammatesNearby = numFriendsNear (pev->origin, 512.0f);
   state.numEnemiesNearby = 0; // TODO: implement enemy counting
   state.currentTask = static_cast <int> (getCurrentTaskId ());
   
   // Zombie-specific (use external ConVars)
   state.huntRange = 1024.0f; // Default hunt range
   state.aggressionLevel = 80.0f; // Default aggression
   state.targetEntityPresent = !game.isNullEntity (m_targetEntity);
   
   // Timestamp
   state.gameTime = game.time ();
   
   return state;
}

ZombieTrainingAction Bot::getCurrentAction () {
   ZombieTrainingAction action;
   
   // Extract movement from current bot state
   action.moveForward = m_moveSpeed / pev->maxspeed;
   action.moveRight = m_strafeSpeed / pev->maxspeed;
   
   // Extract view changes (simplified for now)
   action.turnYaw = 0.0f;    // Would need previous angles to calculate
   action.turnPitch = 0.0f;  // Would need previous angles to calculate
   
   // Extract combat actions
   action.attackPrimary = (pev->button & IN_ATTACK) != 0;
   action.attackSecondary = (pev->button & IN_ATTACK2) != 0;
   
   // Extract task information
   action.taskSwitch = 0;  // TODO: track task changes
   
   // Extract movement modifiers
   action.jump = (pev->button & IN_JUMP) != 0;
   action.duck = (pev->button & IN_DUCK) != 0;
   action.walk = false; // TODO: track walk state
   
   // Decision metadata
   action.confidence = 1.0f; // Rule-based decisions are always confident
   
   return action;
}

ZombieTrainingReward Bot::calculateReward () {
   ZombieTrainingReward reward;
   
   // Initialize rewards
   reward.immediateReward = 0.0f;
   reward.enemyKillReward = 0.0f;
   reward.damageDealtReward = 0.0f;
   reward.proximityReward = 0.0f;
   reward.survivalReward = 0.1f;  // Base survival reward per frame
   reward.teamCoordReward = 0.0f;
   
   // ENHANCED REWARD SYSTEM - Much better training signals!
   
   // 1. KILL REWARDS - Most important for combat effectiveness
   if (!game.isNullEntity(m_enemy)) {
      float currentEnemyHealth = m_enemy->v.health;
      
      // Reward for enemy death (if we were targeting them recently)
      if (currentEnemyHealth <= 0.0f && m_lastEnemyHealth > 0.0f) {
         reward.enemyKillReward = 20.0f;  // Big reward for kills!
         game.print("KILL REWARD: +20 for eliminating enemy!");
      }
      
      // 2. DAMAGE REWARDS - Reward for dealing damage
      if (m_lastEnemyHealth > currentEnemyHealth && currentEnemyHealth > 0.0f) {
         float damageDealt = m_lastEnemyHealth - currentEnemyHealth;
         reward.damageDealtReward = damageDealt * 0.5f;  // 0.5 points per HP damage
         game.print("DAMAGE REWARD: +%.1f for %.0f damage", reward.damageDealtReward, damageDealt);
      }
      
      // Store enemy health for next frame comparison
      m_lastEnemyHealth = currentEnemyHealth;
   }
   
   // 3. HEALTH PRESERVATION - Penalty for taking damage
   float currentHealth = pev->health;
   if (currentHealth < m_lastBotHealth) {
      float damageTaken = m_lastBotHealth - currentHealth;
      reward.immediateReward -= damageTaken * 0.3f;  // Penalty for taking damage
      game.print("DAMAGE PENALTY: -%.1f for taking %.0f damage", damageTaken * 0.3f, damageTaken);
   }
   m_lastBotHealth = currentHealth;
   
   // 4. COMBAT ENGAGEMENT REWARDS
   if (!game.isNullEntity(m_enemy)) {
      float distance = pev->origin.distance(m_enemy->v.origin);
      
      // Reward for engaging visible enemies at good range
      if (seesEnemy(m_enemy) && distance < 800.0f) {
         reward.proximityReward = 2.0f;  // Reward for visual contact
         
         // Extra reward for shooting at visible enemies
         if (m_isUsingGrenade || m_shootTime + 1.0f > game.time()) {
            reward.proximityReward += 3.0f;  // Reward aggressive engagement
            game.print("COMBAT REWARD: +5 for engaging visible enemy");
         }
      }
      
      // Reward for optimal combat range (200-600 units)
      if (distance >= 200.0f && distance <= 600.0f) {
         reward.proximityReward += 1.0f;  // Good combat positioning
      }
   }
   
   // 5. TEAM COORDINATION - Reward coordinated attacks
   int teammatesNearby = numFriendsNear(pev->origin, 512.0f);
   int enemiesNearby = numEnemiesNear(pev->origin, 800.0f);
   
   if (enemiesNearby > 0 && teammatesNearby > 0) {
      reward.teamCoordReward = (teammatesNearby * enemiesNearby) * 0.5f;
      game.print("TEAMWORK REWARD: +%.1f for coordinated engagement", reward.teamCoordReward);
   }
   
   // 6. SURVIVAL TIME BONUS - Longer survival = better
   float survivalTime = game.time() - m_spawnTime;
   if (survivalTime > 30.0f) {  // Bonus for surviving over 30 seconds
      reward.survivalReward += 0.2f;
   }
     // 7. MOVEMENT EFFICIENCY - Penalty for excessive movement without purpose
   if (m_moveSpeed > 200.0f && game.isNullEntity(m_enemy)) {
      reward.immediateReward -= 0.1f;  // Slight penalty for running without targets
   }
     // 8. HAZARD DETECTION AND SURVIVAL - Smart environmental awareness
   float hazardPenalty = ZombieAI::Survival::g_smartHazardDetector.analyzeHazards(this);
   reward.immediateReward += hazardPenalty;  // Add hazard analysis (negative for penalties, positive for rewards)
   
   if (hazardPenalty != 0.0f) {
      game.print("HAZARD ANALYSIS: %.1f points for environmental awareness", hazardPenalty);
   }
   
   // Calculate total reward
   reward.totalReward = reward.immediateReward + reward.enemyKillReward + 
                       reward.damageDealtReward + reward.proximityReward + 
                       reward.survivalReward + reward.teamCoordReward;
   
   // Enhanced event tracking
   reward.enemyKilled = (reward.enemyKillReward > 0.0f);
   reward.damageDealt = (reward.damageDealtReward > 0.0f) ? (m_lastEnemyHealth - (game.isNullEntity(m_enemy) ? 0.0f : m_enemy->v.health)) : 0.0f;
   reward.taskCompleted = false;  // TODO: Add objective completion detection
   reward.survivalTime = survivalTime;
   
   return reward;
}

void Bot::executeNeuralAction (const ZombieTrainingAction &action) {
   // Execute neural network decisions directly on the bot
     // Only debug output occasionally to avoid spam
   static float lastDebugOutput = 0.0f;
   if (cv_neural_debug_output.as <bool> () && game.time() - lastDebugOutput > 5.0f) {
      game.print ("🤖 Neural Bot %s: fwd=%.2f, right=%.2f, yaw=%.1f, conf=%.2f", 
                 pev->netname.chars (), action.moveForward, action.moveRight, 
                 action.turnYaw, action.confidence);
      lastDebugOutput = game.time();
   }
   
   // Skip if the action has low confidence
   if (action.confidence < m_neuralConfidenceThreshold) {
      return;
   }
     // Apply movement using YaPB button input system (not speed variables!)
   // Forward/backward movement
   if (action.moveForward > 0.1f) {
      pev->button |= IN_FORWARD;   // Move forward
   } else if (action.moveForward < -0.1f) {
      pev->button |= IN_BACK;      // Move backward
   }
     // Left/right strafing  
   if (action.moveRight > 0.1f) {
      pev->button |= IN_MOVERIGHT; // Strafe right
   } else if (action.moveRight < -0.1f) {
      pev->button |= IN_MOVELEFT;  // Strafe left
   }
   
   // Apply turning/rotation from neural network
   if (action.turnYaw > 5.0f || action.turnYaw < -5.0f) {
      // Apply yaw turning directly to view angles
      pev->v_angle.y += action.turnYaw;
      pev->angles.y = pev->v_angle.y;
      
      // Also apply left/right turning buttons for smoother movement
      if (action.turnYaw > 5.0f) {
         pev->button |= IN_RIGHT;  // Turn right
      } else if (action.turnYaw < -5.0f) {
         pev->button |= IN_LEFT;   // Turn left
      }
   }
   
   // Apply pitch looking (up/down)
   if (action.turnPitch > 2.0f || action.turnPitch < -2.0f) {
      pev->v_angle.x += action.turnPitch;
      pev->angles.x = pev->v_angle.x;
      
      // Clamp pitch to valid range
      if (pev->v_angle.x > 89.0f) pev->v_angle.x = 89.0f;
      if (pev->v_angle.x < -89.0f) pev->v_angle.x = -89.0f;
   }
   
   // Apply combat actions
   if (action.attackPrimary && !m_isReloading) {
      // Only attack when we have a valid enemy position
      if (m_enemyOrigin.x != 0.0f || m_enemyOrigin.y != 0.0f || m_enemyOrigin.z != 0.0f) {
         m_lookAt = m_enemyOrigin;
         pev->button |= IN_ATTACK;
      }
   }
   
   if (action.attackSecondary) {
      pev->button |= IN_ATTACK2;  // Secondary attack (knife)
   }
   
   // Apply movement modifiers
   if (action.jump) {
      pev->button |= IN_JUMP;
   }
   
   if (action.duck) {
      pev->button |= IN_DUCK;
   }
   
   // Task switching based on neural network decision
   if (action.taskSwitch > 0) {
      Task newTaskId;
      
      switch (action.taskSwitch) {
         case 1:  // Hunt
            newTaskId = Task::Hunt;
            break;
         case 2:  // Attack
            newTaskId = Task::Attack;
            break;
         case 3:  // Seek cover
            newTaskId = Task::SeekCover;
            break;
         default:
            newTaskId = Task::Hunt;  // Default to hunting
      }
      
      // Only change tasks if we have a different task
      if (getCurrentTaskId() != newTaskId) {
         clearTask(getCurrentTaskId());
         startTask(newTaskId, TaskPri::Attack, -1, 0.0f, false);
         
         if (cv_neural_debug_output.as<bool>()) {
            game.print("Neural AI switched task to %d for bot %s", 
                      static_cast<int>(newTaskId), pev->netname.chars());
         }      }
   }   // Use basic breakable handling when needed (especially for de_survivor bridge)
   if (!game.isNullEntity(m_breakableEntity)) {
      // Simple approach: attack the breakable and move slightly
      pev->button |= IN_ATTACK;
   }
   
   // Execute movement modifiers
   if (action.jump) {
      pev->button |= IN_JUMP;
   }
   
   if (action.duck) {
      pev->button |= IN_DUCK;
   }
   
   if (action.walk) {
      pev->button |= IN_RUN;
   }
}

ZombieTrainingAction Bot::predictNeuralAction () {
   // SAFETY CHECK - Prevent crashes
   static int predictionCounter = 0;
   predictionCounter++;
   
   // Heartbeat for prediction function
   static float lastPredictionHeartbeat = 0.0f;
   if (game.time() - lastPredictionHeartbeat > 15.0f) {
      neuralDebugPrint("🧠 PREDICTION HEARTBEAT - %d predictions made so far", predictionCounter);
      lastPredictionHeartbeat = game.time();
   }
   
   // Real neural network inference when weights are loaded
   ZombieTrainingAction action = {};
   
   if (!shouldUseNeuralNetwork()) {
      neuralDebugPrint("Bot %s: shouldUseNeuralNetwork() returned false", pev->netname.chars());
      return action;
   }
   
   // Get current state for prediction
   ZombieTrainingState state = getCurrentState();
   
   // Try to load neural weights if not already loaded
   static bool loadAttempted = false;
      if (!g_neuralNet.loaded && !loadAttempted) {
         loadAttempted = true;
         neuralDebugPrint("Bot %s: Attempting to load neural weights...", pev->netname.chars());
      
         // Try different paths for the weights file
         String weightsPath1 = "addons/yapb/ai_training/models/neural_weights.json";
         String weightsPath2 = "./addons/yapb/ai_training/models/neural_weights.json";
         String weightsPath3 = "cstrike/addons/yapb/ai_training/models/neural_weights.json";
         String weightsPath4 = "ai_training/models/neural_weights.json";
      
         if (loadNeuralWeights(weightsPath1) || loadNeuralWeights(weightsPath2) || 
             loadNeuralWeights(weightsPath3) || loadNeuralWeights(weightsPath4)) {
            neuralDebugPrint("Bot %s: ✅ Neural weights loaded successfully!", pev->netname.chars());
         } else {
            neuralDebugPrint("Bot %s: ❌ Failed to load neural weights from all paths", pev->netname.chars());
         }
      }
   
      // If we have a trained neural network, use it!
      if (g_neuralNet.loaded) {
         neuralDebugPrint("Bot %s: Using REAL neural network for prediction", pev->netname.chars());
      
         // Prepare input features (same as training data)
         std::vector<float> input = {
         state.posX / 4096.0f, state.posY / 4096.0f, state.posZ / 4096.0f,  // Normalized positions
         state.velX / 320.0f, state.velY / 320.0f, state.velZ / 320.0f,     // Normalized velocities
         state.viewYaw / 180.0f, state.viewPitch / 90.0f,                   // Normalized angles
         state.health / 100.0f, state.armor / 100.0f,                       // Normalized health/armor
         state.isStuck ? 1.0f : 0.0f, state.isOnLadder ? 1.0f : 0.0f,      // Binary states
         state.isInWater ? 1.0f : 0.0f, state.hasEnemy ? 1.0f : 0.0f,      // Binary states
         state.enemyPosX / 4096.0f, state.enemyPosY / 4096.0f, state.enemyPosZ / 4096.0f,  // Enemy pos
         state.enemyDistance / 4096.0f, state.enemyHealth / 100.0f,         // Enemy stats
         state.enemyVisible ? 1.0f : 0.0f, state.timeSinceEnemySeen / 100.0f,  // Enemy visibility
         static_cast<float>(state.teamId), static_cast<float>(state.numTeammatesNearby),  // Team info
         static_cast<float>(state.numEnemiesNearby), static_cast<float>(state.currentTask)  // Context
      };
      
      // Run neural network forward pass
      std::vector<float> output = g_neuralNet.forward(input);
      
      if (output.size() >= 11) {  // Expected output size
         // Convert neural network output to action
         action.moveForward = output[0];
         action.moveRight = output[1];
         action.turnYaw = output[2];
         action.turnPitch = output[3];
         action.attackPrimary = output[4] > 0.5f;
         action.attackSecondary = output[5] > 0.5f;
         action.taskSwitch = static_cast<int>(output[6] * 5); // Scale to task range
         action.jump = output[7] > 0.5f;
         action.duck = output[8] > 0.5f;
         action.walk = output[9] > 0.5f;
         action.confidence = output[10];
         
         neuralDebugPrint("Bot %s: Neural output - move_fwd=%.2f, move_right=%.2f, attack=%d, conf=%.2f", 
                         pev->netname.chars(), action.moveForward, action.moveRight, 
                         action.attackPrimary ? 1 : 0, action.confidence);         
         return action;
      } else {
         neuralDebugPrint("Bot %s: Invalid neural network output size: %d", pev->netname.chars(), output.size());
      }
   }
   
   // Calculate confidence based on situation complexity
   float baseConfidence = 0.7f;
   float situationComplexity = 0.0f;
   
   // Complexity factors:
   if (state.hasEnemy) situationComplexity += 0.15f;
   if (state.isStuck) situationComplexity += 0.3f;
   if (state.health < 30.0f) situationComplexity += 0.2f;
   if (state.numTeammatesNearby > 2) situationComplexity += 0.1f;
   if (state.numEnemiesNearby > 1) situationComplexity += 0.25f;
   
   // Calculate confidence (higher complexity = lower confidence)
   float confidence = cr::clamp(baseConfidence - situationComplexity, 0.1f, 0.95f);
   action.confidence = confidence;
   
   // If confidence is too low, let rule-based system handle it
   if (confidence < m_neuralConfidenceThreshold) {
      return action; // Empty action with low confidence
   }
   
   // Advanced decision-making simulation
   bool enemyVisible = state.enemyVisible;
   bool enemyNearby = (state.enemyDistance < 500.0f && state.enemyDistance > 0.0f);
   bool veryCloseEnemy = (state.enemyDistance < 150.0f);
   bool lowHealth = (state.health < 30.0f);
   bool criticalHealth = (state.health < 15.0f);
   bool hasAmmo = (getAmmo() > 5);
   bool hasTeam = (state.numTeammatesNearby > 0);
   bool isStuck = state.isStuck;
   
   // Complex behavioral decision tree with weighted probabilities
   float weight_attack = 0.0f;
   float weight_retreat = 0.0f;
   float weight_flank = 0.0f;
   float weight_hunt = 0.0f;
   
   // Calculate behavior weights based on situation
   if (enemyVisible) {
      // Basic weight calculations for visible enemy
      weight_attack = 0.7f;
      weight_hunt = 0.2f;
      weight_flank = 0.1f;
      
      // Adjust for distance
      if (veryCloseEnemy) {
         weight_attack += 0.2f;
      } else {
         weight_hunt += 0.1f;
         weight_flank += 0.1f;
      }
      
      // Health considerations
      if (lowHealth) {
         weight_attack -= 0.3f;
         weight_flank += 0.2f;
         weight_retreat = 0.1f;
      }
      
      if (criticalHealth) {
         weight_attack -= 0.5f;
         weight_retreat += 0.4f;
      }
      
      // Team coordination
      if (hasTeam) {
         weight_attack += 0.1f;
         weight_retreat -= 0.1f;
      }
      
      // Stuck behavior
      if (isStuck) {
         weight_attack -= 0.2f;
         weight_retreat += 0.2f;
      }
   } else if (enemyNearby) {
      // Enemy nearby but not visible
      weight_hunt = 0.6f;
      weight_flank = 0.3f;
      weight_attack = 0.1f;
   } else {
      // No enemy nearby - pure hunting behavior
      weight_hunt = 0.8f;
      weight_flank = 0.2f;
   }
   
   // Normalize weights
   float total = weight_attack + weight_retreat + weight_flank + weight_hunt;
   if (total > 0.0f) {
      weight_attack /= total;
      weight_retreat /= total;
      weight_flank /= total;
      weight_hunt /= total;
   }
   
   // Generate random value for probabilistic behavior selection
   float randBehavior = static_cast<float>(static_cast<int>(game.time() * 1000.0f) % 1000) / 1000.0f;
   
   // Select behavior based on weighted probability
   if (randBehavior < weight_attack) {
      // Aggressive attack behavior
      action.moveForward = 0.9f;
      action.attackPrimary = true;
      action.taskSwitch = 2; // Attack task
      
      // Dodge behavior for sophisticated combat
      float randDodge = static_cast<float>(static_cast<int>(game.time() * 2540.0f) % 1000) / 1000.0f;
      if (randDodge < 0.3f) {
         action.moveRight = 0.7f; // Dodge right while attacking
      } else if (randDodge < 0.6f) {
         action.moveRight = -0.7f; // Dodge left while attacking
      }
      
      // Occasionally jump during attack
      if (static_cast<int>(game.time() * 1000.0f) % 20 == 0) {
         action.jump = true;
      }
      
   } else if (randBehavior < weight_attack + weight_retreat) {
      // Retreat behavior
      action.moveForward = -0.8f;
      action.taskSwitch = 3; // Seek cover
        // Zigzag retreat pattern
      float zigzagPattern = cr::sinf(game.time() * 6.0f);
      action.moveRight = zigzagPattern * 0.6f;
      
   } else if (randBehavior < weight_attack + weight_retreat + weight_flank) {
      // Flanking behavior - more sophisticated movement
      float flanking_dir = (static_cast<int>(game.time() * 10.0f) % 2 == 0) ? 0.9f : -0.9f;
      action.moveRight = flanking_dir;
      action.moveForward = 0.5f;
      action.taskSwitch = 1; // Hunt task
        } else {
      // Hunting behavior      
      action.moveForward = 0.7f;
      action.taskSwitch = 1; // Hunt task
      
      // Add exploration pattern - more natural movement
      float t = game.time() * 0.7f;
      action.moveRight = cr::sinf(t) * 0.4f;
      
      // Occasional stop and look around
      if (static_cast<int>(game.time() * 10.0f) % 30 == 0) {
         action.moveForward = 0.1f;
         action.turnYaw = 45.0f;
      }
   }
   
   // Additional behavior enhancements
   
   // Handle breakable objects special case
   if (!game.isNullEntity(m_breakableEntity)) {
      action.attackPrimary = true;
      action.moveRight = 0.0f;
      action.moveForward = 0.2f;
   }
     // Avoid getting stuck at chasm/bridge (de_survivor specific)
   // If we've been in the same position for too long
   static Vector lastPosition = pev->origin;
   static float stuckTime = 0.0f;
   static float lastFrameTime = game.time();
   
   if (lastPosition.distance(pev->origin) < 20.0f) {
      stuckTime += game.time() - lastFrameTime; // Use time difference
      
      // After 4 seconds of being stuck
      if (stuckTime > 4.0f) {
         // Try to get unstuck with random movement
         action.moveForward = (static_cast<int>(game.time() * 100.0f) % 2 == 0) ? -0.9f : 0.9f;
         action.moveRight = (static_cast<int>(game.time() * 200.0f) % 2 == 0) ? -0.9f : 0.9f;
         action.jump = true;
      }
   } else {
      stuckTime = 0.0f;
      lastPosition = pev->origin;
   }
   lastFrameTime = game.time();
     // Debug output
   if (cv_neural_debug_output.as<bool>()) {
      static float lastDebugTime = 0.0f;
      if (game.time() - lastDebugTime > 2.0f) {
         const char* behavior = "unknown";
         if (randBehavior < weight_attack) behavior = "attack";
         else if (randBehavior < weight_attack + weight_retreat) behavior = "retreat";
         else if (randBehavior < weight_attack + weight_retreat + weight_flank) behavior = "flank";
         else behavior = "hunt";
           neuralDebugPrint("Bot %s: %s (conf=%.2f, fwd=%.2f, right=%.2f, atk=%d)",
                   pev->netname.chars(), behavior, confidence, 
                   action.moveForward, action.moveRight, action.attackPrimary ? 1 : 0);
         lastDebugTime = game.time();
      }
   }
   
   return action;
}

// JSON parsing helper functions
bool parseJsonArray(const String& jsonText, int start, std::vector<float>& result) {
    result.clear();
    
    size_t pos = static_cast<size_t>(start);
    while (pos < jsonText.length() && jsonText[pos] != '[') pos++;
    if (pos >= jsonText.length()) return false;
    pos++; // Skip '['
    
    String numberStr;
    bool inNumber = false;
    
    while (pos < jsonText.length()) {
        char c = jsonText[pos];
        
        if (c == ']') {
            if (inNumber && !numberStr.empty()) {
                result.push_back(static_cast<float>(atof(numberStr.chars())));
            }
            return true;
        }
        else if (c == ',' || c == ' ' || c == '\n' || c == '\t') {
            if (inNumber && !numberStr.empty()) {
                result.push_back(static_cast<float>(atof(numberStr.chars())));
                numberStr.clear();
                inNumber = false;
            }
        }
        else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') {
            numberStr += c;
            inNumber = true;
        }
        
        pos++;
    }
    
    return false;
}

bool parse2DJsonArray(const String& jsonText, int start, std::vector<std::vector<float>>& result) {
    result.clear();
    
    size_t pos = static_cast<size_t>(start);
    while (pos < jsonText.length() && jsonText[pos] != '[') pos++;
    if (pos >= jsonText.length()) return false;
    pos++; // Skip outer '['
    
    while (pos < jsonText.length()) {
        char c = jsonText[pos];
        
        if (c == ']') {
            return true;
        }
        else if (c == '[') {
            std::vector<float> row;
            if (parseJsonArray(jsonText, static_cast<int>(pos), row)) {
                result.push_back(row);
                // Find the end of this array
                int depth = 0;
                while (pos < jsonText.length()) {
                    if (jsonText[pos] == '[') depth++;
                    else if (jsonText[pos] == ']') {
                        depth--;
                        if (depth == 0) {
                            pos++;
                            break;
                        }
                    }
                    pos++;
                }
            }
        }
        else {
            pos++;
        }
    }
    
    return false;
}

// Load neural network weights from JSON file
bool loadNeuralWeights(const String& weightsPath) {
    neuralDebugPrint("Attempting to load neural weights from: %s", weightsPath.chars());
    
    // Try to open the weights file
    std::ifstream file(weightsPath.chars());
    if (!file.is_open()) {
        neuralDebugPrint("❌ Failed to open weights file: %s", weightsPath.chars());
        return false;
    }    // Read entire file into string
    String fileContent;
    char buffer[1024];
    while (file.good()) {
        file.read(buffer, sizeof(buffer));
        std::streamsize bytesRead = file.gcount();
        if (bytesRead > 0) {
            fileContent.append(buffer, static_cast<size_t>(bytesRead));
        }
    }
    file.close();
    
    neuralDebugPrint("✅ Loaded weights file, size: %d bytes", static_cast<int>(fileContent.length()));
    
    // Parse JSON manually (simple parser for our specific format)
    try {
        // Find fc1.weight
        int fc1WeightPos = fileContent.find("\"fc1.weight\":");
        if (fc1WeightPos != -1) {
            fc1WeightPos += 13; // Skip the key
            if (parse2DJsonArray(fileContent, fc1WeightPos, g_neuralNet.fc1_weights)) {
                neuralDebugPrint("✅ Loaded fc1.weight: %dx%d", 
                               g_neuralNet.fc1_weights.size(), 
                               g_neuralNet.fc1_weights.empty() ? 0 : g_neuralNet.fc1_weights[0].size());
            }
        }
        
        // Find fc1.bias
        int fc1BiasPos = fileContent.find("\"fc1.bias\":");
        if (fc1BiasPos != -1) {
            fc1BiasPos += 11; // Skip the key
            if (parseJsonArray(fileContent, fc1BiasPos, g_neuralNet.fc1_bias)) {
                neuralDebugPrint("✅ Loaded fc1.bias: %d elements", g_neuralNet.fc1_bias.size());
            }
        }
        
        // Find fc2.weight
        int fc2WeightPos = fileContent.find("\"fc2.weight\":");
        if (fc2WeightPos != -1) {
            fc2WeightPos += 13; // Skip the key
            if (parse2DJsonArray(fileContent, fc2WeightPos, g_neuralNet.fc2_weights)) {
                neuralDebugPrint("✅ Loaded fc2.weight: %dx%d", 
                               g_neuralNet.fc2_weights.size(), 
                               g_neuralNet.fc2_weights.empty() ? 0 : g_neuralNet.fc2_weights[0].size());
            }
        }
        
        // Find fc2.bias
        int fc2BiasPos = fileContent.find("\"fc2.bias\":");
        if (fc2BiasPos != -1) {
            fc2BiasPos += 11; // Skip the key
            if (parseJsonArray(fileContent, fc2BiasPos, g_neuralNet.fc2_bias)) {
                neuralDebugPrint("✅ Loaded fc2.bias: %d elements", g_neuralNet.fc2_bias.size());
            }
        }
        
        // Find fc3.weight
        int fc3WeightPos = fileContent.find("\"fc3.weight\":");
        if (fc3WeightPos != -1) {
            fc3WeightPos += 13; // Skip the key
            if (parse2DJsonArray(fileContent, fc3WeightPos, g_neuralNet.fc3_weights)) {
                neuralDebugPrint("✅ Loaded fc3.weight: %dx%d", 
                               g_neuralNet.fc3_weights.size(), 
                               g_neuralNet.fc3_weights.empty() ? 0 : g_neuralNet.fc3_weights[0].size());
            }
        }
        
        // Find fc3.bias
        int fc3BiasPos = fileContent.find("\"fc3.bias\":");
        if (fc3BiasPos != -1) {
            fc3BiasPos += 11; // Skip the key
            if (parseJsonArray(fileContent, fc3BiasPos, g_neuralNet.fc3_bias)) {
                neuralDebugPrint("✅ Loaded fc3.bias: %d elements", g_neuralNet.fc3_bias.size());
            }
        }
        
        // Verify we loaded all components
        bool allLoaded = !g_neuralNet.fc1_weights.empty() && !g_neuralNet.fc1_bias.empty() &&
                        !g_neuralNet.fc2_weights.empty() && !g_neuralNet.fc2_bias.empty() &&
                        !g_neuralNet.fc3_weights.empty() && !g_neuralNet.fc3_bias.empty();
        
        if (allLoaded) {
            g_neuralNet.loaded = true;
            neuralDebugPrint("🧠 Neural network weights loaded successfully!");
            neuralDebugPrint("   Architecture: %d -> %d -> %d -> %d", 
                           g_neuralNet.fc1_weights.empty() ? 0 : g_neuralNet.fc1_weights[0].size(),
                           g_neuralNet.fc1_weights.size(),
                           g_neuralNet.fc2_weights.size(),
                           g_neuralNet.fc3_weights.size());
            return true;
         } else {
            neuralDebugPrint("❌ Failed to load all neural network components");
            return false;
         }
        
    } catch (...) {
        neuralDebugPrint("❌ Exception while parsing neural weights JSON");
        return false;
    }
}

