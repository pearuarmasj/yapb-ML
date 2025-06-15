//
// YaPB External Control Interface
// Simple named pipe communication for ML bot control
//

#include <yapb.h>
#include <windows.h>

class BotExternalControl final : public Singleton <BotExternalControl> {
private:
   HANDLE m_pipe {};
   bool m_connected {};
   
   struct ExternalCommand {
      float forward;     // -1.0 to 1.0
      float side;        // -1.0 to 1.0  
      float yaw;         // view angle yaw
      float pitch;       // view angle pitch
      bool jump;
      bool duck;
      bool attack1;
      bool attack2;
      bool reload;
      int weapon;        // weapon ID to switch to (-1 = no change)
   };
   
public:
   bool init ();
   void shutdown ();
   bool checkForCommands ();
   void applyCommandToBot (Bot *bot, const ExternalCommand &cmd);
   
private:
   bool readCommand (ExternalCommand &cmd);
};

// Global instance
BotExternalControl &extControl = BotExternalControl::instance ();

bool BotExternalControl::init () {
   // Create named pipe for communication
   m_pipe = CreateNamedPipeA(
      "\\\\.\\pipe\\yapb_control",
      PIPE_ACCESS_DUPLEX,      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
      PIPE_UNLIMITED_INSTANCES,
      0,
      sizeof(ExternalCommand),
      0,
      nullptr
   );   if (m_pipe == INVALID_HANDLE_VALUE) {
      return false;
   }
   
   m_connected = false;
   return true;
}

void BotExternalControl::shutdown () {
   if (m_pipe != INVALID_HANDLE_VALUE) {
      CloseHandle (m_pipe);
      m_pipe = INVALID_HANDLE_VALUE;
   }
   m_connected = false;
}

bool BotExternalControl::checkForCommands () {
   if (m_pipe == INVALID_HANDLE_VALUE) {
      return false;
   }
   
   // Check for client connection
   if (!m_connected) {
      if (ConnectNamedPipe (m_pipe, nullptr) || GetLastError () == ERROR_PIPE_CONNECTED) {
         m_connected = true;
      }
      return false;
   }
   
   return true;
}

bool BotExternalControl::readCommand (ExternalCommand &cmd) {
   if (!m_connected || m_pipe == INVALID_HANDLE_VALUE) {
      return false;
   }
   
   DWORD bytesRead = 0;
   if (!ReadFile (m_pipe, &cmd, sizeof(cmd), &bytesRead, nullptr)) {
      // Client disconnected
      DisconnectNamedPipe (m_pipe);
      m_connected = false;
      return false;
   }
   
   return bytesRead == sizeof(cmd);
}

void BotExternalControl::applyCommandToBot (Bot *bot, const ExternalCommand &cmd) {
   if (!bot || !bot->isUnderExternalControl ()) {
      return;
   }
   
   // Apply movement
   bot->setExternalMovement (cmd.forward, cmd.side, cmd.jump, cmd.duck);
   
   // Apply view angles
   Vector angles (cmd.pitch, cmd.yaw, 0.0f);
   bot->setExternalAngles (angles);
   
   // Apply buttons
   bot->setExternalButtons (cmd.attack1, cmd.attack2, cmd.reload);
   
   // Apply weapon switch
   if (cmd.weapon >= 0) {
      bot->setExternalWeapon (cmd.weapon);
   }
}

// Function to enable external control on first bot
void enableExternalControlOnBot () {
   for (const auto &bot : bots) {
      if (bot && bot->m_isAlive) {
         bot->enableExternalControl (true);
         break;
      }
   }
}

// Function to disable external control on all bots
void disableExternalControlOnAllBots () {
   for (const auto &bot : bots) {
      if (bot && bot->isUnderExternalControl ()) {
         bot->enableExternalControl (false);
      }
   }
}

void initExternalControl () {
   extControl.init ();
}

void checkExternalControlCommands () {
   extControl.checkForCommands ();
}
