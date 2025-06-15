import win32pipe
import win32file
import struct
import time

class BotExternalControl:
    def __init__(self):
        self.pipe = None
        self.pipe_name = r'\\.\pipe\yapb_control'
    
    def connect(self):
        try:
            self.pipe = win32file.CreateFile(
                self.pipe_name,
                win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            print(f"Connected to pipe: {self.pipe_name}")
            return True
        except Exception as e:
            print(f"Failed to connect to pipe: {e}")
            return False
    
    def send_command(self, forward=0.0, side=0.0, yaw=0.0, pitch=0.0, jump=False, duck=False, attack1=False, attack2=False, reload=False, weapon=-1):
        if not self.pipe:
            return False
        
        try:
            command_data = struct.pack('ffffbbbbbi', forward, side, yaw, pitch, jump, duck, attack1, attack2, reload, weapon)
            win32file.WriteFile(self.pipe.handle, command_data)
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False
    
    def send_movement(self, forward=0.0, side=0.0, jump=False, duck=False):
        return self.send_command(forward=forward, side=side, jump=jump, duck=duck)
    
    def send_angles(self, pitch=0.0, yaw=0.0):
        return self.send_command(yaw=yaw, pitch=pitch)    
    def send_buttons(self, attack1=False, attack2=False, reload=False):
        return self.send_command(attack1=attack1, attack2=attack2, reload=reload)
    
    def send_weapon(self, weapon_id):
        return self.send_command(weapon=weapon_id)
    
    def disconnect(self):
        if self.pipe:
            win32file.CloseHandle(self.pipe.handle)
            self.pipe = None

def test_basic_control():
    controller = BotExternalControl()
    
    if not controller.connect():
        print("Cannot connect to bot pipe. Make sure YaPB is running.")
        return
    
    print("Testing external control...")
    
    print("1. Testing forward movement...")
    controller.send_movement(forward=1.0)
    time.sleep(2)
    
    print("2. Testing backward movement...")
    controller.send_movement(forward=-1.0)
    time.sleep(2)
    
    print("3. Testing left strafe...")
    controller.send_movement(side=-1.0)
    time.sleep(2)
    
    print("4. Testing right strafe...")
    controller.send_movement(side=1.0)
    time.sleep(2)
    
    print("5. Testing jump...")
    controller.send_movement(jump=True)
    time.sleep(1)
    
    print("6. Testing duck...")
    controller.send_movement(duck=True)
    time.sleep(2)
    
    print("7. Testing view angles...")
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        controller.send_angles(yaw=angle)
        time.sleep(0.5)
    
    print("8. Testing attack...")
    controller.send_buttons(attack1=True)
    time.sleep(1)
    controller.send_buttons(attack1=False)
    
    print("9. Stop all movement...")
    controller.send_movement()
    time.sleep(1)
    
    controller.disconnect()
    print("Test completed.")

if __name__ == "__main__":
    test_basic_control()
