#!/usr/bin/env python3

import os
import subprocess
import time
import sys
import socket
import random
import platform

# Global variable to track AssaultCube process
assaultcube_process = None

def get_unique_instance_id():
    """Generate unique instance ID for this container"""
    try:
        # Use hostname if available
        hostname = platform.node()
        # Add a random component to ensure uniqueness
        rand_id = random.randint(1000, 9999)
        return f"{hostname}_{rand_id}"
    except:
        # Fallback to just random ID
        return f"bot_{random.randint(1000, 9999)}"

def find_free_vnc_port(start_port=5900, max_port=5920):
    """Find a free VNC port"""
    for port in range(start_port, max_port + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:
                return port
        except:
            continue
    return 5900

def start_vnc_server(display, display_num):   
    """Start VNC server"""
    vnc_port = find_free_vnc_port()
    print(f"Starting VNC on port {vnc_port}")
    
    subprocess.run(["mkdir", "-p", "/root/.vnc"], check=True)
    
    vnc_cmd = [
        "x11vnc", 
        "-display", display,
        "-rfbport", str(vnc_port),
        "-listen", "0.0.0.0",
        "-forever",
        "-nopw",
        "-shared",
        "-bg"
    ]
    
    subprocess.Popen(vnc_cmd)
    time.sleep(3)
    
    return vnc_port

def start_xvfb():
    """Start virtual display, window manager, and VNC server"""
    display_num = 100 + (os.getpid() % 100)
    display = f":{display_num}"
    os.environ['DISPLAY'] = display
    
    print(f"Starting Xvfb on display {display}")
    # Start Xvfb with proper extensions for input handling
    cmd = ["Xvfb", display, "-screen", "0", "1920x1080x24", "-ac", "+extension", "RANDR"]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    # Start window manager (fluxbox) for proper window focus and input handling  
    print(f"Starting fluxbox window manager on display {display}")
    fluxbox_cmd = ["fluxbox", "-display", display]
    subprocess.Popen(fluxbox_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # Set up X11 input method environment
    os.environ['XMODIFIERS'] = '@im=none'
    os.environ['GTK_IM_MODULE'] = 'xim'
    os.environ['QT_IM_MODULE'] = 'xim'
    
    # Start VNC server
    vnc_port = start_vnc_server(display, display_num)
    
    # Save VNC info
    instance_id = os.environ.get('INSTANCE_ID', get_unique_instance_id())
    with open(f"/data/vnc_info_{instance_id}.txt", "w") as f:
        f.write(f"VNC Port: {vnc_port}\nDisplay: {display}\n")
    
    return display_num
    
def start_assaultcube(display_num):
    """Start AssaultCube using the compiled source"""
    os.chdir('/root/assaultcube')
    print(f"Starting AssaultCube on display :{display_num}")    # Start AssaultCube with hardware GPU acceleration
    env = os.environ.copy()
    
    # Disable all audio systems comprehensively
    env['SDL_AUDIODRIVER'] = 'dummy'
    env['ALSA_DISABLE'] = '1'
    env['PULSE_DISABLE'] = '1'
    env['OPENAL_DISABLE'] = '1'
    env['PULSE_RUNTIME_PATH'] = '/dev/null'
    env['PULSE_CONFIG_PATH'] = '/dev/null'
    env['ALSA_CONFIG_PATH'] = '/dev/null'
    env['ALSA_MIXER_SIMPLE'] = '1'
    env['SDL_AUDIO_DRIVER'] = 'dummy'
    env['AUDIODEV'] = '/dev/null'
    env['ALSA_PCM_CARD'] = 'null'
    env['ALSA_CTL_CARD'] = 'null'
      # Launch AssaultCube with wrapper script for complete audio disabling
    cmd = ["/app/launch_assaultcube.sh", "/root/assaultcube/bin_unix/native_client", "--home=/root/.assaultcube/v1.3", "--init", "-nosound"]
    print(f"Launching AssaultCube with command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Using hardware GPU acceleration")
    
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    global assaultcube_process
    assaultcube_process = proc
    print(f"AssaultCube started with PID: {proc.pid}")
    
    # Monitor AssaultCube output for first few seconds
    print("Monitoring AssaultCube startup...")
    start_time = time.time()
    while time.time() - start_time < 10:
        if proc.poll() is not None:
            print(f"AssaultCube process exited with code: {proc.returncode}")
            stdout, stderr = proc.communicate()
            if stdout:
                print(f"AssaultCube output: {stdout}")
            break
        time.sleep(0.5)
    
    # Wait for AssaultCube window to appear
    time.sleep(5)
    
    # Test xdotool functionality
    try:
        result = subprocess.run(["xdotool", "search", "--onlyvisible", "--class", "AssaultCube"], 
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            window_id = result.stdout.strip().split('\n')[0]
            print(f"Found AssaultCube window: {window_id}")
            
            # Try to focus the window
            subprocess.run(["xdotool", "windowfocus", window_id], check=False)
            print("AssaultCube window focused")
        else:
            print("AssaultCube window not found by xdotool")
            # List all windows for debugging
            all_windows = subprocess.run(["xdotool", "search", "--onlyvisible", "."], 
                                        capture_output=True, text=True, timeout=5)
            print(f"All visible windows: {all_windows.stdout}")
            
    except subprocess.TimeoutExpired:
        print("xdotool search timed out")
    except Exception as e:
        print(f"xdotool test failed: {e}")
    
    return proc

def send_map_command():
    """Send map command to AssaultCube to set ac_desert for 60 minutes"""
    print("Sending map command to AssaultCube...")
    time.sleep(10)  # Wait for AssaultCube to fully load
    
    try:
        # Find AssaultCube window
        result = subprocess.run(["xdotool", "search", "--onlyvisible", "--class", "AssaultCube"], 
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            window_id = result.stdout.strip().split('\n')[0]
            
            # Focus the window
            subprocess.run(["xdotool", "windowfocus", window_id], check=False)
            time.sleep(1)
            
            # Open chat with T key
            subprocess.run(["xdotool", "key", "t"], check=False)
            time.sleep(0.5)
            
            # Type the map command
            subprocess.run(["xdotool", "type", "/map ac_desert 60"], check=False)
            time.sleep(0.5)
            
            # Press Enter to execute
            subprocess.run(["xdotool", "key", "Return"], check=False)
            
            print("Map command sent successfully: /map ac_desert 60")
            
        else:
            print("Could not find AssaultCube window for map command")
            
    except Exception as e:
        print(f"Failed to send map command: {e}")

def check_assaultcube_running(proc):
    """Simple check if AssaultCube is still running"""
    return proc is not None and proc.poll() is None

def run_data_collection():
    """Run data collection script"""
    os.chdir('/app')
    
    # Set environment for container mode
    os.environ['CONTAINER_MODE'] = '1'
    os.environ['CAPTURE_REGION'] = '0,0,1920,1080'
    
    # Disable pyautogui failsafe and display requirements
    os.environ['PYAUTOGUI_DISABLE_DISPLAY'] = '1'
    
    # Import and run data collection
    sys.path.append('/app')
    from container_experiments import AssaultCubeDataCollector
    
    region = {
        'left': 0,
        'top': 0, 
        'width': 1920,
        'height': 1080
    }
    
    data_dir = os.environ.get('DATA_DIR', '/data')
    instance_id = os.environ.get('INSTANCE_ID', 'container')
    
    collector = AssaultCubeDataCollector(
        region=region, 
        data_dir=f"{data_dir}/instance_{instance_id}"
    )
    
    print(f"Container {instance_id} starting data collection...")
    
    obs = collector.reset()
    total_samples = 10000
    
    for step in range(total_samples):
        action = __import__('random').randint(0, 11)
        obs, reward, done, info = collector.step(action)
        
        if done:
            obs = collector.reset()
            
        if step % 1000 == 0:
            print(f"Container {instance_id}: {step}/{total_samples} samples")
    
    collector.save_data()
    print(f"Container {instance_id} completed data collection")

if __name__ == "__main__":
    print("Starting container...")
      # Set unique instance ID if not already set
    if 'INSTANCE_ID' not in os.environ:
        os.environ['INSTANCE_ID'] = get_unique_instance_id()
    
    instance_id = os.environ['INSTANCE_ID']
    print(f"Container instance ID: {instance_id}")
    
    display_num = start_xvfb()
    assaultcube_process = start_assaultcube(display_num)
    
    # Send map command to set ac_desert for 60 minutes
    send_map_command()
    
    # Run the main experiments script with mode from environment
    bot_mode = os.environ.get('BOT_MODE', 'collect')
    print(f"Starting bot in mode: {bot_mode}")
    
    # Wait for AssaultCube to start
    time.sleep(10)
    
    if bot_mode == 'vnc':
        print("VNC mode: Container will stay running for manual access")
        print("Check /data/vnc_info_*.txt for connection details")
        
        if check_assaultcube_running(assaultcube_process):
            print("AssaultCube is running, starting ML data collection/training")
            run_data_collection()
        else:
            print("AssaultCube is not running, keeping container alive for debugging...")
            try:
                while True:
                    time.sleep(60)
                    status = "RUNNING" if check_assaultcube_running(assaultcube_process) else "STOPPED"
                    print(f"Status check - AssaultCube: {status}")
            except KeyboardInterrupt:
                print("Container shutting down...")
                
    elif bot_mode == 'collect':
        print("Data collection mode: Starting ML data collection")
        
        if check_assaultcube_running(assaultcube_process):
            print("AssaultCube is running, starting data collection")
            run_data_collection()
        else:
            print("AssaultCube is not running, cannot start data collection")
            
    else:
        print(f"Unknown mode: {bot_mode}, running data collection anyway")
        run_data_collection()
