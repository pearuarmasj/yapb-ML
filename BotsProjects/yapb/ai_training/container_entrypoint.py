#!/usr/bin/env python3

import os
import subprocess
import time
import sys
import socket

def start_xvfb():
    """Start virtual display"""
    # Generate unique display number based on hostname
    hostname = socket.gethostname()
    display_num = 99
    if hostname:
        # Extract number from hostname (e.g., ai_training-bot-1 -> 1)
        import re
        match = re.search(r'-(\d+)$', hostname)
        if match:
            display_num = 99 + int(match.group(1))    
    display = f":{display_num}"
    os.environ['DISPLAY'] = display
    
    cmd = ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp", "-ac", "+extension", "GLX"]
    subprocess.Popen(cmd)
    time.sleep(2)
    
    # Create dummy Xauthority file
    subprocess.run(["touch", "/root/.Xauthority"], check=True)
    subprocess.run(["xauth", "add", display, ".", "1234567890abcdef"], check=True)
    
def start_assaultcube():
    """Start AssaultCube in background on specific empty map"""
    os.chdir('/opt/assaultcube')
    
    # Start AssaultCube and then execute commands via config
    cmd = ["./assaultcube.sh", "--home=/root/.assaultcube/v1.3", "--init"]
    proc = subprocess.Popen(cmd)
    time.sleep(8)
    
    # Send commands to load map and configure
    time.sleep(2)
    subprocess.run(["xdotool", "type", "map ac_depot"], check=False)
    subprocess.run(["xdotool", "key", "Return"], check=False)
    time.sleep(1)
    subprocess.run(["xdotool", "type", "maxclients 1"], check=False)
    subprocess.run(["xdotool", "key", "Return"], check=False)
    time.sleep(1)

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
    start_xvfb()
    start_assaultcube()
    
    # Check GPU availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("GPU detected and available")
        else:
            print("GPU not available, using CPU rendering")
    except:
        print("nvidia-smi not found, using CPU rendering")
    
    # Run the main experiments script with menu
    subprocess.run(['python', '/app/container_experiments.py'], cwd='/app')
