#!/usr/bin/env python3

import os
import subprocess
import time
import sys
import socket

def start_xvfb():
    """Start virtual display"""
    # Generate unique display number based on container ID
    import random
    display_num = random.randint(100, 999)
    
    # Check if display is already in use, increment if needed
    while os.path.exists(f"/tmp/.X{display_num}-lock"):
        display_num += 1
        if display_num > 999:
            display_num = 100
    
    display = f":{display_num}"
    os.environ['DISPLAY'] = display
    
    print(f"Starting Xvfb on display {display}")
    
    # Start Xvfb with MIT-SHM disabled to avoid shared memory issues
    cmd = ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp", "-ac", 
           "+extension", "GLX", "-extension", "MIT-SHM"]
    proc = subprocess.Popen(cmd)
    time.sleep(3)
    
    # Create dummy Xauthority file
    subprocess.run(["touch", "/root/.Xauthority"], check=True)
    subprocess.run(["xauth", "add", display, ".", "1234567890abcdef"], check=True)
    
    return display_num
    
def start_assaultcube(display_num):
    """Start AssaultCube in background on specific empty map"""
    os.chdir('/opt/assaultcube')
    
    print(f"Starting AssaultCube on display :{display_num}")
    
    # Create a config file to force GPU rendering and set map
    config_content = """
// Force GPU acceleration
r_vsync 0
maxfps 0

// Load specific map
map ac_depot
maxclients 1
"""
    
    config_dir = "/root/.assaultcube/v1.3/config"
    os.makedirs(config_dir, exist_ok=True)
    
    with open(f"{config_dir}/autoexec.cfg", "w") as f:
        f.write(config_content)
    
    # Start AssaultCube with GPU acceleration forced
    env = os.environ.copy()
    env['LIBGL_ALWAYS_SOFTWARE'] = '0'
    env['NVIDIA_VISIBLE_DEVICES'] = 'all'
    env['NVIDIA_DRIVER_CAPABILITIES'] = 'all'
    
    cmd = ["./assaultcube.sh", "--home=/root/.assaultcube/v1.3", "--init"]
    proc = subprocess.Popen(cmd, env=env)
    time.sleep(10)
    
    print("AssaultCube started, waiting for initialization...")

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
    display_num = start_xvfb()
    start_assaultcube(display_num)
    
    # Check GPU availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("GPU detected and available")
            print(result.stdout)
        else:
            print("GPU not available, using CPU rendering")
    except:
        print("nvidia-smi not found, using CPU rendering")
    
    # Run the main experiments script with menu
    subprocess.run(['python', '/app/container_experiments.py'], cwd='/app')
