#!/usr/bin/env python3

import os
import subprocess
import time
import sys
import socket
import random
import platform

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

def start_xvfb():
    """Start virtual display"""
    # Generate unique display number based on container process ID and random
    base_display = 100 + (os.getpid() % 800)  # Use PID for better uniqueness
    display_num = base_display + random.randint(0, 99)
    
    # Check if display is already in use, increment if needed
    max_attempts = 50
    attempts = 0
    while os.path.exists(f"/tmp/.X{display_num}-lock") and attempts < max_attempts:
        display_num += 1
        if display_num > 999:
            display_num = 100
        attempts += 1
    
    if attempts >= max_attempts:
        print(f"Warning: Could not find free display after {max_attempts} attempts, using {display_num}")
    
    display = f":{display_num}"
    os.environ['DISPLAY'] = display
    
    print(f"Starting Xvfb on display {display} (PID: {os.getpid()})")    
    # Start Xvfb with MIT-SHM disabled to avoid shared memory issues
    cmd = ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp", "-ac", 
           "+extension", "GLX", "-extension", "MIT-SHM"]
    proc = subprocess.Popen(cmd)
    time.sleep(3)
    
    # Create dummy Xauthority file
    subprocess.run(["touch", "/root/.Xauthority"], check=True)
    subprocess.run(["xauth", "add", display, ".", "1234567890abcdef"], check=True)
    
    # Start VNC server on unique port based on display number
    vnc_port = 5900 + display_num
    print(f"Starting VNC server on port {vnc_port}")
    vnc_cmd = ["x11vnc", "-display", display, "-rfbport", str(vnc_port), 
               "-forever", "-shared", "-bg", "-nopw"]
    subprocess.Popen(vnc_cmd)
    time.sleep(2)
    
    # Save VNC connection info for user
    instance_id = os.environ.get('INSTANCE_ID', get_unique_instance_id())
    with open(f"/data/vnc_info_{instance_id}.txt", "w") as f:
        f.write(f"Instance: {instance_id}\n")
        f.write(f"Display: {display}\n")
        f.write(f"VNC Port: {vnc_port}\n")
        f.write(f"Container IP: You can find this with: docker inspect <container_name>\n")
        f.write(f"To connect: Use VNC viewer to connect to <container_ip>:{vnc_port}\n")
    
    print(f"VNC connection info saved to /data/vnc_info_{instance_id}.txt")
    return display_num
    
def start_assaultcube(display_num):
    """Start AssaultCube in background on specific empty map"""
    os.chdir('/opt/assaultcube')
    
    print(f"Starting AssaultCube on display :{display_num}")
      # Create a config file to force GPU rendering and set map
    config_content = """
// Load specific map
map ac_depot
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
    
    # Set unique instance ID if not already set
    if 'INSTANCE_ID' not in os.environ:
        os.environ['INSTANCE_ID'] = get_unique_instance_id()
    
    instance_id = os.environ['INSTANCE_ID']
    print(f"Container instance ID: {instance_id}")
    
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
    
    # Run the main experiments script with mode from environment
    bot_mode = os.environ.get('BOT_MODE', 'collect')
    print(f"Starting bot in mode: {bot_mode}")
    subprocess.run(['python3', '/app/container_experiments.py', bot_mode], cwd='/app')
