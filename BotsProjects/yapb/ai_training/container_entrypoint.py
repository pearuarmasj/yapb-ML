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

def find_free_vnc_port(start_port=5900, max_port=5950):
    """Find a free VNC port - use PID as offset for uniqueness"""
    # Use process ID as offset to reduce collisions
    pid_offset = os.getpid() % 50
    start_port += pid_offset
    
    for port in range(start_port, max_port + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:  # Port is free
                return port
        except:
            continue
    
    # If no port found with offset, try from beginning
    for port in range(5900, max_port + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:  # Port is free
                return port
        except:
            continue
            
    return 5900  # Fallback

def start_vnc_server(display, display_num):   
    """Start VNC server with proper configuration"""
    # Add startup delay to avoid port conflicts when scaling
    startup_delay = random.randint(1, 10)
    print(f"Waiting {startup_delay} seconds to avoid port conflicts...")
    time.sleep(startup_delay)
      # Use display number to calculate VNC port but keep within forwarded range
    vnc_port = 5900 + ((display_num - 100) % 50)  # Ensure port stays in 5900-5950 range
    print(f"Display number: {display_num}, VNC port: {vnc_port}")
    print(f"Using VNC port: {vnc_port} (based on display {display_num})")
    
    # Kill any existing VNC servers first
    subprocess.run(["pkill", "-f", "vnc"], check=False)
    subprocess.run(["pkill", "-f", "x11vnc"], check=False)
    time.sleep(2)    # Create VNC directory and setup
    subprocess.run(["mkdir", "-p", "/root/.vnc"], check=True)
    
    vnc_cmd = [
        "x11vnc", 
        "-display", display,
        "-rfbport", str(vnc_port),
        "-listen", "0.0.0.0",
        "-forever",
        "-nopw",
        "-shared",
        "-bg",
        "-o", "/data/vnc_output.log"
    ]
    
    print(f"Starting VNC with command: {' '.join(vnc_cmd)}")
    # Start VNC server in background using Popen
    proc = subprocess.Popen(vnc_cmd)    
    time.sleep(8)
    
    # Verify VNC is running with better checks
    vnc_running = False
    try:
        # Check if x11vnc process is running
        result = subprocess.run(["pgrep", "-f", f"x11vnc.*{vnc_port}"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"VNC server started successfully with PID: {result.stdout.strip()}")
            vnc_running = True
        else:
            print("VNC server process not found")
              # Check if port is listening on all interfaces
        result = subprocess.run(["netstat", "-ln"], capture_output=True, text=True)
        if f":{vnc_port}" in result.stdout:
            print(f"VNC port {vnc_port} is listening")
            vnc_running = True
            # Show what interface it's listening on
            for line in result.stdout.split('\n'):
                if f":{vnc_port}" in line:
                    print(f"  Listening: {line.strip()}")
        else:
            print(f"VNC port {vnc_port} is not listening")
            print("Available listening ports:")
            print(result.stdout)
            
        # Also test if we can connect to the port locally
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(2)
            test_result = test_sock.connect_ex(('127.0.0.1', vnc_port))
            test_sock.close()
            if test_result == 0:
                print(f"VNC port {vnc_port} is connectable locally")
                vnc_running = True
            else:
                print(f"VNC port {vnc_port} is not connectable locally (error: {test_result})")
        except Exception as e:
            print(f"Could not test VNC port connectivity: {e}")
            
    except Exception as e:
        print(f"Could not check VNC status: {e}")
    
    if not vnc_running:
        print("WARNING: VNC server may not be running properly")
        print("VNC output log contents:")
        try:
            with open("/data/vnc_output.log", "r") as f:
                print(f.read())
        except:
            print("Could not read VNC output log")

    return vnc_port

def start_xvfb():
    """Start virtual display and VNC server"""
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
    # Start Xvfb with RANDR extension enabled for mss compatibility
    cmd = ["Xvfb", display, "-screen", "0", "1920x1080x24", "-ac", 
           "+extension", "GLX", "+extension", "RANDR", "+extension", "RENDER",
           "-dpi", "96", "-noreset"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    # Verify Xvfb started
    if not os.path.exists(f"/tmp/.X{display_num}-lock"):
        print(f"Warning: Xvfb may not have started properly on display {display}")
    
    # Create Xauthority file
    subprocess.run(["touch", "/root/.Xauthority"], check=True)
    subprocess.run(["xauth", "add", display, ".", "1234567890abcdef"], check=True)
    
    # Start window manager for better VNC experience
    print("Starting window manager...")
    subprocess.Popen(["fluxbox"], env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # Start VNC server
    vnc_port = start_vnc_server(display, display_num)
    
    # Get container IP address for connection info
    container_ip = "unknown"
    try:
        # Try to get container IP from hostname command
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            container_ip = result.stdout.strip().split()[0]
        else:
            # Fallback method
            result = subprocess.run(["ip", "route", "get", "1"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'src' in line:
                        container_ip = line.split('src')[1].strip().split()[0]
                        break
    except Exception as e:
        print(f"Could not determine container IP: {e}")
    
    # Save VNC connection info for user
    instance_id = os.environ.get('INSTANCE_ID', get_unique_instance_id())
    with open(f"/data/vnc_info_{instance_id}.txt", "w") as f:
        f.write(f"=== VNC Connection Information ===\n")
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Display: {display}\n")
        f.write(f"VNC Port: {vnc_port}\n")
        f.write(f"Container IP: {container_ip}\n")
        f.write(f"\n=== Connection Instructions ===\n")
        f.write(f"From Host Machine:\n")
        f.write(f"  Use VNC viewer to connect to: localhost:{vnc_port}\n")
        f.write(f"  (Docker should forward port {vnc_port} to host)\n")
        f.write(f"\n")
        f.write(f"From Network:\n")
        f.write(f"  Use VNC viewer to connect to: {container_ip}:{vnc_port}\n")
        f.write(f"  (If container IP is accessible from your network)\n")
        f.write(f"\n")
        f.write(f"VNC Viewer Examples:\n")
        f.write(f"  - TigerVNC, RealVNC, TightVNC\n")
        f.write(f"  - Web browser: http://localhost:{vnc_port + 100} (if noVNC enabled)\n")
        f.write(f"\n")
        f.write(f"Troubleshooting:\n")
        f.write(f"  - Check /data/vnc_debug.log for VNC server logs\n")
        f.write(f"  - Check /data/vnc_output.log for detailed output\n")
        f.write(f"  - Verify port forwarding in docker-compose.yml\n")

    print(f"VNC connection info saved to /data/vnc_info_{instance_id}.txt")
    print(f"Container IP: {container_ip}")
    print(f"Internal VNC port: {vnc_port}")
    print(f"Use 'docker port <container_name> {vnc_port}' to find host port")
    return display_num
    
def start_assaultcube(display_num):
    """Start AssaultCube in background on specific empty map"""
    os.chdir('/opt/assaultcube')
    
    print(f"Starting AssaultCube on display :{display_num}")
    
    # Kill any existing AssaultCube processes
    subprocess.run(["pkill", "-f", "assaultcube"], check=False)
    subprocess.run(["pkill", "-f", "ac_client"], check=False)
    time.sleep(2)
    
    # Create all required config directories and files
    ac_home = "/root/.assaultcube/v1.3"
    config_dir = f"{ac_home}/config"
    private_dir = f"{ac_home}/private"
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)    # Create missing auth config file
    with open(f"{private_dir}/authprivate.cfg", "w") as f:
        f.write("// Auto-generated auth config - disable all authentication\n")
        f.write("authconnect 0\n")
        f.write("autoupdate 0\n")
        f.write("mastermask 0\n")
        f.write("autogetmap 0\n")
        f.write("allowmaster 0\n")
    
    # Create entropy file
    with open(f"{private_dir}/entropy.dat", "w") as f:
        f.write("entropy_data_placeholder\n")
    
    # Create init.cfg to run before anything else
    init_config = """
// Init config - runs first
authconnect 0
autoupdate 0
mastermask 0
autogetmap 0
allowmaster 0
showmenu 0
menuset 0
map ac_depot
"""
    
    with open(f"{config_dir}/init.cfg", "w") as f:
        f.write(init_config)
    
    # Create autoexec config
    config_content = """
// Auto-generated config - offline mode
authconnect 0
autoupdate 0
mastermask 0
autogetmap 0
masterconnect 0
allowmaster 0
showmenu 0
menuset 0
sound 0
map ac_depot
"""
    
    with open(f"{config_dir}/autoexec.cfg", "w") as f:
        f.write(config_content)
    
    # Create saved.cfg to override default settings
    saved_config = """
// Saved config - disable all authentication and menus
authconnect 0
autoupdate 0
mastermask 0
autogetmap 0
masterconnect 0
allowmaster 0
showmenu 0
menuset 0
sound 0
"""
    
    with open(f"{config_dir}/saved.cfg", "w") as f:
        f.write(saved_config)    
    # Start AssaultCube with proper environment
    env = os.environ.copy()
    env['LIBGL_ALWAYS_SOFTWARE'] = '0'
    env['NVIDIA_VISIBLE_DEVICES'] = 'all'
    env['NVIDIA_DRIVER_CAPABILITIES'] = 'all'
    
    cmd = ["./assaultcube.sh", "--home=/root/.assaultcube/v1.3", "--init"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(15)
    
    print("AssaultCube started, aggressively dismissing all prompts...")
    
    # Much more aggressive menu dismissal
    for attempt in range(10):
        print(f"Dismissal attempt {attempt + 1}/10...")
        try:
            # Press Escape multiple times rapidly
            for _ in range(10):
                subprocess.run(["xdotool", "key", "Escape"], check=False)
                time.sleep(0.1)
            
            # Press Enter to dismiss any auth dialogs
            for _ in range(5):
                subprocess.run(["xdotool", "key", "Return"], check=False)
                time.sleep(0.1)
            
            # Press Space to dismiss any other prompts
            for _ in range(3):
                subprocess.run(["xdotool", "key", "space"], check=False)
                time.sleep(0.1)
            
            # Try clicking outside any dialogs
            subprocess.run(["xdotool", "mousemove", "100", "100"], check=False)
            subprocess.run(["xdotool", "click", "1"], check=False)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Warning: Could not dismiss menus in attempt {attempt + 1}: {e}")
        
        time.sleep(1)
    
    print("Menu dismissal complete - game should be ready")

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
    
    # Keep container running for VNC access
    if bot_mode == 'vnc':
        print("VNC mode: Container will stay running for manual access")
        print("Check /data/vnc_info_*.txt for connection details")
        try:
            while True:
                time.sleep(60)
                print(f"VNC server still running... Check logs at /data/vnc_output.log")
        except KeyboardInterrupt:
            print("Container shutting down...")
    elif bot_mode == 'collect':
        print("Data collection mode: Starting ML data collection")
        run_data_collection()
    else:
        print(f"Unknown mode: {bot_mode}, running data collection anyway")
        run_data_collection()
