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

def auto_exec_config():
    subprocess.Popen([
        "/opt/assaultcube/assaultcube.sh",
        "-c", "/opt/assaultcube/config/autoexec.cfg",
    ])

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
    
    # Try to determine external Docker port mapping
    external_port = "unknown"
    try:
        # Get container ID from hostname or environment
        container_id = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
        
        # Try to find external port mapping using docker port command from inside container
        # This won't work from inside container, so we'll use a different approach
        
        # Alternative: Check if we can determine from environment or Docker API
        # For now, we'll just save the internal port and container info
        pass
    except:
        pass
    
    with open(f"/data/vnc_info_{instance_id}.txt", "w") as f:
        f.write(f"=== VNC Connection Information ===\n")
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Container ID: {subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()}\n")
        f.write(f"Display: {display}\n")
        f.write(f"Internal VNC Port: {vnc_port}\n")
        f.write(f"Container IP: {container_ip}\n")
        f.write(f"\n=== Connection Instructions ===\n")
        f.write(f"AUTOMATED EXTERNAL PORT DISCOVERY:\n")
        f.write(f"Run this command on your host to get the external port:\n")
        f.write(f"  docker port $(docker ps -q --filter ancestor=ai_training-bot) {vnc_port}\n")
        f.write(f"\n")
        f.write(f"OR use this one-liner to connect directly:\n")
        f.write(f"  EXTERNAL_PORT=$(docker port $(docker ps -q --filter ancestor=ai_training-bot) {vnc_port} | cut -d: -f2) && echo \"Connect to: localhost:$EXTERNAL_PORT\"\n")
        f.write(f"\n")
        f.write(f"Manual Connection:\n")
        f.write(f"  1. Find external port: docker port <container_name> {vnc_port}\n")
        f.write(f"  2. Connect VNC viewer to: localhost:<external_port>\n")
        f.write(f"\n")
        f.write(f"Direct Container Access (if networking allows):\n")
        f.write(f"  VNC to: {container_ip}:{vnc_port}\n")
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
    os.makedirs(private_dir, exist_ok=True)
    
    # Create missing auth config file
    with open(f"{private_dir}/authprivate.cfg", "w") as f:
        f.write("// Auto-generated auth config - minimal\n")
    
    # Create entropy file
    with open(f"{private_dir}/entropy.dat", "w") as f:
        f.write("entropy_data_placeholder\n")
      # Create init.cfg to run before anything else
    init_config = """// Init config - runs first
showmenu 0
map ac_depot
"""
    
    with open(f"{config_dir}/init.cfg", "w") as f:
        f.write(init_config)
    
    # Skip autoexec.cfg creation - using mounted version from host
    
    # Create saved.cfg to override default settings
    saved_config = """// Saved config - disable all authentication and menus
showmenu 0
sound 0
"""
    
    with open(f"{config_dir}/saved.cfg", "w") as f:
        f.write(saved_config)
    
    # Create server init commands for when we start a map
    with open(f"{config_dir}/serverinit.cfg", "w") as f:
        f.write("// Server initialization\n")
        f.write("sv_pure 0\n")
        f.write("servermotd \"\"\n")
        f.write("servertag \"\"\n")
    
    print("AssaultCube configuration files created")
    
    # Start AssaultCube with proper environment and capture output for debugging
    env = os.environ.copy()
    env['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering to avoid GPU issues
    env['NVIDIA_VISIBLE_DEVICES'] = 'all'
    env['NVIDIA_DRIVER_CAPABILITIES'] = 'all'
      # Create log files for debugging
    stdout_log = open("/data/assaultcube_stdout.log", "w")
    stderr_log = open("/data/assaultcube_stderr.log", "w")
    
    print("Starting AssaultCube with debugging enabled...")
    
    # Use the shell script with mounted autoexec config
    cmd = ["/opt/assaultcube/assaultcube.sh", "-c", "/opt/assaultcube/config/autoexec.cfg"]
    print(f"Command: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, env=env, stdout=stdout_log, stderr=stderr_log)
    
    # Store the process globally so we can monitor it
    global assaultcube_process
    assaultcube_process = proc
    
    print(f"AssaultCube process started with PID: {proc.pid}")
    time.sleep(5)  # Give it time to start up
    
    # Check if the process is still running
    if proc.poll() is None:
        print("AssaultCube process is still running")
    else:
        print(f"AssaultCube process has exited with code: {proc.returncode}")
        # Read the logs to see what went wrong
        stdout_log.close()
        stderr_log.close()
        
        print("=== STDOUT LOG ===")
        try:
            with open("/data/assaultcube_stdout.log", "r") as f:
                stdout_content = f.read()
                print(stdout_content if stdout_content else "No stdout output")
        except Exception as e:
            print(f"Could not read stdout log: {e}")
        
        print("=== STDERR LOG ===")
        try:
            with open("/data/assaultcube_stderr.log", "r") as f:
                stderr_content = f.read()
                print(stderr_content if stderr_content else "No stderr output")
        except Exception as e:
            print(f"Could not read stderr log: {e}")        # Try with a different approach - use binary with minimal args
        print("Attempting to restart with minimal arguments...")
        stdout_log = open("/data/assaultcube_stdout2.log", "w")
        stderr_log = open("/data/assaultcube_stderr2.log", "w")
        
        # Try with just basic home directory argument that AC expects
        cmd = ["./bin_unix/linux_64_client", "-h/root/.assaultcube/v1.3"]
        proc = subprocess.Popen(cmd, env=env, stdout=stdout_log, stderr=stderr_log)
        assaultcube_process = proc
        print(f"Restarted AssaultCube with minimal args, PID: {proc.pid}")
        time.sleep(10)
        
        if proc.poll() is None:
            print("AssaultCube restart with minimal args successful")
        else:
            print(f"AssaultCube restart with minimal args failed with code: {proc.returncode}")
            
            # Show the second set of logs
            stdout_log.close()
            stderr_log.close()
            
            print("=== MINIMAL ARGS STDOUT ===")
            try:
                with open("/data/assaultcube_stdout2.log", "r") as f:
                    content = f.read()
                    print(content if content else "No stdout output")
            except:
                pass
            
            print("=== MINIMAL ARGS STDERR ===")
            try:
                with open("/data/assaultcube_stderr2.log", "r") as f:
                    content = f.read()
                    print(content if content else "No stderr output")
            except:
                pass
    
    print("AssaultCube startup sequence complete, checking process status...")
    
    # More comprehensive process check
    try:
        result = subprocess.run(["pgrep", "-f", "ac_client"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"AssaultCube client process found: {result.stdout.strip()}")
            
            # Try gentle menu dismissal only if process is running
            print("AssaultCube process found, attempting gentle menu dismissal...")
            
            # Wait a bit more for full startup
            time.sleep(5)
            
            # Much gentler approach - just try to dismiss auth dialog
            for attempt in range(3):
                print(f"Gentle dismissal attempt {attempt + 1}/3...")
                try:
                    # Try just pressing Enter to dismiss auth dialog
                    subprocess.run(["xdotool", "key", "Return"], check=False)
                    time.sleep(1)
                    
                    # Single Escape to close any initial menus
                    subprocess.run(["xdotool", "key", "Escape"], check=False)
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Warning: Could not dismiss menus in attempt {attempt + 1}: {e}")
                
                time.sleep(2)
            
            print("Gentle menu dismissal complete")
        else:
            print("WARNING: AssaultCube client process not found after startup")
            result = subprocess.run(["pgrep", "-f", "assaultcube"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"But found general assaultcube process: {result.stdout.strip()}")
            else:
                print("No AssaultCube processes found at all")
    except Exception as e:
        print(f"Could not check AssaultCube process: {e}")
        print("Skipping menu dismissal")
    
    # Don't close the log files yet - keep them open for monitoring
    return proc

def monitor_assaultcube_process(proc):
    """Monitor AssaultCube process and restart if it dies"""
    global assaultcube_process
    
    while True:
        if proc is None or proc.poll() is not None:
            print("AssaultCube process has died, attempting restart...")
            
            # Read logs before restart
            print("=== Last STDOUT ===")
            try:
                with open("/data/assaultcube_stdout.log", "r") as f:
                    content = f.read()
                    if content:
                        print(content[-1000:])  # Last 1000 chars
                    else:
                        print("No stdout output")
            except:
                pass
            
            print("=== Last STDERR ===")
            try:
                with open("/data/assaultcube_stderr.log", "r") as f:
                    content = f.read()
                    if content:
                        print(content[-1000:])  # Last 1000 chars
                    else:
                        print("No stderr output")
            except:
                pass
            
            # Restart AssaultCube
            try:
                display_num = int(os.environ.get('DISPLAY', ':100').replace(':', ''))
                proc = start_assaultcube(display_num)
                assaultcube_process = proc
            except Exception as e:
                print(f"Failed to restart AssaultCube: {e}")
                time.sleep(30)  # Wait before next attempt
                continue
        
        time.sleep(30)  # Check every 30 seconds

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
    
    # Global variable to track AssaultCube process
    assaultcube_process = None
    
    display_num = start_xvfb()
    assaultcube_process = start_assaultcube(display_num)
    
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
        print("Starting AssaultCube process monitor...")
        
        # Start process monitoring in a separate thread
        import threading
        monitor_thread = threading.Thread(target=monitor_assaultcube_process, args=(assaultcube_process,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for AssaultCube to stabilize, then start ML training
        print("Waiting for AssaultCube to stabilize before starting ML training...")
        time.sleep(5)
        
        if assaultcube_process and assaultcube_process.poll() is None:
            print("AssaultCube appears stable, starting ML data collection/training")
            run_data_collection()
        else:
            print("AssaultCube is not running, cannot start ML training")
            print("Keeping container alive for debugging...")
            try:
                while True:
                    time.sleep(60)
                    if assaultcube_process and assaultcube_process.poll() is None:
                        status = "RUNNING"
                    else:
                        status = "STOPPED"
                    print(f"VNC server status: RUNNING, AssaultCube status: {status}")
                    print("Check logs at /data/assaultcube_*.log and /data/vnc_output.log")
            except KeyboardInterrupt:
                print("Container shutting down...")
    elif bot_mode == 'collect':
        print("Data collection mode: Starting ML data collection")
        
        # Start process monitoring in a separate thread
        import threading
        monitor_thread = threading.Thread(target=monitor_assaultcube_process, args=(assaultcube_process,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait a bit to ensure AssaultCube is stable before starting data collection
        print("Waiting for AssaultCube to stabilize...")
        time.sleep(30)
        
        if assaultcube_process and assaultcube_process.poll() is None:
            print("AssaultCube appears stable, starting data collection")
            run_data_collection()
        else:
            print("AssaultCube is not running, cannot start data collection")
            print("Keeping container alive for debugging...")
            try:
                while True:
                    time.sleep(60)
                    print("Container still running for debugging...")
            except KeyboardInterrupt:
                print("Container shutting down...")
    else:
        print(f"Unknown mode: {bot_mode}, running data collection anyway")
        
        # Start process monitoring in a separate thread
        import threading
        monitor_thread = threading.Thread(target=monitor_assaultcube_process, args=(assaultcube_process,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        time.sleep(30)
        run_data_collection()
