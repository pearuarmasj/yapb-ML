import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import mss
import cv2
import ctypes
from collections import deque
import pickle
import os
import sys
import subprocess
import json
import socket
from datetime import datetime
import threading
import time

# Disable pyautogui display requirements
if os.environ.get('PYAUTOGUI_DISABLE_DISPLAY'):
    pass

# Global error counters for input debugging
_mouse_error_count = 0
_key_error_count = 0

def get_assaultcube_window():
    """Get the AssaultCube window ID"""
    try:
        result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--class', 'AssaultCube'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except Exception as e:
        print(f"Failed to find AssaultCube window: {e}")
    return None

def move_mouse_relative(x, y):
    """Move mouse relative to current position, targeting AssaultCube window"""
    global _mouse_error_count
    try:
        window_id = get_assaultcube_window()
        if window_id:
            # Focus the window first
            subprocess.run(['xdotool', 'windowfocus', window_id], check=False)
            # Move mouse relative
            if x < 0 or y < 0:
                subprocess.run(['xdotool', 'mousemove_relative', '--', str(x), str(y)], check=True)
            else:
                subprocess.run(['xdotool', 'mousemove_relative', str(x), str(y)], check=True)
        else:
            # Fallback to global mouse movement
            if x < 0 or y < 0:
                subprocess.run(['xdotool', 'mousemove_relative', '--', str(x), str(y)], check=True)
            else:
                subprocess.run(['xdotool', 'mousemove_relative', str(x), str(y)], check=True)
    except Exception as e:
        _mouse_error_count += 1
        if _mouse_error_count % 50 == 1:  # Print error every 50 failures
            print(f"Mouse movement error: {e}")

def send_key_to_window(key, action='key'):
    """Send keyboard input to AssaultCube window"""
    global _key_error_count
    try:
        window_id = get_assaultcube_window()
        if window_id:
            # Focus the window first
            subprocess.run(['xdotool', 'windowfocus', window_id], check=False)
            # Send key to specific window
            subprocess.run(['xdotool', action, '--window', window_id, key], check=True)
        else:
            # Fallback to global key sending
            subprocess.run(['xdotool', action, key], check=True)
    except Exception as e:
        _key_error_count += 1
        if _key_error_count % 50 == 1:  # Print error every 50 failures
            print(f"Key {action} error for '{key}': {e}")

def send_text_to_window(text):
    """Send text input to AssaultCube window"""
    global _key_error_count
    try:
        window_id = get_assaultcube_window()
        if window_id:
            # Focus the window first
            subprocess.run(['xdotool', 'windowfocus', window_id], check=False)
            # Send text to specific window
            subprocess.run(['xdotool', 'type', '--window', window_id, text], check=True)
        else:
            # Fallback to global text sending
            subprocess.run(['xdotool', 'type', text], check=True)
    except Exception as e:
        _key_error_count += 1
        if _key_error_count % 50 == 1:  # Print error every 50 failures
            print(f"Text input error for '{text}': {e}")

def grab_screen(region=None):
    if region:
        with mss.mss() as sct:
            monitor = {"top": region["top"], "left": region["left"], 
                      "width": region["width"], "height": region["height"]}
            img = np.array(sct.grab(monitor))
            return img[..., :3]
    else:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = np.array(sct.grab(monitor))
            return img[..., :3]

def preprocess(frame, size=(84, 84)):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    return frame.astype(np.uint8)

def find_game_window():
    """Find actual AssaultCube window using xdotool"""
    try:
        patterns = [
            ("--name", "AssaultCube"),
            ("--name", "ac_client"), 
            ("--name", "native_client"),
            ("--class", "AssaultCube"),
            ("--class", "ac_client")
        ]
        
        for search_type, pattern in patterns:
            result = subprocess.run(["xdotool", "search", "--onlyvisible", search_type, pattern], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip().split('\n')[0]
                # Get window name
                name_result = subprocess.run(["xdotool", "getwindowname", window_id], 
                                           capture_output=True, text=True, timeout=2)
                window_name = name_result.stdout.strip() if name_result.returncode == 0 else "AssaultCube"
                return [(window_id, window_name)]
        
        # If no specific AssaultCube window found, return empty list
        return []
        
    except Exception as e:
        print(f"Error finding game window: {e}")
        return []

class CS16Env:
    def __init__(self, region=None, target_window=None, instance_id=None):
        self.region = region
        self.target_window = target_window
        self.target_hwnd = None
        self.instance_id = instance_id or os.environ.get('INSTANCE_ID', 'default')
        
        game_windows = find_game_window()
        print(f"Found {len(game_windows)} game windows: {game_windows}")
        
        if target_window:
            for hwnd, title in game_windows:
                if target_window in title:
                    self.target_hwnd = hwnd
                    print(f"Found target window: {title} (HWND: {hwnd})")
                    break
        
        if not self.target_hwnd and game_windows:
            self.target_hwnd = game_windows[0][0]
            print(f"Using first available game window: {game_windows[0][1]}")
        
        if not self.target_hwnd:
            print("Warning: No game window found. Input may not work.")
        
        self.actions = [
            ('w',), ('a',), ('s',), ('d',),
            ('w','mouse_left'), ('w','mouse_right'),
            ('a','mouse_left'), ('a','mouse_right'),
            ('s','mouse_left'), ('s','mouse_right'),
            ('d','mouse_left'), ('d','mouse_right')
        ]
        self.held_keys = set()
        
    def reset(self):
        for key in list(self.held_keys):
            try:
                send_key_to_window(key, 'keyup')
            except Exception as e:
                print(f"Failed to release {key}: {e}")
        self.held_keys.clear()
        
    def step(self, action):
        used_wasd = False
        used_mouse = False
        
        current_wasd = None
        mouse = None
        
        if isinstance(action, int):
            if 0 <= action < len(self.actions):
                action_tuple = self.actions[action]
                if len(action_tuple) == 1:
                    current_wasd = action_tuple[0]
                elif len(action_tuple) == 2:
                    current_wasd, mouse = action_tuple
        elif isinstance(action, (tuple, list)):
            if len(action) == 2:
                current_wasd, mouse = action
            elif len(action) == 1:                current_wasd = action[0]        
        # Debug print every 100 steps
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 1
        
        for held_key in list(self.held_keys):
            if held_key != current_wasd:
                try:
                    send_key_to_window(held_key, 'keyup')
                except Exception as e:
                    print(f"Failed to release {held_key}: {e}")
                self.held_keys.remove(held_key)
        
        if current_wasd and current_wasd not in self.held_keys:
            try:
                send_key_to_window(current_wasd, 'keydown')
                self.held_keys.add(current_wasd)
                used_wasd = True
            except Exception as e:
                print(f"Failed to press {current_wasd}: {e}")
        elif current_wasd:
            used_wasd = True
            
        if mouse == 'mouse_left':
            try:
                move_mouse_relative(-50, 0)
                used_mouse = True
            except Exception as e:
                print(f"Failed mouse left: {e}")
        elif mouse == 'mouse_right':
            try:
                move_mouse_relative(50, 0)
                used_mouse = True
            except Exception as e:
                print(f"Failed mouse right: {e}")
            
        obs = grab_screen(self.region)
        reward = np.random.random()
        done = False
        return obs, reward, done, {'used_wasd': used_wasd, 'used_mouse': used_mouse}
        
    def get_observation(self):
        return grab_screen(self.region)

def basic_cartpole():
    env = gym.make('CartPole-v1', render_mode='human')
    
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    
    obs, _ = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

def atari_experiment():
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    
    model = DQN('CnnPolicy', env, verbose=1, buffer_size=100000)
    model.learn(total_timesteps=50000)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()

class AssaultCubeGymWrapper(gym.Env):
    def __init__(self, region=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.game_env = CS16Env(region=region)
        self.frame_stack = deque(maxlen=4)
        self.prev_frame = None
        self.step_count = 0
        self.episode_reward = 0
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game_env.reset()
        
        obs = self.game_env.get_observation()
        processed_obs = preprocess(obs)
        
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(processed_obs)
        self.prev_frame = processed_obs
        self.step_count = 0
        self.episode_reward = 0
        stacked_obs = np.stack(self.frame_stack, axis=0)
        return stacked_obs, {}
    
    def step(self, action):
        game_action = self.game_env.actions[action]
        obs, _, done, info = self.game_env.step(game_action)
        
        processed_obs = preprocess(obs)
        self.frame_stack.append(processed_obs)
        stacked_obs = np.stack(self.frame_stack, axis=0)
        
        if self.prev_frame is not None:
            diff = np.abs(processed_obs.astype(np.float32) - self.prev_frame.astype(np.float32))
            reward = np.mean(diff) / 255.0
        else:
            reward = 0.0
        self.prev_frame = processed_obs
        self.step_count += 1
        self.episode_reward += reward
        
        if self.step_count % 100 == 0:
            print(f"Step {self.step_count} | Reward: {reward:.4f} | Episode Total: {self.episode_reward:.4f}")
        
        return stacked_obs, reward, done, False, info

class AssaultCubeDataCollector:
    def __init__(self, region=None, data_dir="training_data"):
        self.game_env = CS16Env(region=region)
        self.frame_stack = deque(maxlen=4)
        self.prev_frame = None
        self.step_count = 0
        self.data_dir = data_dir
        self.data_buffer = []
        
        os.makedirs(data_dir, exist_ok=True)
        
    def reset(self):
        self.game_env.reset()
        obs = self.game_env.get_observation()
        processed_obs = preprocess(obs)
        
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(processed_obs)
        self.prev_frame = processed_obs
        self.step_count = 0
        
        return np.stack(self.frame_stack, axis=0)
    
    def step(self, action):
        game_action = self.game_env.actions[action]
        obs, _, done, info = self.game_env.step(game_action)
        
        processed_obs = preprocess(obs)
        self.frame_stack.append(processed_obs)
        stacked_obs = np.stack(self.frame_stack, axis=0)
        
        if self.prev_frame is not None:
            diff = np.abs(processed_obs.astype(np.float32) - self.prev_frame.astype(np.float32))
            reward = np.mean(diff) / 255.0
        else:
            reward = 0.0
            
        self.data_buffer.append({
            'state': self.frame_stack.copy(),
            'action': action,
            'reward': reward,
            'next_state': stacked_obs.copy(),
            'done': done
        })
        
        self.prev_frame = processed_obs
        self.step_count += 1
        
        if self.step_count % 100 == 0:
            print(f"Collected {self.step_count} samples | Reward: {reward:.4f}")
            
        if len(self.data_buffer) >= 1000:
            self.save_data()
            
        return stacked_obs, reward, done, info
    
    def save_data(self):
        if self.data_buffer:
            timestamp = int(time.time())
            filename = f"{self.data_dir}/data_{timestamp}_{len(self.data_buffer)}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(self.data_buffer, f)
            print(f"Saved {len(self.data_buffer)} samples to {filename}")
            self.data_buffer = []

def collect_assaultcube_data():
    print("Setting up AssaultCube data collection...")
    print("Make sure AssaultCube is running and visible!")
    
    region = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    collector = AssaultCubeDataCollector(region=region)
    
    print("Starting data collection...")
    print("Bot will take random actions and save training data.")
    print("Debug screenshots will be saved every 500 steps to /data/screenshots/")
    
    # Take initial screenshot
    save_debug_screenshot(region, "initial_state")
    
    obs = collector.reset()
    total_samples = 100000
    
    for step in range(total_samples):
        action = np.random.randint(0, 12)
        obs, reward, done, info = collector.step(action)
        
        if done:
            obs = collector.reset()
            
        # Save debug screenshots periodically
        if step % 500 == 0:
            print(f"Progress: {step}/{total_samples} samples collected")
            save_debug_screenshot(region, f"step_{step}")
        elif step % 100 == 0:
            print(f"Progress: {step}/{total_samples} samples collected")
    
    # Take final screenshot
    save_debug_screenshot(region, "final_state")
    
    collector.save_data()
    print(f"Data collection complete! Collected {total_samples} samples.")
    print("Check /data/screenshots/ for debug screenshots of bot activity.")

def train_from_data():
    print("Use the separate offline_trainer.py script for training from collected data.")
    print("Run: python offline_trainer.py")

def train_assaultcube_ppo():
    print("Setting up AssaultCube environment...")
    print("Make sure AssaultCube is running and visible!")
    
    region = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    env = AssaultCubeGymWrapper(region=region)
    print("Creating PPO model with CNN policy...")
    model = PPO(
        'CnnPolicy', 
        env, 
        verbose=2,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        device='cuda'
    )
    
    print("Starting training...")
    
    class LoggingCallback:
        def __init__(self):
            self.episode_rewards = []
            self.episode_lengths = []
            self.current_reward = 0
            self.current_length = 0
            
        def on_step(self):
            return True
    
    total_steps = 50000
    update_freq = 2048
    
    for i in range(0, total_steps, update_freq):
        remaining = min(update_freq, total_steps - i)
        print(f"Training steps {i} to {i + remaining} of {total_steps}")
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)
        print(f"Completed {i + remaining}/{total_steps} steps")
    
    print("Saving model...")
    model.save("assaultcube_ppo")
    
    print("Testing trained model...")
    obs, _ = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()

def algorithm_comparison():
    env = gym.make('LunarLander-v2')
    
    algorithms = {
        'PPO': PPO('MlpPolicy', env, verbose=0),
        'DQN': DQN('MlpPolicy', env, verbose=0)
    }
    
    results = {}
    for name, model in algorithms.items():
        print(f"Training {name}...")
        model.learn(total_timesteps=20000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        results[name] = mean_reward
        print(f"{name} mean reward: {mean_reward:.2f}")
    
    plt.bar(list(results.keys()), list(results.values()))
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Mean Reward')
    plt.show()
    
    env.close()

def hyperparameter_tuning():
    env = gym.make('CartPole-v1')
    
    learning_rates = [1e-4, 3e-4, 1e-3]
    results = []
    
    for lr in learning_rates:
        model = PPO('MlpPolicy', env, learning_rate=lr, verbose=0)
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        results.append(mean_reward)
        print(f"LR {lr}: {mean_reward:.2f}")
    
    plt.plot(learning_rates, results, 'o-')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Reward')
    plt.title('Learning Rate vs Performance')
    plt.xscale('log')
    plt.show()
    
    env.close()

def test_trained_model():
    print("Running trained model test...")
    import subprocess
    subprocess.run(["python", "test_trained_model.py"])

def get_instance_config():
    instance_id = os.environ.get('INSTANCE_ID')
    data_dir = os.environ.get('DATA_DIR', 'training_data')
    capture_region = os.environ.get('CAPTURE_REGION')
    
    region = None
    if capture_region:
        coords = capture_region.split(',')
        region = {
            'left': int(coords[0]),
            'top': int(coords[1]), 
            'width': int(coords[2]),
            'height': int(coords[3])
        }
    
    return {
        'instance_id': instance_id,
        'data_dir': data_dir,
        'region': region
    }

def save_debug_screenshot(region=None, filename_prefix="debug_screenshot"):
    """Save a screenshot for debugging purposes"""
    import datetime
    try:
        screenshot = grab_screen(region)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/data/screenshots/{filename_prefix}_{timestamp}.png"
        
        # Create screenshots directory if it doesn't exist
        os.makedirs("/data/screenshots", exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, screenshot_bgr)
        print(f"Debug screenshot saved: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to save debug screenshot: {e}")
        return None

def get_current_map():
    """Attempt to detect current map by taking a screenshot and analyzing it"""
    try:
        screenshot = grab_screen()
        # Simple hash of screenshot to identify map
        screenshot_hash = hash(screenshot.tobytes())
        return str(screenshot_hash)[:16]
    except:
        return "unknown_map"

def get_available_maps():
    """Get list of available AssaultCube maps"""
    return [
        'ac_desert', 'ac_depot', 'ac_aqueous', 'ac_arctic', 'ac_complex',
        'ac_gothic', 'ac_urban', 'ac_ingress', 'ac_mines', 'ac_outpost',
        'ac_power', 'ac_rattrap', 'ac_scaffold', 'ac_shine', 'ac_snow',
        'ac_sunset', 'ac_toxic', 'ac_werk'
    ]

if __name__ == "__main__":
    # Check for command line arguments first
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "collect":
            collect_assaultcube_data()
            sys.exit(0)
        elif mode == "train":
            train_from_data()
            sys.exit(0)
        elif mode == "both":
            print("Starting data collection and training...")
            import threading
            import time
            
            # Start data collection in background
            collection_thread = threading.Thread(target=collect_assaultcube_data)
            collection_thread.daemon = True
            collection_thread.start()
            
            # Wait a bit for initial data, then start training
            time.sleep(60)
            train_from_data()
            sys.exit(0)
    
    if "--instance-mode" in sys.argv:
        config = get_instance_config()
        print(f"Running in instance mode - ID: {config['instance_id']}")
        if "--data-collection" in sys.argv:
            print("Data collection instance starting...")
            if config['region']:
                print(f"Using predefined region: {config['region']}")
                region = config['region']
            else:
                print("No predefined region - using full screen")
                region = {"left": 0, "top": 0, "width": 1920, "height": 1080}
                print(f"Region selected: {region}")
            
            collector = AssaultCubeDataCollector(region=region, data_dir=config['data_dir'])
            
            # Map synchronization is now auto-detected by the CS16Env constructor
            print("Starting data collection...")
            obs = collector.reset()
            total_samples = 100000
            
            for step in range(total_samples):
                action = np.random.randint(0, 12)
                obs, reward, done, info = collector.step(action)
                
                if done:
                    obs = collector.reset()
                    
                if step % 1000 == 0:
                    print(f"Progress: {step}/{total_samples} samples collected")
            
            collector.save_data()
            print(f"Data collection complete! Collected {total_samples} samples.")

        elif "--training" in sys.argv:
            region = config['region'] or {"left": 0, "top": 0, "width": 1920, "height": 1080}
            env = AssaultCubeGymWrapper(region=region)
            
            model = PPO(
                'CnnPolicy', 
                env, 
                verbose=2,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                device='cuda'
            )
            
            total_steps = 50000
            update_freq = 2048
            
            for i in range(0, total_steps, update_freq):
                remaining = min(update_freq, total_steps - i)
                print(f"Training steps {i} to {i + remaining} of {total_steps}")
                model.learn(total_timesteps=remaining, reset_num_timesteps=False)
                print(f"Completed {i + remaining}/{total_steps} steps")
                model.save(f"assaultcube_ppo_instance_{config['instance_id']}")
            env.close()
        
        sys.exit(0)
    
    # Normal interactive mode
    print("Choose experiment:")
    print("1. Collect AssaultCube Data Only")
    print("2. Train from Collected Data") 
    print("3. Collect Data + Train Simultaneously")
    print("4. Test Trained Model")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        collect_assaultcube_data()
    elif choice == "2":
        train_from_data()
    elif choice == "3":
        print("Starting data collection and training...")
        
        # Start data collection in background
        collection_thread = threading.Thread(target=collect_assaultcube_data)
        collection_thread.daemon = True
        collection_thread.start()
        
        # Wait a bit for initial data, then start training
        time.sleep(60)
        train_from_data()
    elif choice == "4":
        test_trained_model()
    else:
        print("Invalid choice")
