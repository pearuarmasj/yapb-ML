import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import pydirectinput
import time
import mss
import cv2
import ctypes
from collections import deque
import win32gui
import win32api
import win32con
import win32process
import psutil
import pickle
import os
import pyautogui
import tkinter as tk
from tkinter import messagebox
import sys

# Disable pydirectinput failsafe for subprocess mode
pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.01

def move_mouse_relative(x, y):
    try:
        # Try pydirectinput first
        pydirectinput.moveRel(x, y, relative=True)
    except:
        # Fallback to win32api
        current_pos = win32api.GetCursorPos()
        win32api.SetCursorPos((current_pos[0] + x, current_pos[1] + y))

def grab_screen(region=None):
    if region:
        img = pyautogui.screenshot(region=(region["left"], region["top"], region["width"], region["height"]))
        return np.array(img)
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
    """Find AssaultCube window handle"""
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if "AssaultCube" in window_title or "AC_" in window_title or "cube" in window_title.lower():
                windows.append((hwnd, window_title))
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    return windows

class CS16Env:
    def __init__(self, region=None, target_window=None):
        self.region = region
        self.target_window = target_window
        self.target_hwnd = None
          # Find target window
        game_windows = find_game_window()
        print(f"Found {len(game_windows)} game windows: {game_windows}")
        
        if target_window:
            for hwnd, title in game_windows:
                if target_window in title:
                    self.target_hwnd = hwnd
                    print(f"Found target window: {title} (HWND: {hwnd})")
                    break
        
        if not self.target_hwnd and game_windows:
            # Use first available game window
            self.target_hwnd = game_windows[0][0]
            print(f"Using first available game window: {game_windows[0][1]}")
        
        if not self.target_hwnd:
            print("Warning: No game window found. Input may not work.")
        
        self.actions = [
            ('w',), ('a',), ('s',), ('d',),
            ('w','mouse_left'), ('w','mouse_right'),
            ('a','mouse_left'), ('a','mouse_right'),
            ('s','mouse_left'), ('s','mouse_right'),            ('d','mouse_left'), ('d','mouse_right')
        ]
        self.held_keys = set()
        
    def reset(self):
        for key in list(self.held_keys):
            try:
                pydirectinput.keyUp(key)
            except Exception as e:
                print(f"Failed to release {key}: {e}")
        self.held_keys.clear()
        
    def step(self, action):
        used_wasd = False
        used_mouse = False
        
        # Parse action
        current_wasd = None
        mouse = None
        
        if isinstance(action, int):
            # Action index from action space
            if 0 <= action < len(self.actions):
                action_tuple = self.actions[action]
                if len(action_tuple) == 1:
                    current_wasd = action_tuple[0]
                elif len(action_tuple) == 2:
                    current_wasd, mouse = action_tuple
        elif isinstance(action, (tuple, list)):
            if len(action) == 2:
                current_wasd, mouse = action
            elif len(action) == 1:
                current_wasd = action[0]        # Release keys that are no longer needed
        for held_key in list(self.held_keys):
            if held_key != current_wasd:
                try:
                    pydirectinput.keyUp(held_key)
                except Exception as e:
                    print(f"Failed to release {held_key}: {e}")
                self.held_keys.remove(held_key)
        
        # Press new key if needed
        if current_wasd and current_wasd not in self.held_keys:
            try:
                pydirectinput.keyDown(current_wasd)
                self.held_keys.add(current_wasd)
                used_wasd = True
            except Exception as e:
                print(f"Failed to press {current_wasd}: {e}")
        elif current_wasd:
            used_wasd = True
            
        # Mouse movement
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
    
    # Train PPO agent
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Test the trained agent
    obs, _ = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

# More complex environment - Atari
def atari_experiment():
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    
    # Use CNN policy for image inputs
    model = DQN('CnnPolicy', env, verbose=1, buffer_size=100000)
    model.learn(total_timesteps=50000)
    
    # Evaluate performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()

# Gym wrapper for AssaultCube
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
          # Calculate reward based on frame difference
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

# Data collection wrapper for AssaultCube
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
            
        # Store data
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
            
        # Save data every 1000 steps
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

# Data collection mode
def collect_assaultcube_data():
    print("Setting up AssaultCube data collection...")
    print("Make sure AssaultCube is running and visible!")
    
    region = get_capture_region()
    collector = AssaultCubeDataCollector(region=region)
    
    print("Starting data collection...")
    print("Bot will take random actions and save training data.")
    
    obs = collector.reset()
    total_samples = 10000
    
    for step in range(total_samples):
        action = np.random.randint(0, 12)  # Random action
        obs, reward, done, info = collector.step(action)
        
        if done:
            obs = collector.reset()
            
        if step % 1000 == 0:
            print(f"Progress: {step}/{total_samples} samples collected")
    
    collector.save_data()  # Save any remaining data
    print(f"Data collection complete! Collected {total_samples} samples.")

# Train from collected data (use separate offline_trainer.py script)
def train_from_data():
    print("Use the separate offline_trainer.py script for training from collected data.")
    print("Run: python offline_trainer.py")
def train_assaultcube_ppo():
    print("Setting up AssaultCube environment...")
    print("Make sure AssaultCube is running and visible!")
    
    region = get_capture_region()
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
    
    # Add custom callback for better logging
    class LoggingCallback:
        def __init__(self):
            self.episode_rewards = []
            self.episode_lengths = []
            self.current_reward = 0
            self.current_length = 0
            
        def on_step(self):
            return True
    
    # Train with progress updates
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

# Basic Gym environment example

# Experiment with different algorithms
def algorithm_comparison():
    env = gym.make('LunarLander-v2')
    
    # Train different algorithms
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
      # Plot comparison
    plt.bar(list(results.keys()), list(results.values()))
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Mean Reward')
    plt.show()
    
    env.close()

# Hyperparameter tuning example
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

# Test trained model
def test_trained_model():
    print("Running trained model test...")
    import subprocess
    subprocess.run(["python", "test_trained_model.py"])

class RegionSelector:
    def __init__(self):
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.rect = None
        self.root = None
        self.canvas = None
        self.selected_region = None
        
    def select_region(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
        self.root.bind('<Escape>', lambda e: self.cancel_selection())
        
        self.canvas.create_text(
            self.root.winfo_screenwidth() // 2, 50,
            text="Drag to select region. Press ESC to cancel.",
            fill='white', font=('Arial', 16)
        )
        
        self.root.mainloop()
        return self.selected_region
        
    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def on_drag(self, event):
        if self.rect and self.canvas:
            self.canvas.delete(self.rect)
        if self.canvas:
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='red', width=2
            )
        
    def on_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        
        left = min(self.start_x, self.end_x)
        top = min(self.start_y, self.end_y)
        width = abs(self.end_x - self.start_x)
        height = abs(self.end_y - self.start_y)
        
        if width > 10 and height > 10:
            self.selected_region = {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            }
            if self.root:
                self.root.quit()
                self.root.destroy()
        else:
            self.cancel_selection()
            
    def cancel_selection(self):
        self.selected_region = None
        if self.root:
            self.root.quit()
            self.root.destroy()

def get_capture_region():
    print("Region Selection Tool")
    print("1. Use full screen (2560x1440)")
    print("2. Select custom region")
    
    choice = input("Choose option (1-2): ")
    
    if choice == "1":
        return {"top": 0, "left": 0, "width": 2560, "height": 1440}
    elif choice == "2":
        print("Close this window and drag to select your capture region...")
        print("Press ESC to cancel selection")
        
        selector = RegionSelector()
        region = selector.select_region()
        
        if region:
            print(f"Selected region: {region}")
            return region
        else:
            print("Selection cancelled, using full screen")
            return {"top": 0, "left": 0, "width": 2560, "height": 1440}
    else:
        print("Invalid choice, using full screen")
        return {"top": 0, "left": 0, "width": 2560, "height": 1440}

# Check for instance mode
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

if __name__ == "__main__":
    # Check for instance mode
    if "--instance-mode" in sys.argv:
        config = get_instance_config()
        print(f"Running in instance mode - ID: {config['instance_id']}")
        if "--data-collection" in sys.argv:
            print("Data collection instance starting...")
            if config['region']:
                print(f"Using predefined region: {config['region']}")
                region = config['region']
            else:
                print("No predefined region - launching region selector...")
                region = get_capture_region()
                
            print(f"Region selected: {region}")
            collector = AssaultCubeDataCollector(region=region, data_dir=config['data_dir'])
            
            print("Starting data collection...")
            obs = collector.reset()
            total_samples = 10000
            
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
            region = config['region'] or get_capture_region()
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
    print("1. Basic CartPole")
    print("2. Atari Breakout") 
    print("3. Algorithm Comparison")
    print("4. Hyperparameter Tuning")
    print("5. Train AssaultCube with PPO (Real-time)")
    print("6. Collect AssaultCube Data (Offline)")
    print("7. Train from Collected Data")
    print("8. Test Trained Model")
    print("9. Launch Multi-Instance Manager")
    
    choice = input("Enter choice (1-9): ")
    
    if choice == "1":
        basic_cartpole()
    elif choice == "2":
        atari_experiment()
    elif choice == "3":
        algorithm_comparison()
    elif choice == "4":
        hyperparameter_tuning()
    elif choice == "5":
        train_assaultcube_ppo()
    elif choice == "6":
        collect_assaultcube_data()
    elif choice == "7":
        train_from_data()
    elif choice == "8":
        test_trained_model()
    elif choice == "9":
        import subprocess
        subprocess.run(["python", "multi_instance_launcher.py"])
    else:
        print("Invalid choice")

def send_key_to_window(hwnd, key, press=True):
    """Send key directly to window without focus"""
    if key == 'w':
        vk_code = 0x57
    elif key == 'a':
        vk_code = 0x41
    elif key == 's':
        vk_code = 0x53
    elif key == 'd':
        vk_code = 0x44
    else:
        return False
    
    try:
        if press:
            win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
        else:
            win32gui.SendMessage(hwnd, win32con.WM_KEYUP, vk_code, 0)
        return True
    except Exception as e:
        print(f"Failed to send key {key} to window: {e}")
        return False
