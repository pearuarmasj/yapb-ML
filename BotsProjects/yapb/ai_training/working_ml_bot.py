#!/usr/bin/env python3
"""
🎯 CS 1.6 ML Bot Environment for de_survivor
Controls YOUR player to learn the map first!
"""

import os
import time
import numpy as np
import cv2
import win32gui
import win32ui
import win32con
import win32api
from PIL import Image
import ctypes
from ctypes import wintypes
import psutil
import pymem
from typing import Dict, Tuple, Optional, Any
import logging

from cs16_offsets import CS16OffsetManager
from game_controller import CS16GameController

class WorkingCS16Environment:
    """
    CS 1.6 Environment for controlling YOUR player on de_survivor
    This is the core class that will learn to play the map!
    """
    
    def __init__(self, image_size=(160, 120), debug=True):
        self.image_size = image_size
        self.debug = debug
        
        # Initialize components
        self.offset_manager = CS16OffsetManager()
        self.game_controller = CS16GameController()
        
        # Game state
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.last_health = 100.0
        self.last_angles = np.array([0.0, 0.0, 0.0])
        self.step_count = 0
        self.episode_start_time = time.time()
        
        # de_survivor specific landmarks for learning
        self.survivor_landmarks = {
            'spawn_area': np.array([0, 0, 0]),      # Will be updated dynamically
            'bridge_area': np.array([500, 200, 50]),
            'tower_area': np.array([-300, 400, 100]),
            'underground': np.array([200, -400, -50]),
            'rooftop': np.array([300, 300, 200])
        }
        
        # Action space: [forward/back, strafe_lr, turn_lr, look_ud, jump, duck, shoot]
        self.action_space_size = 7
        
        if self.debug:
            print("🎮 CS 1.6 ML Bot Environment Initialized")
            print(f"🗺️  Target Map: de_survivor")
            print(f"📏 Image Size: {image_size}")
            print("🎯 Ready to learn the map!")
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and return initial observation"""
        try:
            # Find and connect to CS 1.6
            if not self._connect_to_cs16():
                raise RuntimeError("Could not connect to CS 1.6! Make sure it's running.")
            
            # Get initial state
            obs = self._get_observation()
            info = {"step": 0, "map": "de_survivor", "mode": "learning"}
            
            self.step_count = 0
            self.episode_start_time = time.time()
            
            if self.debug:
                print("🔄 Environment Reset - Ready to learn!")
                print(f"📍 Starting Position: {obs['position']}")
                print(f"❤️  Health: {obs['health'][0]}")
            
            return obs, info
            
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            return self._get_default_observation(), {"error": str(e)}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute action and return new state"""
        try:
            # Execute the action in CS 1.6
            self._execute_action(action)
            
            # Wait a bit for action to take effect
            time.sleep(0.05)  # 50ms delay
            
            # Get new observation
            obs = self._get_observation()
            
            # Calculate reward
            reward = self._calculate_reward(obs, action)
            
            # Check if episode is done
            done = self._is_episode_done(obs)
            truncated = self.step_count > 1000  # Max steps per episode
            
            # Update counters
            self.step_count += 1
            
            # Info for debugging
            info = {
                "step": self.step_count,
                "reward_breakdown": self._get_reward_breakdown(obs, action),
                "position": obs['position'],
                "health": obs['health'][0]
            }
            
            if self.debug and self.step_count % 20 == 0:
                print(f"Step {self.step_count}: Pos={obs['position'][:2]}, R={reward:.2f}, H={obs['health'][0]}")
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            print(f"❌ Step failed: {e}")
            return self._get_default_observation(), -1.0, True, False, {"error": str(e)}
    
    def _connect_to_cs16(self) -> bool:
        """Find and connect to CS 1.6 process"""
        try:
            # Find CS 1.6 window
            hwnd = win32gui.FindWindow(None, "Counter-Strike")
            if not hwnd:
                hwnd = win32gui.FindWindow(None, "Counter-Strike 1.6")
            
            if hwnd:
                self.game_controller.set_window_handle(hwnd)
                if self.debug:
                    print("✅ Found CS 1.6 window")
                
                # Try to connect to process for memory reading
                if self.offset_manager.connect_to_process():
                    if self.debug:
                        print("✅ Connected to CS 1.6 process")
                    return True
                else:
                    print("⚠️  Window found but couldn't connect to process")
                    return True  # Still allow keyboard control
            
            print("❌ CS 1.6 not found! Make sure it's running.")
            return False
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current game state observation"""
        try:
            # Get player state from memory (if connected)
            player_state = self.offset_manager.get_player_state()
            
            if player_state:
                position = np.array([
                    player_state.get('x', 0.0),
                    player_state.get('y', 0.0), 
                    player_state.get('z', 0.0)
                ])
                health = np.array([player_state.get('health', 100.0)])
                angles = np.array([
                    player_state.get('pitch', 0.0),
                    player_state.get('yaw', 0.0),
                    player_state.get('roll', 0.0)
                ])
                velocity = np.array([
                    player_state.get('vel_x', 0.0),
                    player_state.get('vel_y', 0.0),
                    player_state.get('vel_z', 0.0)
                ])
            else:
                # Fallback to previous values if memory reading fails
                position = self.last_position.copy()
                health = np.array([100.0])
                angles = self.last_angles.copy()
                velocity = np.array([0.0, 0.0, 0.0])
            
            # Capture screen for visual input
            screen = self._capture_screen()
            
            # Calculate distance to de_survivor landmarks
            landmark_distances = self._calculate_landmark_distances(position)
            
            # Create observation dictionary
            obs = {
                'position': position,
                'health': health,
                'angles': angles,
                'velocity': velocity,
                'screen': screen,
                'landmark_distances': landmark_distances,
                'step_count': np.array([self.step_count])
            }
            
            # Update last known values
            self.last_position = position.copy()
            self.last_health = health[0]
            self.last_angles = angles.copy()
            
            return obs
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Observation error: {e}")
            return self._get_default_observation()
    
    def _capture_screen(self) -> np.ndarray:
        """Capture CS 1.6 screen for visual input"""
        try:
            # Get CS 1.6 window
            hwnd = win32gui.FindWindow(None, "Counter-Strike")
            if not hwnd:
                hwnd = win32gui.FindWindow(None, "Counter-Strike 1.6")
            
            if hwnd:
                # Get window dimensions
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                
                # Capture window
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                saveDC.SelectObject(saveBitMap)
                
                saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
                
                # Convert to numpy array
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                img = np.frombuffer(bmpstr, dtype='uint8')
                img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
                img = img[...,:3]  # Remove alpha channel
                img = np.ascontiguousarray(img)
                
                # Resize to target size
                img = cv2.resize(img, self.image_size)
                img = img.astype(np.float32) / 255.0
                
                # Cleanup
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                
                return img
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Screen capture error: {e}")
        
        # Return black screen as fallback
        return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def _execute_action(self, action: np.ndarray):
        """Execute action in CS 1.6"""
        try:
            # Action format: [forward/back, strafe_lr, turn_lr, look_ud, jump, duck, shoot]
            forward_back = float(action[0])
            strafe_lr = float(action[1]) 
            turn_lr = float(action[2])
            look_ud = float(action[3])
            jump = float(action[4]) > 0.5
            duck = float(action[5]) > 0.5
            shoot = float(action[6]) > 0.5
            
            # Movement keys
            if forward_back > 0.3:
                self.game_controller.send_key_down('forward')
            elif forward_back < -0.3:
                self.game_controller.send_key_down('backward')
            else:
                self.game_controller.send_key_up('forward')
                self.game_controller.send_key_up('backward')
            
            if strafe_lr > 0.3:
                self.game_controller.send_key_down('right')
            elif strafe_lr < -0.3:
                self.game_controller.send_key_down('left')
            else:
                self.game_controller.send_key_up('left')
                self.game_controller.send_key_up('right')
            
            # Mouse movement for turning
            if abs(turn_lr) > 0.1:
                mouse_x = int(turn_lr * 10)  # Scale mouse movement
                self.game_controller.move_mouse(mouse_x, 0)
            
            if abs(look_ud) > 0.1:
                mouse_y = int(look_ud * 10)
                self.game_controller.move_mouse(0, mouse_y)
            
            # Action keys
            if jump:
                self.game_controller.send_key_down('jump')
            else:
                self.game_controller.send_key_up('jump')
            
            if duck:
                self.game_controller.send_key_down('duck')
            else:
                self.game_controller.send_key_up('duck')
            
            if shoot:
                self.game_controller.send_key_down('attack1')
            else:
                self.game_controller.send_key_up('attack1')
                
        except Exception as e:
            if self.debug:
                print(f"⚠️  Action execution error: {e}")
    
    def _calculate_landmark_distances(self, position: np.ndarray) -> np.ndarray:
        """Calculate distances to de_survivor landmarks"""
        distances = []
        for landmark_name, landmark_pos in self.survivor_landmarks.items():
            dist = np.linalg.norm(position - landmark_pos)
            distances.append(dist)
        return np.array(distances)
    
    def _calculate_reward(self, obs: Dict[str, Any], action: np.ndarray) -> float:
        """Calculate reward for current state and action"""
        reward = 0.0
        
        # Survival reward (staying alive)
        if obs['health'][0] > 0:
            reward += 0.1
          # Movement reward (encourage exploration)
        position_change = np.linalg.norm(obs['position'] - self.last_position)
        if position_change > 1.0:  # Moved a decent distance
            reward += min(float(position_change * 0.01), 0.5)  # Cap movement reward
        
        # Health penalty
        health_change = obs['health'][0] - self.last_health
        if health_change < 0:
            reward += health_change * 0.01  # Small penalty for losing health
        
        # Landmark exploration reward
        min_landmark_dist = np.min(obs['landmark_distances'])
        if min_landmark_dist < 100:  # Close to a landmark
            reward += 0.2
        
        # Time-based penalty (encourage efficient movement)
        reward -= 0.001  # Small time penalty
        
        return reward
    
    def _get_reward_breakdown(self, obs: Dict[str, Any], action: np.ndarray) -> Dict[str, float]:
        """Get detailed reward breakdown for debugging"""
        breakdown = {}
        
        if obs['health'][0] > 0:
            breakdown['survival'] = 0.1        
        position_change = np.linalg.norm(obs['position'] - self.last_position)
        breakdown['movement'] = min(float(position_change * 0.01), 0.5) if position_change > 1.0 else 0.0
        
        health_change = obs['health'][0] - self.last_health
        breakdown['health'] = health_change * 0.01 if health_change < 0 else 0.0
        
        min_landmark_dist = np.min(obs['landmark_distances'])
        breakdown['landmark'] = 0.2 if min_landmark_dist < 100 else 0.0
        
        breakdown['time_penalty'] = -0.001
        
        return breakdown
    
    def _is_episode_done(self, obs: Dict[str, Any]) -> bool:
        """Check if episode should end"""
        # End if player dies
        if obs['health'][0] <= 0:
            return True
        
        # End if stuck for too long (no movement)
        if self.step_count > 100:
            recent_movement = np.linalg.norm(obs['position'] - self.last_position)
            if recent_movement < 0.1:  # Barely moved
                return True
        
        return False
    
    def _get_default_observation(self) -> Dict[str, Any]:
        """Get default observation when things fail"""
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'health': np.array([100.0]),
            'angles': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'screen': np.zeros((*self.image_size, 3), dtype=np.float32),
            'landmark_distances': np.array([1000.0] * len(self.survivor_landmarks)),
            'step_count': np.array([self.step_count])
        }
    
    def close(self):
        """Clean up environment"""
        try:
            # Release all keys
            for key in ['forward', 'backward', 'left', 'right', 'jump', 'duck', 'attack1']:
                self.game_controller.send_key_up(key)
            
            if self.debug:
                print("🔒 Environment closed - all keys released")
                
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")

if __name__ == "__main__":
    # Quick test
    print("🧪 Testing CS 1.6 ML Environment...")
    env = WorkingCS16Environment(debug=True)
    
    obs, info = env.reset()
    print(f"✅ Reset successful: {info}")
    
    # Test a few actions
    for i in range(5):
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move forward
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.3f}, Done={done}")
        
        if done:
            break
        
        time.sleep(1)
    
    env.close()
    print("🎉 Test completed!")
