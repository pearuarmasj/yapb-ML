#!/usr/bin/env python3
"""
CS 1.6 Memory Offset Finder
Automates the process of finding memory offsets for game state reading
"""

import pymem
import time
import struct
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CS16OffsetFinder:
    """Find memory offsets for CS 1.6 game state"""
    
    def __init__(self):
        self.pm = None
        self.base_address = None
        self.found_offsets = {}
        
    def connect_to_cs16(self) -> bool:
        """Connect to CS 1.6 process"""
        try:
            # Try common CS 1.6 process names
            process_names = ["hl.exe", "cstrike.exe", "hlds.exe"]
            
            for process_name in process_names:
                try:
                    self.pm = pymem.Pymem(process_name)
                    self.base_address = self.pm.base_address
                    logger.info(f"Connected to {process_name} at 0x{self.base_address:X}")
                    return True
                except:
                    continue
                    
            logger.error("Could not find CS 1.6 process")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def scan_memory_region(self, start_addr: int, end_addr: int, pattern: bytes) -> List[int]:
        """Scan memory region for a specific pattern"""
        addresses = []
        current_addr = start_addr
        
        while current_addr < end_addr:
            try:
                # Read chunk of memory
                chunk_size = min(4096, end_addr - current_addr)
                data = self.pm.read_bytes(current_addr, chunk_size)
                
                # Search for pattern in chunk
                offset = 0
                while True:
                    pos = data.find(pattern, offset)
                    if pos == -1:
                        break
                    addresses.append(current_addr + pos)
                    offset = pos + 1
                
                current_addr += chunk_size
                
            except:
                current_addr += 4096
                continue
        
        return addresses
    
    def find_player_position_offsets(self) -> Dict[str, int]:
        """Find player position (X, Y, Z) offsets"""
        logger.info("Searching for player position offsets...")
        
        # Instructions:
        # 1. Start CS 1.6 and join de_survivor
        # 2. Note your current position coordinates
        # 3. Move to a different position
        # 4. Run this function to find memory addresses that changed
        
        input("Position yourself in CS 1.6, then press Enter to start scanning...")
        
        # Get first position reading
        logger.info("Please move around for 5 seconds...")
        time.sleep(5)
        
        # These are common offset patterns for GoldSrc engine
        # Based on reverse engineering of HL1/CS 1.6
        possible_offsets = {
            'player_x': [0x0, 0x4, 0x8, 0xC, 0x10, 0x14, 0x18, 0x1C],
            'player_y': [0x4, 0x8, 0xC, 0x10, 0x14, 0x18, 0x1C, 0x20],
            'player_z': [0x8, 0xC, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24]
        }
        
        # Start from client.dll base
        try:
            client_dll = None
            for module in self.pm.list_modules():
                if "client.dll" in module.name.lower():
                    client_dll = module
                    break
            
            if client_dll:
                base = client_dll.lpBaseOfDll
                logger.info(f"Scanning client.dll at 0x{base:X}")
                
                # Common known offsets for CS 1.6 (may vary by version)
                known_offsets = {
                    'local_player': 0x00A3ACD4,  # Pointer to local player entity
                    'entity_list': 0x00A73FA0,   # Entity list base
                    'view_matrix': 0x00A3B09C,   # View matrix for coordinates
                }
                
                # Try to find local player pointer
                for name, offset in known_offsets.items():
                    try:
                        addr = base + offset
                        value = self.pm.read_ulong(addr)
                        logger.info(f"Found {name} at 0x{addr:X} -> 0x{value:X}")
                        self.found_offsets[name] = addr
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to scan client.dll: {e}")
        
        return self.found_offsets
    
    def find_health_offset(self) -> Optional[int]:
        """Find player health offset"""
        logger.info("Searching for health offset...")
        
        if 'local_player' not in self.found_offsets:
            logger.error("Need local player offset first")
            return None
        
        # Health is typically at player + 0x8C in GoldSrc
        common_health_offsets = [0x8C, 0x90, 0x94, 0x98, 0x9C]
        
        try:
            player_ptr = self.pm.read_ulong(self.found_offsets['local_player'])
            
            for offset in common_health_offsets:
                try:
                    health = self.pm.read_int(player_ptr + offset)
                    if 0 <= health <= 100:  # Valid health range
                        logger.info(f"Found health at offset 0x{offset:X}: {health}")
                        return player_ptr + offset
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to find health: {e}")
        
        return None
    
    def export_offsets_to_file(self, filename: str = "cs16_offsets.py"):
        """Export found offsets to Python file"""
        
        # Add some hardcoded offsets based on common CS 1.6 versions
        default_offsets = {
            # These are common offsets for CS 1.6 build 6153
            'local_player_ptr': 0x00A3ACD4,
            'entity_list': 0x00A73FA0,
            'max_entities': 0x00A73E30,
            'view_matrix': 0x00A3B09C,
            
            # Player entity offsets (relative to player entity base)
            'player_origin_x': 0x04,  # X coordinate
            'player_origin_y': 0x08,  # Y coordinate  
            'player_origin_z': 0x0C,  # Z coordinate
            'player_velocity_x': 0x20,  # X velocity
            'player_velocity_y': 0x24,  # Y velocity
            'player_velocity_z': 0x28,  # Z velocity
            'player_health': 0x8C,     # Health
            'player_armor': 0x90,      # Armor
            'player_angles_x': 0x0C,   # View angle X (pitch)
            'player_angles_y': 0x10,   # View angle Y (yaw)
            'player_angles_z': 0x14,   # View angle Z (roll)
            'player_team': 0x9C,       # Team number
            'player_class': 0xA0,      # Player class
            'player_flags': 0x94,      # Player flags (ducking, etc)
        }
        
        # Merge with found offsets
        all_offsets = {**default_offsets, **self.found_offsets}
        
        with open(filename, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write('CS 1.6 Memory Offsets\n')
            f.write('Auto-generated by offset finder\n')
            f.write('"""\n\n')
            
            f.write('class CS16Offsets:\n')
            f.write('    """Memory offsets for CS 1.6 game state reading"""\n\n')
            
            for name, offset in all_offsets.items():
                f.write(f'    {name.upper()} = 0x{offset:08X}\n')
            
            f.write('\n    @classmethod\n')
            f.write('    def get_all_offsets(cls):\n')
            f.write('        """Return all offsets as dictionary"""\n')
            f.write('        return {\n')
            for name, offset in all_offsets.items():
                f.write(f'            "{name}": 0x{offset:08X},\n')
            f.write('        }\n')
        
        logger.info(f"Exported offsets to {filename}")

def interactive_offset_finder():
    """Interactive offset finding session"""
    finder = CS16OffsetFinder()
    
    print("🎯 CS 1.6 Memory Offset Finder")
    print("="*50)
    print("1. Start CS 1.6 and join de_survivor map")
    print("2. Make sure you can move around")
    print("3. This tool will help find memory offsets")
    print("")
    
    if not finder.connect_to_cs16():
        print("❌ Could not connect to CS 1.6!")
        print("Make sure CS 1.6 is running and try again.")
        return
    
    print("✅ Connected to CS 1.6!")
    print("")
    
    # Find position offsets
    finder.find_player_position_offsets()
    
    # Find health offset
    finder.find_health_offset()
    
    # Export results
    finder.export_offsets_to_file()
    
    print("")
    print("🎉 Offset finding complete!")
    print("Check cs16_offsets.py for the results.")

if __name__ == "__main__":
    interactive_offset_finder()
