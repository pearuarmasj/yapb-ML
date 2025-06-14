# YaPB Zombie Bot Enhancement Plan

## Current YaPB Zombie Functionality Analysis

YaPB already has comprehensive zombie support:

### Existing Features:
1. **Creature Detection (`m_isCreature`)**
   - Model-based detection: zombie ('zo'), chicken ('ch')
   - Team-based detection (`m_isOnInfectedTeam`)
   - Method: `isCreature()` returns true for zombies

2. **Zombie Mode Detection (`GameFlags::ZombieMod`)**
   - Auto-detects zombie mods via ConVars (zp_delay, etc.)
   - Custom configuration in custom.cfg
   - Support for major zombie mod plugins

3. **Enhanced Combat for Creatures**
   - Higher hunt desire (TaskPri::Attack) for creatures
   - Special case handling in `reactOnEnemy()` for creatures
   - Simplified reachability: distance < 118.0f units

4. **Task System Integration**
   - Task filtering prioritizes hunting for creatures
   - Seek cover desire disabled for creatures
   - Knife mode detection for creatures

5. **Zombie-Specific Configurations**
   - ZMDetectCvar, ZMDelayCvar in custom.cfg
   - ZMInfectedTeam configuration
   - Integration with CSDM detection

## Proposed Enhancements from SyPB

### 1. Advanced Zombie Target Management
- **SetMoveTarget() System**: Enhanced target tracking for zombies
- **ZombieModeAi()**: Dedicated AI for zombie hunting behavior
- **Target Entity Management**: Better target switching and prioritization

### 2. Enhanced Zombie Movement
- **Aggressive Pathfinding**: Prioritize direct routes to humans
- **Jump/Climb Behavior**: Enhanced mobility for zombies
- **Speed Modulation**: Dynamic speed based on proximity to targets

### 3. Improved Human Hunting
- **Multi-Target Assessment**: Evaluate multiple human targets
- **Collaborative Hunting**: Zombies share target information
- **Persistence**: Zombies don't give up easily on targets

### 4. Configuration Variables
- **Zombie Hunt Range**: Configurable detection radius
- **Zombie Speed Multiplier**: Speed boost for zombies
- **Hunt Update Frequency**: How often zombies reassess targets
- **Aggression Level**: Zombie behavior intensity

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Add zombie-specific ConVars for tuning
2. Enhance the existing `reactOnEnemy()` for creatures
3. Improve target entity management

### Phase 2: Advanced AI
1. Implement zombie-specific hunt logic
2. Add collaborative target sharing
3. Enhanced movement patterns

### Phase 3: Integration
1. Test with popular zombie mods
2. Performance optimization
3. Configuration fine-tuning

## Files to Modify
- `src/botlib.cpp` - Core zombie logic enhancements
- `src/tasks.cpp` - Zombie-specific task handling
- `src/combat.cpp` - Enhanced creature combat
- `src/engine.cpp` - Additional ConVars
- `inc/yapb.h` - New member variables if needed

This plan builds upon YaPB's excellent existing zombie foundation while adding SyPB's advanced features.
