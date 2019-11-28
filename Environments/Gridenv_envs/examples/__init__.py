from gym.envs.registration import register

# =============================================================================
# KEY-DOOR
# =============================================================================
for i in (10,20,30):
    register(id='GE_MazeKeyDoor-v%i'%i,
         entry_point='gridenvs.examples.key_door:maze%ix%i'%(i, i),
         kwargs={'max_moves': 1000, 'key_reward': True},
         nondeterministic=False)