import gym
import numpy as np
import json


_hex_to_int = lambda x:int(x, 16)-128

def load_description():
    with open('ram.json', 'r') as f:
        ram = json.load(f)

    with open('inventory.json', 'r') as f:
        inventory = json.load(f)
    return ram, inventory

def will_die(obs):
    is_falling = _hex_to_int('0xd8')
    lost_life = _hex_to_int('0xba')
    _will_die =  obs[is_falling]>=8 | obs[lost_life] <5
    return _will_die

def has_key(obs):
    idx = _hex_to_int('0xc1')
    inventory = obs[idx]
    key = _hex_to_int('0x1e')
    _has_key = (inventory & key) != 0
    return _has_key

def get_xy(obs):
    x = obs[_hex_to_int('0xAA')]
    y = obs[_hex_to_int('0xAB')]
    return x,y

def get_skull(obs, direction):
    pos = obs[_hex_to_int('0xAF')] - _hex_to_int('0x16')
    if pos == 0:
        direction = 0
    elif pos == 50:
        direction = 1
    return pos,direction

def parse_state(obs,direction=0):
    x,y = get_xy(obs)
    pos, direction = get_skull(obs, direction)
    _has_key = has_key(obs)
    _will_die = will_die(obs)
    return np.array([x,y,pos, direction, _has_key, _will_die])


def  main():
    env = gym.make("MontezumaRevenge-ram-v0")
    obs = env.reset()
    direction  = 1 # direction is initalized as 1
    built_sate = parse_state(obs, direction)
    print(built_sate)
    ram, inventory = load_description()
    for idx, description in ram.items():
        print(obs[_hex_to_int(idx)], description)

if __name__ == "__main__":
    main()

