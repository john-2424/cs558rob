import importlib, inspect
mod = importlib.import_module("modified_gym_env.reacher_env_mod")
cls = mod.ReacherBulletEnv

print("MRO:")
for c in cls.__mro__:
    print(" -", c)

import importlib, inspect
mod = importlib.import_module("modified_gym_env.reacher_env_mod")
base_cls = mod.ReacherBulletEnv.__mro__[1]

print("BASE CLASS:", base_cls)
print("BASE INIT SIGNATURE:", inspect.signature(base_cls.__init__))
print("\nBASE INIT SOURCE:\n")
print(inspect.getsource(base_cls.__init__))

import importlib, inspect
mod = importlib.import_module("modified_gym_env.reacher_env_mod")
base_cls = mod.ReacherBulletEnv.__mro__[1]

print(inspect.getsource(base_cls.render))


import gym
import modified_gym_env
import time

env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)

print("Calling render() before reset...")
try:
    env.render()
    print("render() call succeeded")
except Exception as e:
    print("render() error:", e)

obs = env.reset()

for i in range(50):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)

    try:
        env.render()
    except Exception as e:
        print("render() step error:", e)
        break

    time.sleep(1.0 / 30.0)

    if done:
        break

env.close()


import gym
import modified_gym_env
import time

env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)


# Improve offscreen render quality
env._render_width = 960
env._render_height = 720

# Zoom camera in toward the arm
env._cam_dist = 1.4
env._cam_yaw = 0
env._cam_pitch = -65

# Important: call human render before reset
env.render(mode="human")
obs = env.reset()

for i in range(300):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    time.sleep(1.0 / 60.0)
    if done:
        obs = env.reset()

env.close()