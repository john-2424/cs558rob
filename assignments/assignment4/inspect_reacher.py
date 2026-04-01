import gym
import modified_gym_env
import pybullet  # needed for rendering-related environment support

env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs = env.reset()
print("Initial observation type:", type(obs))
print("Initial observation shape:", getattr(obs, "shape", None))
print("Initial observation:", obs)

print("Action low:", env.action_space.low)
print("Action high:", env.action_space.high)

env.close()