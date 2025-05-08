from touhou_env import touhou_env
import gym

env = touhou_env()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)
    if done:
        print("DONE")
        env.reset()
env.close()