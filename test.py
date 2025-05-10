from touhou_env import touhou_env
import gym

game_path = r"C:/Users/Nathan/Desktop/touhou game files/th14/th14.steam.exe"
# game_title = "Mountain of Faith"
game_title = "Double Dealing Character"
# game_title = "Undefined Fantastic Object"
env = touhou_env(game_path, game_title)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # print(reward)
    if done:
        print("DONE")
        env.reset()
env.close()