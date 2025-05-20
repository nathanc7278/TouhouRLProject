from touhou_env import touhou_env
import gym

game_info = {
    6: (r"C:/Users/Nathan/Desktop/touhou game files/th6/th06 (en)", "The Embodiment of Scarlet Devil"),
    7: (r"C:/Users/Nathan/Desktop/touhou game files/th7/th07 (en)", "Perfect Cherry Blossom"),
    8: (r"C:/Users/Nathan/Desktop/touhou game files/th8/th08 (en)", "Imperishable Night"),
    10: (r"C:/Users/Nathan/Desktop/touhou game files/th10/th10 (en)", "Mountain of Faith"),
    11: (r"C:/Users/Nathan/Desktop/touhou game files/th11/th11 (en)", "Subterranean Animism"),
    12: (r"C:/Users/Nathan/Desktop/touhou game files/th12/th12 (en)", "Undefined Fantastic Object"),
    13: (r"C:/Users/Nathan/Desktop/touhou game files/th13/th13 (en)", "Ten Desires"),
    14: (r"C:/Users/Nathan/Desktop/touhou game files/th14/th14 (en)", "Double Dealing Character"),
    15: (r"C:/Users/Nathan/Desktop/touhou game files/th15/th15 (en)", "Legacy of Lunatic Kingdom"),
    16: (r"C:/Users/Nathan/Desktop/touhou game files/th16/th16 (en)", "Hidden Star in Four Seasons"),
    17: (r"C:/Users/Nathan/Desktop/touhou game files/th17/th17 (en)", "Wily Beast and Weakest Creature"),
    18: (r"C:/Users/Nathan/Desktop/touhou game files/th18/th18 (en)", "Unconnected Marketeers")
}

game_number = 12
game_path = game_info[game_number][0]
game_title = game_info[game_number][1]

env = touhou_env(game_number, game_path, game_title)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # print(reward)
    if done:
        print("DONE")
        env.reset()
env.close()