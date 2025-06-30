from touhou_env import touhou_env, SkipFrame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from datetime import datetime
import os

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

game_number = 10
game_path = game_info[game_number][0]
game_title = game_info[game_number][1]

def make_env():
    return touhou_env(game_number, game_path, game_title)

env = DummyVecEnv([make_env])
env = SkipFrame(env, skip=4)
env = VecFrameStack(env, n_stack=4)

# Cnn Policy is better for images compared to Mlp policy
log_dir = f"./logs/ppo_run_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
model = PPO("CnnPolicy", env, verbose=1, 
            ent_coef=0.1,
            tensorboard_log=log_dir)
try:
    loop_count = 0
    while True:
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        loop_count += 1
        checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_{loop_count}_million_steps")
        model.save(checkpoint_path)
except KeyboardInterrupt:
    path = os.path.join(log_dir, "final_model")
    model.save(path)