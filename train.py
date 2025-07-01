from touhou_env import touhou_env
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from datetime import datetime
import os
import pygetwindow
import glob

class SkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class CrashRecoveryCallback(BaseCallback):
    def __init__(self, game_title, make_env_fn, check_every, verbose=0):
        super().__init__(verbose)
        self.game_title = game_title
        self.make_env_fn = make_env_fn
        self.check_every = check_every
        self.step_counter = 0
    
    def _on_step(self):
        self.step_counter += 1
        if self.step_counter >= self.check_every:
            self.step_counter = 0
            if not any(self.game_title in title for title in pygetwindow.getAllTitles()):
                print("Game Crashed")
                self.training_env.close()
                new_env = DummyVecEnv([self.make_env_fn()])
                new_env = VecFrameStack(new_env, n_stack=4)
                self.model.set_env(new_env)
        return True

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
    def _init():
        env = touhou_env(game_number, game_path, game_title)
        env = SkipFrame(env, skip=4)
        return env
    return _init

def is_game_running(game_number, game_info):
    title = game_info[game_number][1]
    print("Active windows:", pygetwindow.getAllTitles())
    return any(title in window for window in pygetwindow.getAllTitles())

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)

# Cnn Policy is better for images compared to Mlp policy
log_dir = f"./logs/ppo_run_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model = PPO.load(latest_checkpoint, env=env, tensorboard_log=log_dir)
else:
    model = PPO("CnnPolicy", env, verbose=1, 
            ent_coef=0.1,
            tensorboard_log=log_dir)

checkpoint_callback = CheckpointCallback(
    save_freq=2000,
    save_path=checkpoint_dir,
    name_prefix="ppo_model"
)

crash_callback = CrashRecoveryCallback(
    game_title=game_title,
    make_env_fn=make_env,
    check_every=200
)

try:
    while True:
        if not is_game_running(game_number, game_info):
            env.close()
            env = DummyVecEnv([make_env()])
            env = VecFrameStack(env, n_stack=4)
            model.set_env(env)
            continue
        model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=[checkpoint_callback, crash_callback])
except KeyboardInterrupt:
    path = os.path.join(log_dir, "final_model")
    model.save(path)