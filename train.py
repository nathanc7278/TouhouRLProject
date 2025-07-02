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
import psutil
import win32process
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

def make_env():
    def _init():
        env = touhou_env(game_number, game_path, game_title)
        env = SkipFrame(env, skip=4)
        return env
    return _init

def is_game_running(game_title):
    window = pygetwindow.getWindowsWithTitle(game_title)[0]
    if not window:
        return False
    _, pid = win32process.GetWindowThreadProcessId(window._hWnd)
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
    except Exception:
        return False
    return True

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def get_latest_final_model(logs_root="./logs"):
    final_models = glob.glob(os.path.join(logs_root, "**", "final_model.zip"), recursive=True)
    if not final_models:
        return None
    return max(final_models, key=os.path.getctime)

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

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)

# Cnn Policy is better for images compared to Mlp policy
log_dir = f"./logs/ppo_run_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
latest_final_model = get_latest_final_model()
if latest_final_model:
    model = PPO.load(latest_final_model, env=env, tensorboard_log=log_dir)
else:
    model = PPO("CnnPolicy", env, verbose=1, 
                learning_rate=1e-5,
                n_steps=2048,
                batch_size=64,
                n_epochs=4,
                ent_coef=0.01,
                clip_range=0.2,
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
        if not is_game_running(game_title):
            env.close()
            env = DummyVecEnv([make_env()])
            env = VecFrameStack(env, n_stack=4)
            model.set_env(env)
            continue
        model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=[checkpoint_callback, crash_callback])
except KeyboardInterrupt:
    path = os.path.join(log_dir, "final_model")
    model.save(path)