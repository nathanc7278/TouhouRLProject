from touhou_env import touhou_env
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from datetime import datetime
import os
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
                if info.get("crash", False):
                    print("Game Instance Crashed")
                break
        return obs, total_reward, terminated, truncated, info

class CrashLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.crash_count = 0
    
    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("crash", False):
                self.crash_count += 1
                print(f"Crash #{self.crash_count} detected at step {self.num_timesteps}")
        return True
    

game_info = {
    10: (r"C:/Users/Nathan/Desktop/touhou game files/thcrap/th10 (lang_en-CustomRecoloredPatch)", "Mountain of Faith"),
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
    model = PPO("CnnPolicy", 
            env, 
            verbose=1,
            learning_rate=5.0e-5,
            n_steps=512,
            batch_size=256,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log=log_dir)

checkpoint_callback = CheckpointCallback(
    save_freq=2000,
    save_path=checkpoint_dir,
    name_prefix="ppo_model"
)

crash_logger_callback = CrashLoggerCallback()

try:
    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=[checkpoint_callback, crash_logger_callback])
except KeyboardInterrupt:
    path = os.path.join(log_dir, "final_model")
    model.save(path)


