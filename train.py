from touhou_env import touhou_env
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import os
import glob

class CurriculumManager:
    def __init__(self, stages, threshold=400):  # Threshold needs adjustment based on rewards in run
        self.stages = stages
        self.current_stage = 0
        self.threshold = threshold
        self.performance_history = []

    def get_current_stage(self):
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        else:
            return None
    
    def update_performance(self, episode_reward):
        self.performance_history.append(episode_reward)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

    def should_advance(self):
        if len(self.performance_history) < 20:
            return False
        average_reward = sum(self.performance_history) / len(self.performance_history)
        return average_reward >= self.threshold
    
    def advance_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Advanced to stage {self.current_stage + 1}")

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
    

class PrintEpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            print(f"Episode finished. Reward: {self.episode_reward}")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        return True

game_info = {
    10: (r"C:/Users/Nathan/Desktop/touhou game files/thcrap/th10 (lang_en-CustomRecoloredPatch)", "Mountain of Faith"),
}

game_number = 10
game_path = game_info[game_number][0]
game_title = game_info[game_number][1]

def make_env(stage):
    def _init():
        env = touhou_env(game_number, game_path, game_title, stage)
        env = SkipFrame(env, skip=4)
        return env
    return _init


def get_latest_run(logs_root="./logs"):
    runs = glob.glob(os.path.join(logs_root, "ppo_run_*", "final_model.zip"))
    if not runs:
        return None
    return max(runs, key=os.path.getctime)


stages = [1, 2, 3, 4, 5, 6]
curriculum = CurriculumManager(stages, threshold=400)

while curriculum.current_stage < len(stages):
    stage = curriculum.get_current_stage()
    print(f"Training on Stage {stage}")
    env = DummyVecEnv([make_env(stage)])
    env = VecFrameStack(env, n_stack=4)

    latest_run = get_latest_run()
    # Cnn Policy is better for images compared to Mlp policy
    log_dir = f"./logs/ppo_run_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if latest_run:
        print(f"Resuming from latest run: {latest_run}")
        model = PPO.load(latest_run, env=env, tensorboard_log=log_dir)
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

    print_reward_callback = PrintEpisodeRewardCallback()
    
    try:
        while True:
            model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=[checkpoint_callback, crash_logger_callback, print_reward_callback])

            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=False)
            curriculum.update_performance(mean_reward)
            if curriculum.should_advance():
                env.close()
                del env
                curriculum.advance_stage()
                break
    except KeyboardInterrupt:
        path = os.path.join(log_dir, "final_model")
        model.save(path)

model.save(os.path.join(log_dir, "final_model"))
print("Training complete. Final model saved.")


