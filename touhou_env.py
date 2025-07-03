import gymnasium as gym
import os
from gymnasium import spaces
import numpy as np
import mss
import cv2
import pydirectinput
import time
import pygetwindow
from pymem import Pymem
import psutil

ADDRESS_OF_LIVES = 0x00474C70
ADDRESS_OF_POWER = 0x00474C48

class touhou_env(gym.Env):
    def __init__(self, game_number, game_path, game_title):
        super().__init__()
        self.game_number = game_number
        self.game_path = game_path
        self.game_title = game_title

        self.monitor = {'top': 0 , 'left': 0, 'width': 1280, 'height': 960}
        self.sct = mss.mss()
        self.process = None

        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete([9, 2, 2])
        self.movement_mapping = {
            0: [],
            1: ['left'],
            2: ['right'],
            3: ['up'],
            4: ['down'],
            5: ['left', 'up'],
            6: ['right', 'up'],
            7: ['left', 'down'],
            8: ['right', 'down']
        }
        self.held_keys = []
        
        self.num_lives = 0
        self.power = 0
        self._start_game()

    
    def _start_game(self):
        try:
            os.startfile(self.game_path)
            time.sleep(6)
            all_windows = pygetwindow.getAllTitles()
            matched_window = None
            for title in all_windows:
                if self.game_title in title:
                    matched_window = title
                    break
            if not matched_window:
                print(f"Window with title containing '{self.game_title}' not found.")
                return
            window = pygetwindow.getWindowsWithTitle(matched_window)[0]
            window.restore()
            window.activate()
            window.moveTo(0, 0)
            time.sleep(6)
            for i in range(8):
                pydirectinput.press('z')
                time.sleep(0.2)
            time.sleep(3)
            self.process = Pymem("th10.exe")
            if not self.is_process_alive():
                return
        except Exception as e:
            print(f"Error starting the game: {e}")

    def reset(self, *, seed=None, options=None):
        if not self.is_process_alive():
            self._start_game()
        for k in self.held_keys:
            pydirectinput.keyUp(k)
            self.held_keys.remove(k)
        time.sleep(1)
        pydirectinput.press('esc')
        time.sleep(1)
        pydirectinput.press('r')
        time.sleep(1)
        obs = self._get_obs()
        self.current_step = 0
        info = {
            "lives": self.num_lives
        }
        return obs, info

    def step(self, action):
        movement, shift, shoot = action
        for k in self.held_keys:
            pydirectinput.keyUp(k)
            self.held_keys.remove(k)

        keys_to_hold = self.movement_mapping.get(movement, [])
        if shift == 1:
            pydirectinput.keyDown("shift")
            self.held_keys.append("shift")
        if shoot == 1:
            pydirectinput.keyDown('z')
            self.held_keys.append("z")
        for k in keys_to_hold:
            pydirectinput.keyDown(k)
            self.held_keys.append(k)
        prev_lives = self.num_lives
        prev_power = self.power
        obs = self._get_obs()
        try:
            self.num_lives = self.process.read_int(ADDRESS_OF_LIVES)
            self.power = self.process.read_int(ADDRESS_OF_POWER)
        except Exception as e:
            reward = 0
            terminated = True
            truncated = True
            info = {"lives": self.num_lives, "crash": True}
            return obs, reward, terminated, truncated, info

        if prev_lives > self.num_lives:
            reward = -10
        elif prev_lives < self.num_lives:
            reward = 10
        elif prev_lives == self.num_lives:
            reward = 0.1
        
        if prev_power < self.power:
            reward = 0.3

        if self.num_lives == 0:
            terminated = True
        else:
            terminated = False
        truncated = False
        info = {
            "lives": self.num_lives, "crash": False
        }
        return obs, reward, terminated, truncated, info


    def _get_obs(self):
        image = np.array(self.sct.grab(self.monitor))[:, :, :3]
        # cv2.imwrite("./assets/image.jpg", image)
        game_area = image[55:965, 45:855]
        resized = cv2.resize(game_area, (84, 84), interpolation=cv2.INTER_AREA)
        cv2.imwrite("./assets/resized.jpg", resized)
        return resized

    def is_process_alive(self):
        if not self.process or not self.process.process_handle:
            return False
        try:
            proc = psutil.Process(self.process.process_id)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False
        
    
    def close(self):
        cv2.destroyAllWindows()



