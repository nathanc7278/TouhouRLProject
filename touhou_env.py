import gym
from gym import spaces
import numpy as np
import mss
import cv2
import pydirectinput
import time
import subprocess
import pygetwindow

class touhou_env(gym.Env):
    def __init__(self):
        super().__init__()
        self.monitor = {'top': 0 , 'left': 0, 'width': 1290, 'height': 990}
        self.sct = mss.mss()
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(960, 1280, 3), dtype=np.uint8)

        self.action_space = spaces.Discrete(8)
        self.action_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.curr_life = None
        self.one_life_val = 3
        self.num_lives = 3

        self._start_game()

    
    def _start_game(self):
        try:
            game_path = "C:/Users/Nathan/Desktop/touhou game files/th10/vpatch.exe"
            self.game_process = subprocess.Popen(game_path)
            time.sleep(6)
            all_windows = pygetwindow.getAllTitles()
            
            window_title = "Mountain of Faith"
            matched_window = None
            for title in all_windows:
                if window_title in title:
                    matched_window = title
                    break
            if matched_window:
                window = pygetwindow.getWindowsWithTitle(matched_window)[0]
                print(window)
                window.restore()
                window.activate()
                window.moveTo(0, 0)
                time.sleep(2)
                for _ in range(10):
                    pydirectinput.press('z')
                    time.sleep(0.1)
                time.sleep(2)
            else:
                print(f"Window with title containing '{window_title}' not found.")
        except Exception as e:
            print(f"Error starting the game: {e}")

    def reset(self):
        time.sleep(3)
        pydirectinput.press('z')
        pydirectinput.press('r')
        self.curr_life = None
        self.num_lives = 3
        time.sleep(3)

    def step(self, action):
        key = self.action_keys[action]
        if (key == 0):
            pydirectinput.keyDown('shift')
        if (key == 1): 
            pydirectinput.keyUp('shift')
        if (key == 2): 
            pydirectinput.press('left')
        if (key == 3): 
            pydirectinput.press('right')
        if (key == 4): 
            pydirectinput.press('up')
        if (key == 5):
            pydirectinput.keyDown('z')
        if (key == 6):
            pydirectinput.keyUp('z')
        if (key == 7):
            # no operation
            pass
        

        time.sleep(0.016)       # time for 1 frame when 60 fps

        obs, new_life = self._get_obs()

        if self.curr_life == None:
            self.curr_life = np.mean(new_life)
            print(self.curr_life)
        if self.curr_life - np.mean(new_life) > self.one_life_val:
            time.sleep(0.5)
            reward = -100
            self.curr_life = self.curr_life - (self.curr_life - np.mean(new_life))
            self.num_lives -= 1
            print(self.curr_life)
        else:
            reward = 1
        info = {}
        if self.num_lives == 0:
            done = True
        else:
            done = False

        return obs, reward, done, info


    def _get_obs(self):
        image = np.array(self.sct.grab(self.monitor))[:, :, :3]
        cv2.imwrite("./temp/image.jpg", image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./temp/gray_image.jpg", gray_image)
        life = np.array(self.sct.grab(self.monitor))[230:280, 1000:1280, :3]
        gray_life = cv2.cvtColor(life, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./temp/gray_life.jpg", gray_life)
        return gray_image, gray_life
    
    
    def close(self):
        cv2.destroyAllWindows()
