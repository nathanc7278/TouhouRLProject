import gym
import os
from gym import spaces
import numpy as np
import mss
import cv2
import pydirectinput
import time
import pygetwindow

class touhou_env(gym.Env):
    def __init__(self, game_number, game_path, game_title):
        super().__init__()
        self.game_number = game_number
        self.game_path = game_path
        self.game_title = game_title

        self.monitor = {'top': 0 , 'left': 0, 'width': 1290, 'height': 990}
        self.sct = mss.mss()

        self.observation_space = spaces.Box(low=0, high=255, shape=(960, 1280, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(8)
        self.action_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.life_bar_x = 0
        self.life_bar_y = 0
        self.life_bar_h = 0
        self.num_lives = 0

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
                exit(1)
                
            window = pygetwindow.getWindowsWithTitle(matched_window)[0]
            window.restore()
            window.activate()
            window.moveTo(0, 0)
            time.sleep(6)
            for i in range(8):
                if self.game_number == 15 and i == 1:   # LoLK needs to click down once to go to Legacy mode
                    pydirectinput.press('down')
                pydirectinput.press('z')
                time.sleep(0.2)
            time.sleep(3)
            self.find_life_bar_template()
            if (self.life_bar_x == 0 and self.life_bar_y == 0):
                print("Life bar not found")
                exit(1)
        except Exception as e:
            print(f"Error starting the game: {e}")

    def reset(self):
        time.sleep(3)
        pydirectinput.press('z')
        pydirectinput.press('down')
        pydirectinput.press('z')
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

        obs = self._get_obs()
        print(self.num_lives)
        reward = 0
        done = 0
        info = 0
        return obs, reward, done, info


    def _get_obs(self):
        image = np.array(self.sct.grab(self.monitor))[:, :, :3]
        cv2.imwrite("./assets/image.jpg", image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./assets/gray_image.jpg", gray_image)
        life_image = gray_image[self.life_bar_y: self.life_bar_y + self.life_bar_h, self.life_bar_x:1280]
        cv2.imwrite("./assets/life_image.jpg", life_image)
        threshold = 100
        if self.game_number in [12, 14, 15, 17, 18]:          # These games have high contrast around lives
            threshold = 240
        _, bw_image_lives = cv2.threshold(life_image, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite("./assets/bw_image_lives.jpg", bw_image_lives)
        coutours, _ = cv2.findContours(bw_image_lives, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
        self.num_lives = len(coutours) + 1
        return gray_image
    
    def find_life_bar_template(self):
        image = np.array(self.sct.grab(self.monitor))[:, :, :3]
        templates = {
            6: "./assets/templates/eosd_lives.jpg",
            7: "./assets/templates/pcb_lives.jpg",
            8: "./assets/templates/in_lives.jpg",
            10: "./assets/templates/mof_lives.jpg",
            11: "./assets/templates/sa_lives.jpg",
            12: "./assets/templates/ufo_lives.jpg",
            13: "./assets/templates/td_lives.jpg",
            14: "./assets/templates/ddc_lives.jpg", 
            15: "./assets/templates/lolk_lives.jpg",
            16: "./assets/templates/hsifs_lives.jpg",
            17: "./assets/templates/wbawc_lives.jpg",
            18: "./assets/templates/um_lives.jpg"
        }
        template = cv2.imread(templates[self.game_number])
        if template is None:
            print("template failed to load")
            exit(1)
        res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.life_bar_x = max_loc[0] + 120
        self.life_bar_y = max_loc[1]
        self.life_bar_h = template.shape[0]


    def close(self):
        cv2.destroyAllWindows()
