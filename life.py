import cv2
from matplotlib import pyplot as plt
import numpy as np
image_file = "./temp/0.jpg"
img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
mean_val = np.mean(img)
print(mean_val)
