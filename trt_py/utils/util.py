import os
import numpy as np

def is_image(path):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return (os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions)  

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))