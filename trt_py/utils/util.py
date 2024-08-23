import os

def is_image(path):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return (os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions)  