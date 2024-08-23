import os
import cv2
import sys
import numpy as np 
from utils.util import *
from typing import Union


class ImageBatcher: 
    def __init__(
            self, 
            input : str, 
            shape : Union[list, tuple], 
            dtype : np.dtype, 
            max_num_images : int = None, 
            exact_batches : bool = False
        ) -> None:
        """ 
        Args:
            input (str): The input directory to read images from.
            shape (tuple | list): The tensor shape of the batch to prepare, either in NCHW or NHWC format.
            dtype (numpy): The (numpy) datatype to cast the batched data to.
            max_num_images (int): The maximum number of images to read from the directory.
            param exact_batches (bool): This defines how to handle a number of images that is not an exact multiple of the batch size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        # Gathering Input Image
        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort() 

        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)

        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No Image Found")
            sys.exit(1) 

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape 

        # Make sure the shape has 4 values (batch size, C, W, H)
        assert len(self.shape) == 4
        self.batch_size = shape[0]  
        assert self.batch_size > 0
