"""
Adapted from code on https://github.com/thepanlab/medical-imaging-framework/tree/processing_3d
Written by:
    - Parker Brandt
    - Jessica Shaw
"""

# Imports
import numpy as np

from abc import ABC, abstractclassmethod
from skimage import io

"""
Abstract ImageReader class
"""
class ImageReader(ABC):
    def __init__(self):
        return

    @abstractclassmethod
    def io_read(self, filename):
        # Reads an image from a file and loads it into memory
        pass

    @abstractclassmethod
    def save_img(self, save_path):
        # Saves an image
        pass


"""
Handles reading and loading of files that can be read using io.imread
Includes:
    - .jpg
    - .jpeg
    - .png
    - .tiff
"""
class ImageReaderGlobal(ImageReader):
    def __init__(self):
        ImageReader.__init__(self)
        return

    def io_read(self, filename):
        return io.imread(filename.numpy().decode())

    def save_img(self, save_path):
        return


"""
Handles reading and loading of .csv images
Notes:
    - CSV images cannot be read by io.imread()
    - Will be stored as 1-Dimensional arrays, and when loaded will need to be reshaped
"""
class ImageReaderCSV(ImageReader):
    def __init__(self, configs):
        ImageReader.__init__(self)
        self._configs = configs
        return
    
    def io_read(self, filename):
        image = np.genfromtxt(filename)
        if "csv_shape" in self._configs:
            image = image.reshape(self._configs["csv_shape"])
        else:
            # Use a default reshape
            image = image.reshape(185, 210, 185, 1)
        return

    def save_img(self, save_path):
        return