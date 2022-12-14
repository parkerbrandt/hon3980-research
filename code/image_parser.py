"""
Adapted from code on https://github.com/thepanlab/medical-imaging-framework/tree/processing_3d
Written by:
    - Parker Brandt
    - Jessica Shaw
"""

# Imports
from abc import ABC, abstractclassmethod

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
    def parse_image(self):
        # Parses an image given a filename and some parameters
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
        return

    def parse_image(self):
        return


"""
Handles reading and loading of .csv images
Notes:
    - CSV images cannot be read by io.imread()
    - Will be stored as 1-Dimensional arrays, and when loaded will need to be reshaped
"""
def ImageReaderCSV(ImageReader):
    def __init__(self, configs):
        ImageReader.__init__(self)
        self.configs = configs
        return
    
    def io_read(self, filename):
        return

    def parse_image(self):
        return