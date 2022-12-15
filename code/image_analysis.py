# Imports
import numpy as np

from skimage import io
from image_parser import ImageReaderGlobal, ImageReaderCSV


"""
Takes in a set of images as files, and then sums up their values to create an average image
"""
def create_avgimage(files, configs):
    average_image=np.zeros((320, 479, 320))
    imreader = ImageReaderGlobal()
    if configs["image_out_format"] == "csv":
        imreader = ImageReaderCSV(configs)

    for file in files:
        average_image+=imreader.io_read(file)[:,:,:320,0]

    return average_image