# Imports
import numpy as np

from skimage import io
from data_processing import load_image
from image_analysis import create_avgimage

"""
    For a list of files, iterates through each file and checks 

    Assumptions:
        - Files with over 70 percent of pixels as nonzero pixels are noisy and will be discarded
"""
def check_noise_levels(files, classname):
    # Count the number of non-zero pixels
    file_nonzeros={}
    count=0
    for image in files:
        print("    %d/%d   %s"%(count, len(files), image.split('CW')[1]),end="\r")
        
        img=load_image(image)
        nonzeros=np.count_nonzero(img)
        
        file_nonzeros[image]=nonzeros
        count+=1

    # TODO: Print noise info

    # TODO: Histograms

    # TODO: Average images before and after removing noisy images

    return