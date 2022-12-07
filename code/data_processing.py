"""
Fall 2022 Research
Authors:    Adrien Badre  - PhD (adbadre@ou.edu)
            Parker Brandt - BS  (parker.a.brandt-1@ou.edu)
            Sinaro Ly     - MS  (sinaro.ly@ou.edu)
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy
import sklearn
import sys
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
from multiprocessing.pool import ThreadPool as Pool
from skimage import io
from termcolor import colored






"""
Main Function
    - Reads in command-line arguments
    - Parses and reads a config json file
    - Uses data from config file to determine input/output, and how to augment data
"""
def main():

    # Get command-line arguments
    # Format: [0] = config file location, [1] = ...
    args = sys.argv


    print(colored("Reading config file...", "green"))
    configs = read_configs(args[0])

    print(colored("Retrieving images", "green"))
    image_paths_list = get_images()
    create_dataset()

    return


"""
Start of Program Logic
"""
if __name__ == "__init__":
    main()