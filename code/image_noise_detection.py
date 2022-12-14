# Imports
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from termcolor import colored

from image_analysis import create_avgimage
from image_parser import ImageReaderGlobal, ImageReaderCSV


"""
    For a list of files, iterates through each file and checks 

    Assumptions:
        - Files with over 70 percent of pixels as nonzero pixels are noisy and will be discarded
"""
def check_image_noise(files, configs):
    clean_images = {}
    noisy_images = {}

    # Count the number of non-zero pixels
    file_nonzeros={}
    count=0
    for image in files:
        print("    %d/%d   %s"%(count, len(files), image.split('CW')[1]),end="\r")
        imreader = ImageReaderGlobal()
        if image.split(".")[-1] == "csv":
            imreader = ImageReaderCSV(configs)
        
        img=imreader.io_read(image)
        nonzeros=np.count_nonzero(img)
        
        file_nonzeros[image]=nonzeros
        count+=1

    # Divide by classes
    classes = configs["classes"]
    for img, nz_count in file_nonzeros.items():
        classtype = img.split("_")[1]
        if classtype in classes:
            classes[classtype].append(img)
        else:
            classes[classtype] = [img]

    # Create histograms of noise distribution in the images
    for classtype in classes.keys():
        print(colored("Histogram of non-zero value count for {classtype}:", "yellow"))
        plt.hist()



    cutoff = configs["noise_tolerance"]


    # TODO: Average images before and after removing noisy images
    for classtype in classes:
        print(colored(f"The average image of {classtype} before removing noisy images...", "yellow"))
        create_avgimage(classes[classtype])

        print(colored(f"The average image of {classtype} after removing noisy images...", "yellow"))
        create_avgimage(clean_images[classtype])


    # Return the list of noisy images
    return noisy_images, clean_images