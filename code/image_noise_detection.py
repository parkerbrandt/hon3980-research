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
    for image in files:
        imreader = ImageReaderGlobal()
        if image.split(".")[-1] == "csv":
            imreader = ImageReaderCSV(configs)
        
        img=imreader.io_read(image)
        nonzeros=np.count_nonzero(img)
        
        file_nonzeros[image]=nonzeros

    # Divide by classes
    classes = configs["classes"]
    for img, nz_count in file_nonzeros.items():
        classtype = img.split("_")[1]
        if classtype in classes:
            classes[classtype].append(img)
        else:
            classes[classtype] = [img]

    # Create histograms of noise distribution in the images
    for classtype, image_list in classes.items():
        print(colored("Histogram of non-zero value count for {classtype}:", "yellow"))
        nz_counts = []
        for image in image_list:
            nz_counts.append(np.count_nonzero(image))     
        plt.hist(nz_counts)   
        
    # Remove the noisy images from the dataset
    for classtype, image_list in classes.items():
        if len(clean_images[classtype]) == 0:
            clean_images[classtype] = []

        if len(noisy_images[classtype]) == 0:
            noisy_images[classtype] = []

        for image in image_list:
            if np.count_nonzero(image) / configs["image_size"] < configs["noise_tolerance"]:
                clean_images[classtype].append(image)
            else:
                noisy_images[classtype].append(image)

    # Average images before and after removing noisy images
    for classtype in classes:
        print(colored(f"The average image of {classtype} before removing noisy images...", "yellow"))
        create_avgimage(classes[classtype])

        print(colored(f"The average image of {classtype} after removing noisy images...", "yellow"))
        create_avgimage(clean_images[classtype])


    # Return the list of noisy images
    return noisy_images, clean_images