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

from image_noise_detection import check_image_noise


"""
"""
def read_configs():
    return


"""
"""
def get_images(directory):
    return


"""

    Done by Sinaro Ly
"""
def crop_image(image, depth, diameter):
    best_sum = 0
    best_index_xy = 0
    #finding the plan with most information
    for i in range(image.shape[2]):
        new_sum=np.sum(image[:,:,i,:])
        if new_sum>best_sum:
            best_sum=new_sum
            best_index_xy=i
    best_sum = 0
    best_index_xz = 0
    for i in range(image.shape[1]):
        new_sum=np.sum(image[:,i,:,:])
        if new_sum>best_sum:
            best_sum=new_sum
            best_index_xz=i
    best_index_xz -= 5
    original=np.sum(image)
    #searching for the center of the circle y axis
    ymid = 5
    for j in range(image.shape[0]):
        new_image=image[j:,:,:]
        integration=np.sum(new_image)
        #print("integration y : ", (integration/original))
        if (integration/original)<0.748:
            ymid=j
            break
    if (ymid>5):
        ymid-=5
    #searching for the center of the circle x axis
    #xmid = 5
    for j in range(image.shape[2]):
        new_image=image[:,:,j:,:]
        integration=np.sum(new_image)
        if (integration/original)<0.90:
            xmid=j
            break
    #xmid-=5
    r = int(diameter/2)
    #return image[(best_index_yz-r):(best_index_yz+r), best_index_xz:(best_index_xz+depth), best_index_yz-r:best_index_yz+r, :]
    return image[ymid:(ymid+diameter), best_index_xz:(best_index_xz+depth), xmid:xmid+diameter, 0]
    return


""" 
Rotates an image by a specified angle through the z-axis
Parameters:
    - image
    - angle (in degrees)
"""
def rotate_image(image, angle):
    return scipy.ndimage.rotate(image, angle, axes=(0,2), reshape=False)


"""
"""
def create_dataset(image_paths, output_path, n):
    for image in image_paths:
        # Load the original image
        oimg=io.imread(image)[:,40:,:320,:]
        
        # Crop the image
        cropoimg=crop_image(oimg, 210, 185)
        # plt.imshow(best_image(oimg, 1))
        
        filename=(image.split('/')[len(image.split('/'))-1]).split('.tiff')[0].split('_')
        
        # Check for any typos
        if filename[1] == 'medula':
            filename[1]='medulla'
        
        # Save the original image as a CSV
        folder=image.split('/')
        folder=folder[len(folder)-3]+'/'+folder[len(folder)-2]+'/'
              
        ofilename=output_path+folder+filename[0]+'_'+filename[1]+'_'+filename[2]+'_0_Mode3D'+'.csv'
        flatoimg=np.ravel(cropoimg)
        np.savetxt(ofilename, flatoimg)
        
        # Rotate the image n times
        angle = 360 / (n + 1)
        for i in range(1, n + 1):
            altimg=rotate_image(cropoimg, angle * i)
            # plt.imshow(best_image(altimg, 1))
            
            # Save the image as a CSV
            savename=output_path+folder+filename[0]+'_'+filename[1]+'_'+filename[2]+'_'+str(i)+'_Mode3D'+'.csv'            
            flatimg=np.ravel(altimg)
            np.savetxt(savename, flatimg)
         
            # To save as .tiff instead:
            # io.imsave(filename, altimg)
            
        print('Finished: ' + folder+filename[0]+'_'+filename[1]+'_'+filename[2])

    return


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