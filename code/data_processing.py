"""
Fall 2022 Research
Authors:    Adrien Badre  - PhD (adrien.f.badre-1@ou.edu)
            Parker Brandt - BS  (parker.a.brandt-1@ou.edu)
            Sinaro Ly     - MS  (sinaro.ly-1@ou.edu)
"""

# Imports
import json
import numpy as np
import os
import random
import scipy
# import sklearn
import sys

# from mpl_toolkits.mplot3d import Axes3D
from multiprocessing.pool import ThreadPool as Pool
from skimage import io
from termcolor import colored

from image_noise_detection import check_image_noise


"""
Reads configs/config.json to get information on processing the input images
"""
def read_configs(filename):
    f = open(filename, 'r')
    return json.load(f)

"""
"""
def load_image(filename, image_shape):

    # Get the extension and use that to determine how to load the function
    ext = filename.split(".")[-1]
    if ext == "tiff":
        return io.imread(filename)[:,40:,:320,:]
    elif ext == "csv":
        # Will need to reshape using image shape from config file
        image = np.genfromtxt(filename)
        return image.reshape(image_shape)


"""
Reads all images from a directory and returns their paths in a list
Assumed structure is 
    path -> Kidney Folders -> Class Folder -> Images
    Example (with naming conventions):
        path/Kidney_01/k1_cortex/k1_cortex_0031_0_Mode3D.tiff
Parameters:
    - path: A string location of the path 
"""
def get_images(path, imgformat='csv'):
    filelist=[]
    try:
        kidney_folders=os.listdir(path)
        
        # Loop through each kidney and open their class folders
        for kidney in kidney_folders:
            if os.path.isdir(path+kidney):   
                class_folders=os.listdir(path+kidney)

                # Loop through each class and add each image to the filelist
                for k_class in class_folders:
                    images=os.listdir(path+kidney+'/'+k_class)

                    # Add all valid images
                    # Valid images have the same extension as imgformat parameter
                    for image in images:
                        filename=image.split('.')
                        if imgformat == filename[len(filename)-1]:
                            imgpath=path+kidney+'/'+k_class+'/'+image
                            filelist.append(imgpath)
            
    except OSError as e:
        print(e)
    return filelist


"""
    Crops an image 
    Function created by Sinaro Ly
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
def create_dataset(image_paths, output_path, n, outimg_format="csv"):
    for image in image_paths:
        # Load the original image
        oimg=io.imread(image)[:,40:,:320,:]
        
        # Crop the image
        cropoimg=crop_image(oimg, 210, 185)
        
        filename=(image.split('/')[len(image.split('/'))-1]).split('.tiff')[0].split('_')
        
        # Check for any known typos
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
            
            # Save the image as a CSV
            if outimg_format == "csv":
                savename=output_path+folder+filename[0]+'_'+filename[1]+'_'+filename[2]+'_'+str(i)+'_Mode3D'+'.csv'            
                flatimg=np.ravel(altimg)
                np.savetxt(savename, flatimg)
            elif outimg_format == "tiff":
                # To save as .tiff instead:
                io.imsave(filename, altimg)
            else:
                print(colored("Invalid file type to save as.\nValid options are: .csv and .tiff", "red"))
            
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
    # Format: [1] = config file location
    args = sys.argv

    # Read the config file for information on how to process data
    print(colored(f"Reading config file at {args[1]}...", "green"))
    configs = read_configs(args[1])
    print(colored("Configuration retrieved.\n", "green"))

    # Retrieve the actual images
    print(colored("Retrieving images from " + configs["image_location"] + "...", "green"))
    image_paths_list = get_images(configs["image_location"], imgformat=configs["image_in_format"])
    print(colored(f"Input images ({len(image_paths_list)}): ", "green"))
    for path in image_paths_list:
        print(f"\t{path}")

    # Run noise detection for all images
    print(colored("\nRunning noise analysis on images...", "green"))
    noisy_images, clean_images = check_image_noise(image_paths_list, configs)
    noisy_image_count = 0
    for classtype, image_list in noisy_images.items():
        noisy_image_count += len(image_list)

    print(colored(f"Found {noisy_image_count} noisy images...", "green"))
    if noisy_image_count > 0:
        print(colored(f"Removing: ", "green"))
        for noisy in noisy_images:
            print(noisy)


    # TODO: Create the dataset using only the good images
    all_clean_imgs = []
    for classtype, image in clean_images.items():
        all_clean_imgs.append(image)

    print(colored("Cropping and rotating images...", "green"))
    # create_dataset(image_paths=all_clean_imgs, output_path=configs["save_location"], n=configs[""], outimg_format=configs[""])


    print(colored("Image Analysis Complete!", "green"))

    return


"""
Start of Program Logic
"""
if __name__ == "__main__":
    main()