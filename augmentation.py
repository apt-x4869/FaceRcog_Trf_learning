import cv2
import numpy as np
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise

#Lets define functions for each operation
def anticlockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)
# warp not recommended
'''
def warp_shift(image): 
    transform = AffineTransform(translation=(0,40))  #chose x,y values according to your convinience
    warp_image = warp(image, transform, mode="wrap")
    return warp_image
'''
transformations = {'rotate anticlockwise': anticlockwise_rotation,
                   'rotate clockwise': clockwise_rotation,
                   'horizontal flip': h_flip,
                   'vertical flip': v_flip,
                   #'warp shift': warp_shift,
                   'adding noise': add_noise,
                   'blurring image':blur_image
                   }

def AugSingleDir(path):
    #path to original images
    images_path=path
    #path to augmented images
    #you can change to store 
    augmented_path=images_path+"Aug"
    os.mkdir(augmented_path)
    
    # image = glob.glob(path+'/*.jpg')
    #to store images paths
    images=[]
    for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
        images.append(os.path.join(images_path,im))

    #no of images to generate
    images_to_generate=30
    #iterator initialize to 1
    i=1
    while i<=images_to_generate:    
        image=random.choice(images)
        original_image = io.imread(image)
        transformed_image=None
        n = 0       #variable to iterate till number of transformation to apply
        transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image
        
        while n <= transformation_count:
            key = random.choice(list(transformations)) #randomly choosing method to call
            transformed_image = transformations[key](original_image)
            n = n + 1
            
        new_image_path= "%s/augmented_image_%s.jpg" %(augmented_path, i)
        transformed_image = img_as_ubyte(transformed_image)  #Convert an image to unsigned byte format, with values in [0, 255].
        transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it
        cv2.imwrite(new_image_path, transformed_image) # save transformed image to path
        i =i+1
        #to generate more images, put above 3 statement inside while n<... loop

#for Multiple dir in one directory
def AugMultiDir(path):
    dirList=[]
    for im in os.listdir(path):
        dirList.append(os.path.join(path,im))
    for image_folder in dirList :
        AugSingleDir(image_folder)

AugMultiDir("/root/mlopsproject/UntitledFolder/Testing")
