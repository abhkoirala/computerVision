import numpy as np
import cv2
from skimage.filters import gabor
from math import pi
import os

# Function to extract Gabor features
def get_gabor_feature(image, name):
    classify = np.array([])
    label = np.array([])
    # For 5 scales
    for i in range(0, 5, 1):
        # 8 orientations
        for j in range(0, 8, 1):
            # Get the real values from gabor function
            real_val = gabor(image, frequency=(i + 1) / 10.0, theta=j * pi / 8)[0]
            # Get the imaginary values from gabor function
            img_val = gabor(image, frequency=(i + 1) / 10.0, theta=j * pi / 8)[1]
            # Get the square of both values and add them up to get a complete result
            result = real_val * real_val + img_val * img_val
            res_mean = np.mean(result)
            classify = np.append(classify, res_mean)
            label = np.append(label, name)
    print('Gabor Features:')
    print(classify)
    print('Length Gabor Features:')
    print(len(classify))
    allVectors.append(classify)
    allLabels.append(name)

# Get values from this path
dirpath = './randomTrain/'
subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, o))]
i = 0

j = 0

# To get all images from the path
while (i < len(subdirs)):
    allVectors = []
    allLabels = []
    print(i)
    res = subdirs[i][subdirs[i].rindex('/') + 1:]
    oldDir = dirpath + res
    filenames = os.listdir(oldDir)
    for fname in filenames:
        if (fname != '.DS_Store'):
            print(j)
            srcpath = os.path.join(oldDir, fname)
            print(srcpath)
            # read each image and apply gabor extraction function
            img = cv2.imread(srcpath, 0)
            get_gabor_feature(img, res)
            j += 1
        
        # Save the values and labels
        np.save('gabor_features_now/gabor'+res+'.npy', np.asarray(allVectors))
        np.save('gabor_features_now/label'+res+'.npy', np.asarray(allLabels))

    i += 1