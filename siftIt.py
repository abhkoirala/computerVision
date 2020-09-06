import random
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
from skimage import io
import pickle

# Feature extractor
def extract_features(image_path,name,vector_size=32):
    classify = np.array([])
    label = np.array([])
    image = io.imread(image_path)
    try:
        # SIFT was removed in thee latest version of openCV hence we need to downgrade
        alg = cv2.xfeatures2d.SIFT_create()
        kps = alg.detect(image)
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None
    print(len(kps))
    print(dsc)
    classify = np.append(classify, dsc)
    label = np.append(label, name)
    allVectors.append(classify)
    allLabels.append(label)


dirpath = './randomTrainGray/'
subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, o))]
i = 0

j = 0

while (i < len(subdirs)):
    allKeypoints = []
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
            
            extract_features(srcpath,res)
            j += 1

        np.save('sift_features_now/sift_'+res+'.npy', np.asarray(allVectors))
        np.save('sift_features_now/label_'+res+'.npy', np.asarray(allLabels))

    i += 1
