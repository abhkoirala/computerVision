import cv2
import numpy as np
import os


# Function to do preprocessing before doing gabor feature extraction
def preprocess_for_gabor(filePath,name,dest):
    
    img=cv2.imread(filePath)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    #Morphological Operations
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))    
    #Binary conversion
    HSV_result = cv2.bitwise_not(HSV_mask)
    fullpath= dest+'/'+name
    cv2.imwrite(fullpath,HSV_result)

    
#Set source directory and destination directories
dirpath = './asl_alphabet_train'
destDirectory = './randomTrain/'

#Get List of sub directories
subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,o))]
i=0
while(i<len(subdirs)):
    res=subdirs[i][subdirs[i].rindex('/')+1:]
    newDest = destDirectory+res
    oldDir = dirpath+'/'+res
    os.mkdir(newDest)
    filenames = os.listdir(oldDir)
    for fname in filenames: 
        srcpath = os.path.join(oldDir, fname)
        preprocess_for_gabor(srcpath,fname,newDest)
    i+=1
print(len(subdirs))
# Run the gaborextract python file next
import gaborextract

