import cv2
import numpy as np
import os

# Function to convert from RGB to Gray for sift
def pre_process(filePath,name,dest):
    
    img=cv2.imread(filePath)
    img_HSV = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
    fullpath= dest+'/'+name
    cv2.imwrite(fullpath,img_HSV)

    
#Set source directory and destination directories
dirpath = './asl_alphabet_train'
destDirectory = './randomTrainGray/'

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

        pre_process(srcpath,fname,newDest)

    i+=1
print(len(subdirs))
# Call the sift python file for further process
import siftIt


