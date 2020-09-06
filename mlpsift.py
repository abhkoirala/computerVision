import cv2
import numpy as np
import os
import pandas as pd
import csv
from skimage import io
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from sklearn.svm import SVC

# The cluster should be about number of classes (26) * 10
def read_and_clusterize(num_cluster=260):
    
    dirpath = './randomTrainGray/'
    subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, o))]
    i = 0
    j = 0
    
    sift_keypoints = []
    
    while (i < len(subdirs)):
        print(i)
        res = subdirs[i][subdirs[i].rindex('/') + 1:]
        oldDir = dirpath + res
        filenames = os.listdir(oldDir)
        for fname in filenames:
            if (fname != '.DS_Store'):
                print(j)
                srcpath = os.path.join(oldDir, fname)
                print(srcpath)          
                #read image
                image = cv2.imread(srcpath,1)
                # grayscale conversion
                image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                # SIFT extraction
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(image,None)
                sift_keypoints.append(descriptors)
                
                j += 1
        i += 1

    sift_keypoints=np.asarray(sift_keypoints)
    sift_keypoints=np.concatenate(sift_keypoints, axis=0)
    print("Training kmeans")    
    kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(sift_keypoints)
    return kmeans

a = read_and_clusterize()
dump(a, 'kmeans.joblib')

#Changing the class name from alphabets to number
def define_class(img_patchname):
    
    if(img_patchname=="A"):
        class_image=0

    if(img_patchname=="B"):
        class_image=1
    
    if(img_patchname=="C"):
        class_image=2
        
    if(img_patchname=="D"):
        class_image=3
        
    if(img_patchname=="E"):
        class_image=4
    
    if(img_patchname=="F"):
        class_image=5
    
    if(img_patchname=="G"):
        class_image=6
    
    if(img_patchname=="H"):
        class_image=7
    
    if(img_patchname=="I"):
        class_image=8
    
    if(img_patchname=="J"):
        class_image=9
    
    if(img_patchname=="K"):
        class_image=10
    
    if(img_patchname=="L"):
        class_image=11
    
    if(img_patchname=="M"):
        class_image=12
    
    if(img_patchname=="N"):
        class_image=13
    
    if(img_patchname=="O"):
        class_image=14
    
    if(img_patchname=="P"):
        class_image=15
    
    if(img_patchname=="Q"):
        class_image=16
    
    if(img_patchname=="R"):
        class_image=17
    
    if(img_patchname=="S"):
        class_image=18
    
    if(img_patchname=="T"):
        class_image=19
    
    if(img_patchname=="U"):
        class_image=20
    
    if(img_patchname=="V"):
        class_image=21
    
    if(img_patchname=="W"):
        class_image=22
    
    if(img_patchname=="X"):
        class_image=23
    
    if(img_patchname=="Y"):
        class_image=24
    
    if(img_patchname=="Z"):
        class_image=25

    return class_image


#generating the feature vector after the kmeans clustering model 
#creating the histogram of classified keypoints from kmeans 
def calculate_centroids_histogram(model):
    
    dirpath = './randomTrainGray/'
    subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, o))]
    i = 0
    j = 0

    feature_vectors=[]
    class_vectors=[]
    
    while (i < len(subdirs)):
        print(i)
        res = subdirs[i][subdirs[i].rindex('/') + 1:]
        oldDir = dirpath + res
        filenames = os.listdir(oldDir)
        for fname in filenames:
            if (fname != '.DS_Store'):
                print(j)
                srcpath = os.path.join(oldDir, fname)
                print(srcpath)  
                #read image
                image = cv2.imread(srcpath,1)
                #grayscale conversion
                image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #SIFT extraction
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(image,None)
                predict_kmeans=model.predict(descriptors)
                #histogram calculation
                hist, bin_edges=np.histogram(predict_kmeans, bins=260)
                feature_vectors.append(hist)
                class_sample=define_class(res)
                class_vectors.append(class_sample)
                
                j += 1
        i += 1

    feature_vectors=np.asarray(feature_vectors)
    class_vectors=np.asarray(class_vectors)
    #vectors and classes are returned for classification
    return class_vectors, feature_vectors

y, x = calculate_centroids_histogram(a)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = MLPClassifier(hidden_layer_sizes=(40,40,40), max_iter=40000).fit(x_train, y_train)
y_pred = model.predict(x_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Training: ", model.score(x_train, y_train))
print("Testing: ", model.score(x_test, y_test))

dump(model, './now_output/sift_mlp.joblib')




