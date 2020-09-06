import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import os, re

# Function with path given from UI_design class
def mlp(path):
    i=0
    feature_array=[]
    labels_array=[]
    for f in path:
        # Get all gabor feature files
       if re.match('g', f):
           features = np.load('./gabor_features/'+f).tolist()
           feature_array += features
           i+=1
           
    feature_array = np.array(feature_array)
    
    for f in path:
        # Get all the labels
       if re.match('l', f):
           labels = np.load('./gabor_features/'+f).tolist()
           labels_array += labels
           i+=1
           
    data = np.array(feature_array)
    labels=np.array(labels_array)
    # Train the model usig MLP
    model = MLPClassifier(hidden_layer_sizes=(40,40,40), max_iter=40000).fit(data, labels)
    
    
    #Computing the MLP for the complete dataset and saving the model
    dump(model, './now_output/finalmlpgabor.joblib')