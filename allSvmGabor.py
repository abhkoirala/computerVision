import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import re

# Function with path given from UI_design class
def svm(path):
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
    model = SVC(gamma='auto').fit(data, labels)
    
    
    #Computing the SVM for the complete dataset and saving the model
    dump(model, './now_output/finalsvmgabor.joblib')