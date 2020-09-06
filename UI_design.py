# This is the python file which when run will generate the UI.
import tkinter as tk 
import numpy as np
import PIL.Image, PIL.ImageTk
import cv2
from tkinter import messagebox
from sklearn.externals import joblib
from skimage.filters import gabor
from math import pi
import os

class main_controller_class:  
    # Make the basic UI
    def generateUI(self):
        root = tk.Tk() 
        
        # Creating Frames to hold similar type objects
        label_frame = tk.Frame(root)
        label_frame.pack()
        
        radioButton1_frame = tk.Frame(root)
        radioButton1_frame.pack(fill = tk.X) 
        
        button1_frame = tk.Frame(root)
        button1_frame.pack(fill = tk.X)
        
        radioButton2_frame = tk.Frame(root)
        radioButton2_frame.pack(fill = tk.X)
        
        
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(side = tk.BOTTOM,fill = tk.X) 
        
        button2_frame = tk.Frame(root)
        button2_frame.pack(fill = tk.X)
        
        label_warning2 = tk.Label(label_frame, text="We have extracted features ourselves and provided them. The models are also trained by us")
        label_warning2.pack()
        label_warning1 = tk.Label(label_frame,fg="Red", text="NOTE: SIFT cannnot be run on the latest version of openCV. We are using OpenCV '3.4.2'")
        label_warning1.pack()
        
        
        # 4 Radio buttons for each type of Feature extraction process
        self.v = tk.IntVar()
        radiobutton1 = tk.Radiobutton(radioButton1_frame, text='Gabor Features', variable=self.v, value=0)
        radiobutton1.pack(side = tk.LEFT) 
        tk.Radiobutton(radioButton1_frame, text='SIFT Features', variable=self.v, value=1).pack(side = tk.LEFT) 
        tk.Radiobutton(radioButton1_frame, text='Our Gabor Features', variable=self.v, value=2).pack(side = tk.LEFT) 
        tk.Radiobutton(radioButton1_frame, text='Our SIFT Features', variable=self.v, value=3).pack(side = tk.LEFT)
        
        # To start extracting features from images, a button
        f_extract_button = tk.Button(button1_frame, text = 'Extract Features',command= lambda: self.extractFeatures_button_event()) 
        f_extract_button.pack(fill = tk.X)
        
        # 4 Radio buttons for each type of classifier process
        self.var = tk.IntVar()
        tk.Radiobutton(radioButton2_frame, text='SVM', variable=self.var, value=0).pack(side = tk.LEFT) 
        tk.Radiobutton(radioButton2_frame, text='MLP', variable=self.var, value=1).pack(side = tk.LEFT) 
        tk.Radiobutton(radioButton2_frame, text='Our SVM', variable=self.var, value=2).pack(side = tk.LEFT) 
        tk.Radiobutton(radioButton2_frame, text='Our MLP', variable=self.var, value=3).pack(side = tk.LEFT)
        
        # To start classification
        classify_button = tk.Button(button2_frame, text = 'Classify',command= lambda: self.classify_button_event()) 
        classify_button.pack(fill = tk.X)
        
        # Button to play video in a tkinter frame
        video_button = tk.Button(button2_frame, text = 'Start Video',command= lambda: self.start_video(root)) 
        video_button.pack(fill = tk.X)
        
        #Button to classification of the snapshot clicked
        test_image_button = tk.Button(button2_frame, text = 'Classify the snap',command= lambda: self.classify_snap()) 
        test_image_button.pack(fill = tk.X)
        
        # Exit button
        tk.Button(bottom_frame, text ='Exit',command=root.destroy).pack(side = tk.BOTTOM,fill = tk.X) 
        root.mainloop()

    # Function called when extract features button is clicked
    def extractFeatures_button_event(self):
        # Get the value of the first group of radio buttons
        selection = self.v.get()
        if selection == 0:            
            print ("Gabor Features")
            if messagebox.askyesno("Warning","This will take a lot of time. Do you want to continue?"):
                #Call the python file
                import preprocess
            
        elif selection == 1: 
            if messagebox.askyesno("Warning","This will take a lot of time. Do you want to continue?"):
                #Call the python file
                print ("SIFT")
                import pregray
        elif selection == 2: 
            print ("Our Gabor Features")
            # Use features created by us. Do nothing

        elif selection == 3: 
            print ("Our SIFT")
            # Use features created by us. Do nothing

            
    
    # Function called when classify button is clicked
    def classify_button_event(self):     
        selection = self.var.get()
        select = self.v.get()
        if selection == 0:
            if messagebox.askyesno("Warning","This will take a lot of time. Do you want to continue?"):
                if select == 0:
                    
                    print("SVM + Gabor")
                    #Call the python file
                    import allSvmGabor
                    # Give the proper path
                    allSvmGabor.svm(sorted(os.listdir('./gabor_features_now')))
                    
                elif select == 1:
                    print("SVM + SIFT")
                    #Call the python file
                    import sipiup
                    
                elif select == 2:
                    print("SVM + Our Gabor")
                    #Call the python file
                    import allSvmGabor
                      # Give the proper path
                    allSvmGabor.svm(sorted(os.listdir('./gabor_features')))
                elif select == 3:
                    print("SVM + Our SIFT")
                    #Call the python file
                    import sipiup
            
        elif selection == 1: 
            if messagebox.askyesno("Warning","This will take a lot of time. Do you want to continue?"):
                if select == 0:
                    print("MLP + Gabor")
                    #Call the python file
                    import allmlpGabor
                      # Give the proper path
                    allmlpGabor.mlp(sorted(os.listdir('./gabor_features_now')))
                elif select == 1:
                    print("MLP + SIFT")
                    #Call the python file
                    import mlpsift
                    
                elif select == 2:
                    print("MLP + Our Gabor")
                    #Call the python file
                    import allmlpGabor
                      # Give the proper path
                    allmlpGabor.mlp(sorted(os.listdir('./gabor_features')))
                elif select == 3:
                    print("MLP + Our SIFT")
                    #Call the python file
                    import mlpsift
            
        elif selection == 2: 
            print("Our SVM")
            # Use features created by us. Do nothing

        elif selection == 3: 
            print("Our MLP")
            # Use features created by us. Do nothing
            
            
    # Button Event which corresponds to starting the video camera
    def start_video(self,root):
        # The first window has frames related to Gabor filter (the thresholded values)
        self.window_threshold = tk.Toplevel()
        self.window_threshold.title("Thresholding for Gabor")
        self.canvas_2 = tk.Canvas(self.window_threshold, width = 300, height = 300)
        self.canvas_2.pack()
        self.snap_button=tk.Button(self.window_threshold, text="Snap", command=self.get_snap)
        self.snap_button.pack(anchor=tk.CENTER, expand=True,fill = tk.X)
        
        # The second window is just a regular Video Capture except its OpenCV played in a Tkinter
        self.window = tk.Toplevel()
        self.window.title("Video Capture")
        self.vid = video_capture_opencv_tkinter()
        self.canvas = tk.Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.exit_button=tk.Button(self.window, text="Exit", command=self.destroy_everything)
        self.exit_button.pack(anchor=tk.CENTER, expand=True,fill = tk.X)

        # Add a delay so the processing does not take too much power
        self.delay = 15
        self.update()
        self.window_threshold.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
           
    # Update, updates the Tkinter window every 15 milliseconds with the latest value
    def update(self):
        ret, frame,_,frame_2,__ = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            
            self.photo_2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_2))
            self.canvas_2.create_image(0, 0, image = self.photo_2, anchor = tk.NW)
         
        self.window.after(self.delay, self.update)
        
    # Saving the snapshot for classification
    def get_snap(self):
        ret, frame,_,frame_2,roi = self.vid.get_frame()
        cv2.imwrite("./camera_image_output/test_image_gabor.png", frame_2)
        cv2.imwrite("./camera_image_output/test_image_sift.png", roi)
        
    # Destroy the object and all windows to avoid errors when navigating
    def destroy_everything(self):
        if messagebox.askyesno("Warning","Are you Sure you want to Exit?"):
            self.window.destroy()
            self.window_threshold.destroy()
            del self.vid
    
    # Similar to destroy_everything function but is to handle when user press the x button instead of exit
    def on_closing(self):
        if messagebox.askyesno("Warning","Are you Sure you want to Exit?"):
            self.window.destroy()
            self.window_threshold.destroy()
            del self.vid
    
    # Get gabor features for the snap image
    def get_gabor_feature(self,image):    
        classify = np.array([])
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
        return classify   
     
    # Classify snap image when the button is pressed
    def classify_snap(self):
        selection = self.var.get()
        select = self.v.get()
        if selection == 0:                
            if select == 0 or select == 2:
                self.model = joblib.load("./now_output/finalsvmgabor.joblib")
                
                #Some pre processing
                img = cv2.imread("./camera_image_output/test_image_gabor.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gb = self.get_gabor_feature(img)
                gb_reshape = gb.reshape((1,40))
                print(self.model.predict(gb_reshape))
                
            elif select == 1 or select == 3:
                model = joblib.load("kmeans.joblib")
                #read image
                image = cv2.imread("./camera_image_output/test_image_sift.png",1)
                image = cv2.resize(image,(200,200))
                #Convert them to grayscale
                image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #SIFT extraction
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(image,None)
                #classification of all descriptors in the model
                predict_kmeans=model.predict(descriptors)
                #calculates the histogram
                hist, bin_edges=np.histogram(predict_kmeans, bins=260)
                #histogram is the feature vector
                self.model = joblib.load("./now_output/sift_svm.joblib")

                hist = hist.reshape(1, -1)
                print(self.model.predict(hist))
            
        elif selection == 1: 
                
            if select == 0 or select == 2:
                self.model = joblib.load("./now_output/finalmlpgabor.joblib")
                #Some pre processing
                img = cv2.imread("./camera_image_output/test_image_gabor.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gb = self.get_gabor_feature(img)
                gb_reshape = gb.reshape((1,40))
                print(self.model.predict(gb_reshape))
            elif select == 1 or select == 3:
                model = joblib.load("kmeans.joblib")
                #read image
                image = cv2.imread("./camera_image_output/test_image_sift.png",1)
                image = cv2.resize(image,(200,200))
                #Convert them to grayscale
                image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #SIFT extraction
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(image,None)
                #classification of all descriptors in the model
                predict_kmeans=model.predict(descriptors)
                #calculates the histogram
                hist, bin_edges=np.histogram(predict_kmeans, bins=260)
                #histogram is the feature vector
                self.model = joblib.load("./now_output/sift_mlp.joblib")

                hist = hist.reshape(1, -1)
                print(self.model.predict(hist))
        
        elif selection == 2:
            if select == 0 or select == 2:
                self.model = joblib.load("finalsvmgabor.joblib")
                #Some pre processing
                img = cv2.imread("./camera_image_output/test_image_gabor.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gb = self.get_gabor_feature(img)
                gb_reshape = gb.reshape((1,40))
                print(self.model.predict(gb_reshape))
            
            elif select == 1 or select == 3:
                model = joblib.load("kmeans.joblib")
                #read image
                image = cv2.imread("./camera_image_output/test_image_sift.png",1)
                image = cv2.resize(image,(200,200))
                #Convert them to grayscale
                image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #SIFT extraction
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(image,None)
                #classification of all descriptors in the model
                predict_kmeans=model.predict(descriptors)
                #calculates the histogram
                hist, bin_edges=np.histogram(predict_kmeans, bins=260)
                #histogram is the feature vector
                self.model = joblib.load("finalsvmsift.joblib")

                hist = hist.reshape(1, -1)
                print(self.model.predict(hist))
                
                
                
        elif selection == 3:
            if select == 0 or select == 2:
                self.model = joblib.load("gabormlp1.joblib")
                #Some pre processing
                img = cv2.imread("./camera_image_output/test_image_gabor.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gb = self.get_gabor_feature(img)
                gb_reshape = gb.reshape((1,40))
                print(self.model.predict(gb_reshape))
            
            elif select == 1 or select == 3:
                model = joblib.load("kmeans.joblib")
                #read image
                image = cv2.imread("./camera_image_output/test_image_sift.png",1)
                image = cv2.resize(image,(200,200))
                #Convert them to grayscale
                image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #SIFT extraction
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(image,None)
                #classification of all descriptors in the model
                predict_kmeans=model.predict(descriptors)
                #calculates the histogram
                hist, bin_edges=np.histogram(predict_kmeans, bins=260)
                #histogram is the feature vector
                self.model = joblib.load("sipiup.joblib")

                hist = hist.reshape(1, -1)
                print(self.model.predict(hist))
                
                
# This class has functions to help with opening an Opencv videocapture in a Tkinter window     
class video_capture_opencv_tkinter:
    def __init__(self):
        # The Opencv function to start the webcam
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to start the video capture")
        # Get the height and width of the frame
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # This is the ROI (Region of Intrest) which is shown on video capture
        self.top, self.right, self.bottom, self.left = 10, 150, 310, 450

    # Function to get each frame from the video
    def get_frame(self):
        if self.vid.isOpened():
            # Read the frame from the video and store it so now we deal with an image
            ret, frame = self.vid.read()
            
            frame = cv2.flip(frame, 1)
            # Two bounding boxes for the same region of intrest
            roi = frame[self.top:self.bottom, self.right:self.left]
            roi2 = frame[self.top:self.bottom, self.right:self.left]
            # Make a rectagle for the region of intrest ROI
            cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0,255,0), 2)
            
            # Convert the image into HSV (RGB to HSV conversion)
            img_HSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Adding a medianBlur filter to reduce the noise and smoothen the image
            img_HSV = cv2.medianBlur(img_HSV,3)
            # Set the values for skin color for HSV
            HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
            # Use morphological operation (opening; erosion then dialation)
            HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            roi = cv2.bitwise_not(HSV_mask)
            # Convert the image into binary image
            et,thresh1 = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            if ret:
                # Return two frames for two windows (Note: Third value is also returned however its just a Grayscle value which is no longer used)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),et,thresh1,cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY))
            else:
                return (ret, None,None,None)
        else:
            return (ret, None,None,None)
    # Rekease the video when this object is deleted
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

     

              
if __name__ == "__main__":
    app = main_controller_class()
    app.generateUI()
