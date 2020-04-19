# azure-cv-python-integration
To integrate a Microsoft Azure Custom Vision project in a Python program.   
This is motivated by the time-consuming tag creation and image uploading tasks through the GUI provided by Microsoft Azure Custom Vision project.

## Prerequisites
The below Python modules are required :
1. Python built-in modules: time, os
2. Python libraries for data science: pandas
3. Python libraries for data visualization:  matplotlib.pyplot, seaborn 
4. Python libraries for image processing: PIL, cv2 (from openCV)
5. Other Python libraries:
import PySimpleGUI as sg
6. Microsoft Azure Custom Vision APIs for Python:  
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient  
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry  
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

## Python files explained:
1. azurecv_integration.py - the main python program (python azurecv_integration.py)
2. cv_plot.py - all the Computer Vision plots used in the application (import cv_plot as cplt)

## Dataset
1. 'trainLabels_bird.csv' - the csv file contains the labels for image file, 2 columns: 'filename' and 'label'
2. training dataset - all image files for training shall be put under a directory, each file shall have an entry 'trainLabels_bird.csv'
 
## The python program explained:
1) To execute:  
a) Run "python azurecv_integration.py"  
b) Below menu will be displayed:

============================================   
 Main Menu

 1) Load data
 2) Create tags
 3) Upload images
 4) Train and publish
 5) Training performance
 6) Predict class
 7) Delete tags  
Please enter your choice (1-7, 'q' to quit):  
============================================  

c) Enter '1' to '7' to execute the choice 

2) Instructions:  
a) Option 1) - 4) are interdependent and has to be executed in the sequence:  
	- Option 1) is to read a text file consists of filenames and labels to a dataframe and generate a dictionary 
	- Option 2) is to create tags in Azure based on the input files, only tags with more than five images will be created 
	- Option 3) is to upload images from local disk to Azure with respective tagging  
	- Option 4) is to train and publish the model in Azure  
	- The input file in 1) and images in 3) has to be matching   
b) Option 5) & 6) can be executed independently (without 1) - 4)) ; they are executed on the *latest* iteration of Azure project: 
	- Option 5) is to display the training performance of the Azure model  
	- Option 6) is to make prediction with a user input image file with a simple GUI popped up to allow user to select a image file from the local disk   
c) Option 7) is for deleting tags recorded in a text file. 

## Test model
- select '6' for '6) Predict class'
- a simple GUI popped up to allow user to select a image file from the local disk
- the result will be a bar chart showing the top-5 probablities of the image classification
