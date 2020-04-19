# azure-cv-python-integration
To integrate a Microsoft Azure Custom Vision project in a Python program.   
This is motivated by the time-consuming tag creation and image uploading tasks through the GUI provided by Microsoft Azure Custom Vision project.

## Prerequisites
**A) The Microsoft Azure project endpoints and points must be ready and accessible, they shall be defined as the global variables:**  
\# Azure project: tipp-aai-cv-bird-v1  
project_id = 'b5c03c9c-08fe-407c-898a-6dbc0b74e94c'  
\# Training key and endpoint  
ENDPOINT = 'https://tipp-aai-cv-bird-v1.cognitiveservices.azure.com/'  
training_key = '108a361166c94b3b82866e9cdf04ad03' #not required if only publish PREDICTION  
\# Initialise TrainingClient  
trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)  

\# Prediction key and endpoint  
PRED_ENDPOINT = 'https://tippaaicvbirdv1-prediction.cognitiveservices.azure.com/'  
prediction_key = '71287f1951f340fe923a13399c6170e1'  
prediction_resource_id = '/subscriptions/441baf38-1c7d-468c-9c74-47f6d8cd7e7d/resourceGroups/tipp-aai-cv/providers/Microsoft.CognitiveServices/accounts/tippaaicvbirdv1-Prediction'  
publish_iteration_name = 'bird_iter8'   

**B) The below Python modules are required:**
1. Python built-in modules: time, os
2. Python libraries for data science: pandas
3. Python libraries for data visualization:  matplotlib.pyplot, seaborn 
4. Python libraries for image processing: PIL, cv2 (from openCV)
5. Other Python libraries: PySimpleGUI (for a pop-up GUI asking for user input)
6. Microsoft Azure Custom Vision APIs for Python:  
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient  
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry  
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

## Python files explained:
1. azurecv_integration.py - the main python program (python azurecv_integration.py)
- the global variables shall be modified according to the training dataset   
\-------------------  
data_dir = 'datasets'  
img_csv = '{}/trainLabels_bird.csv'.format(data_dir)    
img_dir = '{}/birds'.format(data_dir)  
label_col = 'label'  
idx_col = 'filename'  
\-------------------

2. cv_plot.py - all the Computer Vision plots used in the application (import cv_plot as cplt)

## Dataset
1. 'trainLabels_bird.csv' - the csv file contains the labels for image file, 2 columns: 'filename' and 'label'
2. training dataset - all image files for training shall be put under a directory, each file shall have an entry 'trainLabels_bird.csv'
 
## The python program explained:
### To execute:  
a) Run "python azurecv_integration.py"  
b) Below menu will be displayed:

\============================================   
Main Menu

1) Load data
2) Create tags
3) Upload images
4) Train and publish
5) Training performance
6) Predict class
7) Delete tags  

Please enter your choice (1-7, 'q' to quit):  
\============================================  
c) Enter '1' to '7' to execute the choice 

### Instructions:  
a) Option 1) - 4) are interdependent and has to be executed in the sequence:    
- Option 1) is to read a text file consists of filenames and labels to a dataframe and generate a dictionary 
- Option 2) is to create tags in Azure based on the input files, only tags with more than five images will be created 
- Option 3) is to upload images from local disk to Azure with respective tagging  
- Option 4) is to train and publish the model in Azure  
- The input file in 1) and images in 3) has to be matching
	
b) Option 5) & 6) can be executed independently (without 1) - 4)) ; they are executed on the *latest* iteration of Azure project:   
- Option 5) is to display the training performance of the Azure model    
- Option 6) is to make prediction with a user input image file with a simple GUI popped up to allow user to select a image file from the local disk
	
c) Option 7) is for deleting tags recorded in a text file, by deleting the tag, all the images tagged with the id will be deleted as well. 

## Test model
- select '6' for '6) Predict class'
- a simple GUI popped up to allow user to select a image file from the local disk  
![gui-imgfile](/images/gui-imgfile.PNG)
- the result will be a bar chart showing the top-5 probablities of the image classification  
![chart-prob](/images/chart-prob.png)

## Reference
1. https://docs.microsoft.com/en-sg/azure/cognitive-services/custom-vision-service/python-tutorial
2. https://docs.microsoft.com/en-us/python/api/overview/azure/cognitiveservices/customvision?view=azure-python
