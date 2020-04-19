"""
Reference: 
1. https://docs.microsoft.com/en-sg/azure/cognitive-services/custom-vision-service/python-tutorial
2. https://docs.microsoft.com/en-us/python/api/overview/azure/cognitiveservices/customvision?view=azure-python
"""
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

import cv_plot as cplt
import time, os
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

def read_file(data_file):    
    """ To read the data file to a pandas dataframe """
    
    print("Reading data from data file %s..." %(data_file))
    
    #idx_col = idx_col
    #label_col = label_col
    
    df = pd.read_csv(data_file, index_col=idx_col)
    df.sort_values(label_col, inplace = True)
    #print("Index:", df.index)
    #print("Columns:", df.columns)
    samples_counts = df[label_col].count()
    print("There are %i samples." %samples_counts)
    classes = df[label_col].unique()
    print("There are %i unique classes:" %(len(classes)))
    list_classes = list(classes)
    print(list_classes)
    
    cplt.plot_hist_df(df[label_col], 'Bird class', list_classes)
    
    df_bird = df.copy()
    classes = list_classes
     
    print("File reading to dataframe completed.")
    
    return df_bird, classes
    
def load_img(img_dir, df_bird, classes):
    """ To load the image data to a dictionary """
    
    print("Loading images from directory: %s..." %img_dir)
    df_y = df_bird
    #print("df_y:", df_y, type(df_y))
    
    X, y = [], []
    dict_img = {label: [] for label in classes} #initialise dictionary for images per class
    #to open image files from the directory
    for root, dirs, files in os.walk(img_dir): 
        for file in files:
            img_file = img_dir + '/' + file
            #print("Image:", img_file)
            Image.open(img_file)
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (299, 299))
            #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            #plt.show()
            X.append(img)
            y_idx = file.split('.')[0]  #get the filename without extension (.jpg)
            y_col = df_y.loc[y_idx, label_col]
            dict_img[y_col].append(img_file)
            #print("y_col:", y_col)
            y.append(y_col)
    
    dict_images = dict_img
    print("Image data loading completed.")
    
    return dict_images

def load_data():
    """ To load the image data for Azure CV model training """
    
    print("Loading data from ...", img_csv)
    
    global dict_data
    df_bird, b_classess = read_file(img_csv)
    
    #print("df_bird:", df_bird)
    dict_images = load_img(img_dir, df_bird, b_classess)
    dict_data = dict_images
    
    return None

def create_tags():
    """ To create tags on Azure CV services """
    
    print("Creating tags ...")
    global dict_tag_ids
    global list_tags
    
    dict_tag_ids = {}
    list_tags = []
    
    try:
        tag_count = 0
        for k in dict_data:
            if len(dict_data[k]) >= 5:
                print("Creating tag name:", k)
                tag = k
                tag_obj = trainer.create_tag(project_id, tag)  #azure api
                dict_tag_ids[k] = tag_obj.id
                list_tags.append(k)
                tag_count += 1
        
        #write tag_ids to a text file
        with open(tag_file, 'w') as filehandle:
            for k in dict_tag_ids:
                filehandle.write('%s,%s\n' %(k, dict_tag_ids[k]))
        
        print(tag_count, "tags created.")
    except Exception as e:
        print("Error:", e)
        
    return None

def delete_tags():
    """ To delete tags from Azure CV services """
    
    print("Deleting tags ...")
    
    try:
        tag_count = 0
        #read tag_ids from the text file
        with open(tag_file, 'r') as filehandle:
            for line in filehandle:
                tag = line.split(',')[1]
                tag_id = tag.replace("\n", "")
                print("Deleting tag id:", tag_id)
                trainer.delete_tag(project_id, tag_id)  #azure api
                tag_count += 1
    
        print(tag_count, "tags deleted.")
    except Exception as e:
        print("Error:", e)
    
    return None

def upload_images():
    """ To upload images to Azure CV services """
    
    print("Uploading images ...")
    
    img_count = 0
    for tag in list_tags:
        print("Tag name:", tag)
        #print("Tag id:", dict_tag_ids[tag])
        print("dict_data:", dict_data[tag])
    
        image_list = []
        files = dict_data[tag]
        for file in files:
            file_name = file
            #print("Filename:", file_name)
            with open(file_name, "rb") as image_contents:
                #print(image_contents.read())
                image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[dict_tag_ids[tag]]))    #azure api
            img_count += 1
        
        print("Image_list:", image_list)
        
        upload_result = trainer.create_images_from_files(project_id, images=image_list)  #azure api
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            exit(-1)

    print(img_count, "images uploaded.")
    return None

def train_publish():
    """ To train and publish the Azure CV model """
    
    print("Training and publishing the model ...")
    
    try:
        iteration = trainer.train_project(project_id)  #azure api
        while (iteration.status != "Completed"):
            iteration = trainer.get_iteration(project_id, iteration.id)
            print ("Training status: " + iteration.status)
            time.sleep(2)
        
        # The iteration is now trained. Publish it to the project endpoint
        trainer.publish_iteration(project_id, iteration.id, publish_iteration_name, prediction_resource_id)  #azure api
        
        print("Publish iteration name: %s. Iteration id: %s" %(publish_iteration_name, iteration.id))
    except Exception as e:
        print("Error:", e)
    
    print("Training and publishing completed.")

def get_latest_iter():
    """ To get the latest iteration of the Azure CV model """
    
    print("Getting the latest iteration ...")
    
    list_iter_id = []
    for iter in trainer.get_iterations(project_id):
        list_iter_id.append(iter.id)
        #print("Iteration id: %s" %(iter.id))
    
    latest_iter_id = list_iter_id[0]
    iter_name = trainer.get_iteration(project_id, latest_iter_id)  #azure api
    latest_iter_name = iter_name.publish_name
    
    return latest_iter_id, latest_iter_name

def disp_performance():
    """ To display performance of the latest iteration of the Azure model """
    
    print("Displaying Azure model performance ...")
    
    latest_iter_id, latest_iter_name = get_latest_iter()
    iter_perf = trainer.get_iteration_performance(project_id, latest_iter_id)  #azure api
    print("Performance for iteration: %s (id:%s)" %(latest_iter_name, latest_iter_id))
    
    dict_tags = {}
    list_tag_name = []
    for tag in iter_perf.per_tag_performance:
        scores = {}
        #print(tag.id, tag.name)
        #print(tag.precision, tag.recall, tag.average_precision)
        list_tag_name.append(tag.name)
        scores['precision'] = tag.precision
        scores['recall'] = tag.recall
        scores['average_precision'] = tag.average_precision
        dict_tags[tag.name] = scores
    cplt.plot_prec_rec(dict_tags, list_tag_name, [])
    
    #visualize the performance
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    cplt.plot_donut(iter_perf.precision, 'Precision', (1,3),(0,0))
    cplt.plot_donut(iter_perf.recall, 'Recall', (1,3),(0,1))
    cplt.plot_donut(iter_perf.average_precision, 'AveragePrecision', (1,3),(0,2))
    plt.show()

def predict_class():
    """ To make prediction of an image """
    
    print("Predicting the bird class...")
    
    # Now there is a trained endpoint that can be used to make a prediction
    predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)
    
    fig = plt.figure(figsize=(12,8))
    with plt.style.context('seaborn-whitegrid'):
        ax2 = fig.add_subplot(122)
    ax1 = fig.add_subplot(121)
    plt.rcParams.update({'font.size': 16})
 
    img_file = sg.PopupGetFile('Please enter a file name')
    print("Image file:", img_file)
    img = Image.open(img_file).resize((499, 499), Image.LANCZOS)
    ax1.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    ax1.set_xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    
    latest_iter_id, latest_iter_name = get_latest_iter()
    print("Prediction done on the latest iteration: %s (id:%s)" %(latest_iter_name, latest_iter_id))
    with open(img_file, "rb") as image_contents:  #local images
        results = predictor.classify_image(
            project_id, latest_iter_name, image_contents.read())  #azure api
    
    #display the top 5 probabilities
    list_tag, list_prob = [], []
    for i, prediction in zip(range(5), results.predictions):
        if i < 5:
            list_tag.append(prediction.tag_name)
            list_prob.append(round(prediction.probability, 2))
            #print(list_tag[i], list_prob[i])
        else:
            break
    
    cplt.plot_bar_prob(list_tag, list_prob, 'Bird classification', 'Top 5', ax2)
    plt.show()

    return None

def ask_choice():
    print("Main Menu\n")
    print("1) Load data")
    print("2) Create tags")
    print("3) Upload images")
    print("4) Train and publish")
    print("5) Training performance")
    print("6) Predict class")
    print("7) Delete tags")
    inChoice = input("Please enter your choice (1-7, \'q\' to quit): ")

    return inChoice

def select_func(choice):
    print("Your choice is", choice)
    dict_func = {
        '1': load_data,
        '2': create_tags,
        '3': upload_images,
        '4': train_publish,
        '5': disp_performance,
        '6': predict_class,
        '7': delete_tags
    }

    #get the function of the choice from the dictionary
    exec_func = dict_func.get(choice, lambda: print("Invalid choice, please try again..."))
    return exec_func

def main(): 
    is_quit = False
    while not is_quit:
        choice = ask_choice()
        if choice == 'q':
            print("Your choice is \'q\'.")
            is_quit = True
        else:
            select_func(choice)()  #execute the function of the choice


# global variables     
data_dir = 'datasets'
img_csv = '{}/trainLabels_bird.csv'.format(data_dir)   
img_dir = '{}/birds'.format(data_dir)
label_col = 'label'
idx_col = 'filename'

# Azure project: tipp-aai-cv-bird-v1
project_id = 'b5c03c9c-08fe-407c-898a-6dbc0b74e94c'
# Training key and endpoint
ENDPOINT = 'https://tipp-aai-cv-bird-v1.cognitiveservices.azure.com/'
training_key = '108a361166c94b3b82866e9cdf04ad03' #not required if only publish PREDICTION
# Initialise TrainingClient
trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)

# Prediction key and endpoint
PRED_ENDPOINT = 'https://tippaaicvbirdv1-prediction.cognitiveservices.azure.com/'
prediction_key = '71287f1951f340fe923a13399c6170e1'
prediction_resource_id = '/subscriptions/441baf38-1c7d-468c-9c74-47f6d8cd7e7d/resourceGroups/tipp-aai-cv/providers/Microsoft.CognitiveServices/accounts/tippaaicvbirdv1-Prediction'
publish_iteration_name = 'bird_iter8' 

# File to store tag-id 
tag_file = '{}/list_tag_ids.txt'.format(data_dir)

#execute the main() function            
if __name__ == '__main__': 
    main()