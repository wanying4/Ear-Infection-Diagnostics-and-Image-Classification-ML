# Innovating Ear Infection Diagnostics

## Background of the project
Acute otitis media (AOM), or ear infection, is the most common childhood infection. It accounts for 60% of antibiotic prescriptions in children, making it the main reason for using antibiotics in children. About 60% of cases of ear infections are caused by bacteria, 40% are viral or a combination of both. Only severe bacterial ear infection requires antibiotics. Visual examination through an otoscope is the primary but inaccurate method for diagnosing bacterial ear infections. The inaccuracy in diagnosis leads to two main consequences: 1) children may experience the buildup of ear pain and discomfort, which will result in prolonged treatment time; 2) based on a survey that asked pediatricians whether or not they would prescribe antibiotics based on different patient profiles, 45% of pediatricians prophylactically administer antibiotics. It is estimated that 50% of antibiotic prescriptions for ear infections are unnecessary and potentially lead to a weaker immune system, expose the child to subsequent episodes of ear infection, and further increase the likelihood of antibiotic resistance. This project aims to create a rapid, effective, and non-invasive solution that will improve the accuracy of diagnosis of bacterial ear infection in order to decrease the unnecessary use of antibiotics and potentially the treatment time.

## How does the project work?
The Raspberry Pi captures a picture of the eardrum with a spy camera. Then, it runs the image through an image classification model, and makes a diagnosis about whether you have an ear infection or not. 

## Creation of the image classification model
**Data collection**: An image library was created from online databases consisting of eardrum images of different severity degrees and of normal conditions. These images are labeled as ear infections or normal by medical professionals. In addition to online databases, images were also collected from researchers and specialists, and they were confirmed to be eardrums with middle ear infections. These images are stored in the "IMAGES" directory, but this directory is not visible to the public for confidentiality reasons.

**Build the image classification model**: A pre-trained deep neural network -- Inception-V3 from TensorFlow -- was used to extract features from images in the "IMAGES" directory. A linear SVM from scikit-learn was trained on these features (alongside the image labels) to classify the eardrum images. This model is available to the public.

## Model performance
The model has **91% accuracy**, which is much higher compared to that of medical residents (61%) and otolaryngologists (77%).
| Measurements | Results |
| :-------------: | :-------------: |
| Accuracy (Probability of detecion) | 91% |
| Sensitivity (True Positive Rate) | 86% |
| Specificity (True Negative Rate) | 91% |
| Precision (Positive predictive value) | 90% |  

In addition, the computational time was collected during the performance testing. The average was **3.7 seconds** with a standard deviation of 0.4 seconds. This means the doctors can rapidly obtain a result without interrupting their current routine for the diagnosis of bacterial ear infection.

## Description of the python files

`create_img_classification_model.py`: The first part contains all the work to create the image classification model. The second part performs several important statistical tests and performance tests to validate the effectiveness of the image classification model, i.e. verify that we are diagnosing ear infections accurately and rapidly. Note: you can train/create your own ear infection diagnosis model with this file by building your own image library.

`capture_image_and_display_results_in_raspberry.py`: It acquires an image from the Raspberry Pi camera and posts the data to the URL, which is produced by running the command "ngrok http 5000" in the terminal (note the command should be run in the same directory as ngrok.exe). After posting the data, it asks the server to perform image classification -- during which the script `perform_classification_in_server.py` is run -- and prints out the response received from the server. 

`setup_local_server.py`: It entails how the local server at localhost:5000 is set up. It takes a post request from localhost:5000 (where localhost:5000 is rerouted to a new URL through ngrok). The post request contains the image data, which is then saved onto the local server and classified as 'normal' or 'infect.' Lastly, this result is returned to the client.

`perform_classification_in_server.py`: The server receives the eardrum image from Raspberry, then classifies this image using the classification model we trained and saved (which is named CNN_clf). The server sends the results back to Raspberry.

## How to set up and run the image classification algorithm

### Save the following items to your local server (e.g. your computer)
- `setup_local_server.py`
- tensorflow_inception_graph.pd
- CNN_clf

### Create the following item(s) in your local server (e.g. your computer)
- post_test folder (for saving the image received from the Raspberry Pi)

### Save the following item(s) to Raspberry Pi
- `capture_image_and_display_results_in_raspberry.py`

### To run the setup code and algorithm
#### In the Local server:
1. Run `setup_local_server.py`
2. Run 'ngrok http 5000' in the terminal and copy the URL down
#### In the Raspberry Pi:
1. Update the URL in `capture_image_and_display_results_in_raspberry.py`
2. Run `capture_image_and_display_results_in_raspberry`
3. Collect image from Raspberry Pi, you will see the results on the monitor and the LEDs
