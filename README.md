# Innovating Ear Infection Diagnostics

## Problem
Acute otitis media (AOM), or ear infection, is the most common childhood infections. It accounts for 60% of antibiotic prescriptions in children, making it the main reason for the use of antibiotics in children. About 60% of cases of ear infection are caused by bacteria, 40% are viral or a combination of both. Only severe bacterial ear infection requires antibiotics. Visual examination through an otoscope is the primary but inaccurate method for the diagnosis of bacterial ear infection. The inaccuracy in diagnosis leads to two main consequences: 1) children may experience the buildup of ear pain and discomfort, which will result in prolong treatment time; 2) based on a survey that asked pediatricians whether or not they would prescribe antibiotics based on different patient profiles, 45% of pediatricians prophylactically administer antibiotics. It is estimated that 50% of antibiotic prescriptions for ear infections are unnecessary and potentially lead to a weaker immune system, expose the child to subsequent episodes of ear infection, and further increase the likelihood of antibiotic resistance. The goal is to come up with a rapid, effective, and non-invasive solution that will improve the accuracy of diagnosis of bacterial ear infection in order to decrease the unnecessary use of antibiotics and potentially the treatment time.

## How does it work?
The Raspberry Pi captures a picture of the eardrum with a spy camera. Then, it runs the image through a image classification model, and make a diagnosis about whether you have ear infection or not. 


## Creation of the image classification model
Data collection: An image library was created from online databases consisting of eardrum images of different severity degrees and of normal condition. These images are labeled as ear infection or normal by medical professionals. In addition to online databases, images were also collected from researchers and specialists and they were confirmed to be eardrums with middle ear infection. These images are stored in the "IMAGES" directory, but this directory is not visible to the public for confidentiality reasons.

Build the image classification model: A pre-trained deep neural network -- Inception-V3 from TensorFlow -- was used to extract features from images in the "IMAGES" directory. A linear SVM from scikit-learn was trained on these features (alongside with the image labels) to classify the eardrum images. The outcome of the performance testing shows that the accuracy of the prototype (91%) is much higher compared to that of medical residents (61%) and otolaryngologists (77%). This model is available for the public.

## Descripton of python files

create_img_classification_model.py: The first part contains all the work for the creation of the image classification model. The second part performs several important statistical tests and performance tests to validate the effectiveness of the model, i.e. verify that we are diagnosing ear infection accuratly and rapidly. You can train/create your own ear infection diagnosis model with this file.

capture_image_and_display_results_in_raspberry.py: It acquires image from raspberry pi camera and posts the data to the url which is produced by running the command "ngrok http 5000" in the terminal (note the command should be run in the same directory as ngrok.exe). After posting the data, it asks the server to perform image classification -- during which the script perform_classification_in_server.py is run -- and prints out the response received from the server. 

setup_local_server.py: It entails how the local server at localhost:5000 is setup. It takes a post request from localhost:5000 (where localhost:5000 is rerouted to a new url through ngrok). The post request contains the image data, which is then saved onto the local server and classfied as 'normal' or 'infect'. Lastly, this result is returned to the client.

perform_classification_in_server.py: The server receives the eardrum image from raspberry, then classifies this image using the classification model we trained and saved (which is named CNN_clf). The server sends the results back to Raspberry.

## How to set up and run
The server (a computer) contains:
	tensorflow_inception_graph.pd
	CNN_clf
	post_test folder (for saving the image received from the raspberry)
and
	setup_local_server.py

The raspberry pi contains:
	capture_image_and_display_results_in_raspberry.py

The process of setting up everything:
	in the server
		run setup_local_server.py
		run 'ngrok http 5000' in the terminal and copy the url down
	in the raspberry pi
		update the url in capture_image_and_display_results_in_raspberry.py
		run capture_image_and_display_results_in_raspberry
		click the button to collect image and you will see the results on the monitor and the LEDs