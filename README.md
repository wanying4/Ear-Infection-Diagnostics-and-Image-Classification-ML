# Innovating Ear Infection Diagnostics

3 python files in total:

client_input.py: It acquires image from picamera and posts the data onto the defined url, where the url is produced from running the command "ngrok http 5000" in the terminal (note the command should be run in the same directory as ngrok.exe). After posting the data, it prints out the response from the server. 

local_server_setup.py: It entails how the local server at localhost:5000 is setup. It takes a post request from localhost:5000 (where localhost:5000 is rerouted to a new url through ngrok). The post request contains the image data, which is then saved onto the local server and classfied as 'normal' or 'infect'. Lastly, this result is returned to the client.

CNN_img_classification_train_model.ipynb: it uses a pre-trained model (inception v3), then trains the eardrum images where the image features are extracted using convolutional neural network (CNN). The features and labels are used to train a classfier using support vector machine (SVM). The classfier is then saved onto the server. 

The server (our computer) contains
	tensorflow_inception_graph.pd
	CNN_clf_binary
	post_test folder for saving image posted by the client
and
	local_server_setup.py

The raspberry pi contains:
	client_input.py

The process of seting up everything:
	in the server
		run local_sever.py
		run 'ngrok http 5000' in the terminal and get the url
	in the raspberry pi
		update the url in client_input.py and continue running client_input