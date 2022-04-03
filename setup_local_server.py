#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Wanying Li
# Created Date: May 15, 2017
# ---------------------------------------------------------------------------
"""
It entails how the local server at localhost:5000 is set up.
It takes a post request from localhost:5000 (where localhost:5000 is rerouted to a new URL through ngrok).
The post request contains the image data, which is then saved onto the local server and classified as 'normal' or 'infect.'
Lastly, this result is returned to the client.
""" 
# ---------------------------------------------------------------------------


import os, os.path
from flask import Flask, flash, request, redirect, url_for, jsonify
import pickle
import numpy as np
import sklearn
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile


def create_graph(model_path):
    """
    The create_graph function loads the inception model to memory. This function should be called before
    calling extract_features or extract_features_single_img.
    
    Input:
        model_path = path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features_single_img(image_path, verbose=False):
    """
    The extract_features_single_img function computes the inception bottleneck feature for one single image.
    
    Input:
        image_path = directory path of the image
    Output: 
        feature = 2-d np array in the shape of (1, 2048)
    """
    feature_dimension = 2048
    feature = np.empty((1, feature_dimension))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
 
        if verbose:
            print('Processing %s...' % (image_path))

        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image)

        image_data = gfile.FastGFile(image_path,'rb').read()
        feature_tmp = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
        feature[:] = np.squeeze(feature_tmp)

    return feature


# create a graph from the Inception V3.0 model
model_path = '/innovating_ear_infection_diagnostics/inception_dec_2015/tensorflow_inception_graph.pb'
create_graph(model_path)

# load classifier
clf = pickle.load(open('CNN_clf_binary','rb'))

# setup webserver
UPLOAD_FOLDER = '/innovating_ear_infection_diagnostics/post_test'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/predict', methods=['POST'])
def predict():
    # upload user's photo to webserver and save the photo to the UPLOAD_FOLDER on the webserver
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER,'data.JPG')
    file.save(filepath)

    # extraction feature of an user input image
    img_input = filepath
    print(img_input)
    img_feature = extract_features_single_img(img_input)

    # classification
    y_pred = clf.predict(img_feature)
    output = np.array2string(y_pred[0])
    print(output)
    
    return output

if __name__ == "__main__":
    app.run()