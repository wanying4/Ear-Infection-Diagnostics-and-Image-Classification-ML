# coding: utf-8

import pickle
import numpy as np
import sklearn
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile


######### Create a graph from the Inception V3.0 model#########
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
        

######### Extract features #########
def extract_features(image_paths, verbose=False):
    """
    The extract_features function computes the inception bottleneck feature for a list of images.
    Input:
        image_paths = array of image path
    Output:
        features = 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))
    
    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))
            
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image)
            
            image_data = gfile.FastGFile(image_path,'rb').read()
            feature = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
            features[i, :] = np.squeeze(feature)
    
    return features

image_dir = 'IMAGES/'
image_paths = [image_dir+f for f in os.listdir(image_dir) if re.search('jpg|JPG', f)]
features = extract_features(image_paths)

######### Extract labels from image filenames for binary classification #########
labels = []
for f in image_paths:
    if 'normal' in f:
        labels.append('normal')
    else:
        labels.append('infected')

######### Classfication and performance #########
# Prepare training and test datasets. We will use 80% of the data as the training set and 20% as the test set.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Classifying the images with a Linear Support Vector Machine (SVM)
clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr',class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))

labels_dis=sorted(list(set(labels)))
print("\nConfusion matrix:")
print("Labels: {0}\n".format(",".join(labels_dis)))
print(confusion_matrix(y_test, y_pred, labels=labels_dis))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# binarize the labels where normal = 0, infected = 1
y_pred = label_binarize(y_pred, classes=['normal','infected']) # y_score refers to y_pred
y_test = label_binarize(y_test, classes=['normal','infected'])

# Compute Precision-Recall and plot curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, lw=6, color='navy',label='Precision-Recall curve (area = {0:0.2f})'.format(average_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve of the Classification Algorithm\nUsed in the Final Prototype', fontsize=28)
plt.legend(loc="lower left")
plt.show()

######### Save classifier/model for local server #########
pickle.dump(clf, open('CNN_clf', 'wb'))
