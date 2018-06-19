
# coding: utf-8

# In[1]:


# Genre Classification - Marc Jones

# imports
import os, itertools, sys, time, json, pandas, warnings, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from utils import flatten_json_dicts, flatten_json_lists, remove_non_numerical_data
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')
if not os.path.isfile('genre_df.csv'):
    sys.exit('ERROR: Cannot find the \'genre_df.csv\' file. The features from included audio must first be extracted. Run the scikit version of this classifier first')
    
print('Imports successful')


# In[2]:


# Organize Data
''' We'll use a data frame object (from the pandas library) to store all the feature files 
    (where rows represent *one song analyzed* and columns represent the individual features. 
    In order for the data frame to accurately model a each feature vector, I've chosen to 
    flatten the dictionary/json read feature data such that there are no nested dictionaries 
    or lists of data. Each nested dictionary becomes its own dictionary key while each datapoint 
    found in a list also becomes its own dictionary key based on sequential ordering. This process 
    converts the individual json files into one combined data structure that closely resembles a 
    csv file, which we can then preview the first few elements in-line. '''


if os.path.isfile('genre_df.csv'):
    df = pandas.read_csv('genre_df.csv')
    # read_csv automatically adds row nums but they're already saved to file so we must remove the duplicates
    df = df.drop('Unnamed: 0',1)
else:
    # use a pandas dataframe to hold all the data
    df = pandas.DataFrame(flat, index=[0])
    # load and flatten json files then add them to our data frame
    for json_file in json_filepaths:
        data = json.load(open(json_file), strict=False)
        # add genre annotation to our data read from the json file
        data['genre'] = data['genre'][0]
        # remove the metadata field as we wont be using it for classification
        data.pop('metadata',None)
        # flatten the nested dictionaries features into one
        almostflat = flatten_json_dicts(data)
        # flatten the nested lists of features into sequential features 
        flat = flatten_json_lists(flatten_json_lists(almostflat)) # needs two runs for the mfcc covariance stats
        flat = remove_non_numerical_data(flat)
        df = df.append(flat,ignore_index=True)
    # remove all rhythm_beats_pos columns (varied sequential length = a lot of NaN values)
    # store altered data frame into new object 
    for col in df.columns:
        if col.startswith('rhythm_beats') and col.endswith('_position'):
            df = df.drop(col,1) 
    # save dataframe to csv 
    df.to_csv('genre_df.csv')
# unwated columns removed in line with 'Gaia' standard.
unwanted_columns = ['_dmean', '_dvar', '_min', '_max', 'cov', 'tonal.hpcp', 'lowlevel_silence_rate']
for col in df.columns:
    if any([unwanted in col for unwanted in unwanted_columns]):
        df = df.drop(col,1)
        
print('Data loaded into memory')


# In[3]:


# Preprocess Data
''' Before classification the data must be standardized to reduce its overal range. 
    This is accomplished using scikit's standard scalar method which removes the mean, 
    scales to unit variance, ultimately transforming the data to appear Gaussian. '''

# create feature vectors
classes = {'cla':0,'dan':1,'hip':2,'jaz':3,'pop':4,'rhy':5,'roc':6,'spe':7}
feature_vectors = df.drop('genre',1) # seperate the labels from the features
labels = df['genre'] # store labels seperately
labels = np.array([classes[i] for i in labels]) # convert to numeric values

# standardize feature values 
scaler = StandardScaler()
scaled_feature_vectors = scaler.fit_transform(feature_vectors)

# reduce data dimensionality w/ PCA (max data variation) & LDA (max interclass variation)
pca = PCA(n_components=5)
pca_reduced_feature_vectors = pca.fit(scaled_feature_vectors).transform(scaled_feature_vectors)

lda = LinearDiscriminantAnalysis(n_components=5)
lda_reduced_feature_vectors = lda.fit(scaled_feature_vectors, labels).transform(scaled_feature_vectors)

print('Data preprocessed using normalization then reduced with PCA and LDA')


# In[4]:


# SVM CLASSIFICATION USING PCA
# create test and training subsets of the data
features_train, features_test, pred_train, pred_test = train_test_split(
    pca_reduced_feature_vectors, labels, test_size=0.2)

# train the classifier
classifier = LinearSVC()
# classifier = SVC()
classifier.fit(features_train, pred_train)

cci = classifier.score(features_test,pred_test)
predictions = classifier.predict(features_test)

print('SVM CLASSIFICATION USING PCA')
print('\n'+'Correct Classified Instances:',cci)

cm = confusion_matrix(pred_test,predictions)
print('Confusion Matrix\n(y-true : x-predicted)\n')
print(pandas.DataFrame(cm))
print('\nLabel Key:')
print(classes)

class_report = classification_report(pred_test,predictions,target_names=classes)
print('\n'+class_report)


# In[5]:


# SVM CLASSIFICATION USING LDA
# create test and training subsets of the data
features_train, features_test, pred_train, pred_test = train_test_split(
    lda_reduced_feature_vectors, labels, test_size=0.2)

# train the classifier
classifier = LinearSVC()
# classifier = SVC()
classifier.fit(features_train, pred_train)

cci = classifier.score(features_test,pred_test)
predictions = classifier.predict(features_test)

print('SVM CLASSIFICATION USING LDA')
print('\n'+'Correct Classified Instances:',cci)

cm = confusion_matrix(pred_test,predictions)
print('Confusion Matrix\n(y-true : x-predicted)\n')
print(pandas.DataFrame(cm))
print('\nLabel Key:')
print(classes)

class_report = classification_report(pred_test,predictions,target_names=classes)
print('\n'+class_report)


# In[6]:


# NN CLASSIFICATION USING PCA
# create test and training subsets of the data
features_train, features_test, pred_train, pred_test = train_test_split(
    pca_reduced_feature_vectors, labels, test_size=0.2)

print('SVM CLASSIFICATION USING PCA4')

# simple multi-layer perceptron architecture with 16 hidden units
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=5))
model.add(Dense(8, activation='softmax'))

# configure the model for training
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# one-hot encoding
onehot_labels = to_categorical(pred_train, num_classes=None)
onehot_pred_test = to_categorical(pred_test, num_classes=None)

model.fit(features_train, onehot_labels, epochs=200, batch_size=None, verbose=2)

scores = model.evaluate(x=features_test, y=onehot_pred_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
# print(score)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


# NN CLASSIFICATION USING LDA
# create test and training subsets of the data
features_train, features_test, pred_train, pred_test = train_test_split(
    lda_reduced_feature_vectors, labels, test_size=0.2)

print('NN CLASSIFICATION USING LDA')

# simple multi-layer perceptron architecture with 16 hidden units
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=5))
model.add(Dense(8, activation='softmax'))

# configure the model for training
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# one-hot encoding
onehot_labels = to_categorical(pred_train, num_classes=None)
onehot_pred_test = to_categorical(pred_test, num_classes=None)

model.fit(features_train, onehot_labels, epochs=200, batch_size=None, verbose=2)

scores = model.evaluate(x=features_test, y=onehot_pred_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
# print(score)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Training and evaluation complete')