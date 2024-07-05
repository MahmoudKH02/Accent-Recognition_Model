#!/usr/bin/env python
# coding: utf-8

# In[188]:


import librosa
import numpy as np
import pandas as pd
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay


# # Loading Data

# In[189]:


# just a sample of what the data is like.
ipd.Audio('./training_set/Hebron/hebron_test021.wav')


# ## Loading the Speach and Sampling Rates
# The speach and its sampling rate will be extracted using the librosa library with the help of the `librosa.load()` function

# In[190]:


sampling_rate = 8000

def load_directory(directory):
    subdirectories = ["Hebron", "Nablus", "Jerusalem", "RamallahReef"]
    audio_list = []
    labels = []
    rates = []
    
    # Iterate through each subdirectory
    for subdir in subdirectories:
        # Get the full path of the subdirectory
        subdir_path = os.path.join(directory, subdir)

        # Iterate through each file in the subdirectory
        for filename in os.listdir(subdir_path):
            # Get the full path of the file
            file_path = os.path.join(subdir_path, filename)

            # Check if the path is a file
            if os.path.isfile(file_path):
                # Load the audio file
                audio, sr = librosa.load(file_path, sr=sampling_rate)
                
                # Normalize the audio to have a maximum absolute value of 1.0
                audio = librosa.util.normalize(audio)
                
                # Append audio and label
                audio_list.append(audio)
                labels.append(subdir)
                rates.append(sr)
    
    return audio_list, labels, rates


# In[191]:


training_directory = './training_set'
testing_directory = './testing_set'

training_voices, Y_train, train_sr = load_directory(training_directory)
testing_voices, Y_test, test_sr = load_directory(testing_directory)


# In[192]:


# Ensure all voices have the same Sampling Rate
def check_rates(train_rates, test_rates):
    unique_train_rates = set(train_rates)
    unique_test_rates = set(test_rates)
    
    print(f'Training Set Sampling Rates: {unique_train_rates} Hz')
    print('------')
    print(f'Testing Set Sampling Rates: {unique_test_rates} Hz')
    
    
check_rates(train_sr, test_sr)


# In[193]:


# Sample Sound to check if reading was correct.
plt.figure()
plt.plot(training_voices[0])

plt.title(f'Sample Speach Signal of {Y_train[0]}')
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.show()


# # Data Preprocessing
# ## 1. Extract MFCCs and Delta Features
# The MFCC features will be extracted along with the delta and delta-delta features using the `librosa.mfcc()`, and `librosa.delta()` functions.
# 
#  The `librosa.feature.mfcc()`:
# * This function takes in the frame length, and in this project, the frame size was chosen to be of length 20ms.
# * it also takes the hop_lenth, which is the percetange of overlap between each frame, and this is choesen to be 50%.
# * lastly, the type of windowing to be applied, in which a *hamming window* was chosen in this case.
# 

# In[194]:


def get_frame_size(sampling_rate=sampling_rate, frame_duration=20): # 20 ms default frame_duration
    return int(sampling_rate * (frame_duration / 1000))


# In[195]:


def extract_features(audio, frame_size, sr=sampling_rate, n_mfcc=13):

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        hop_length=int(frame_size // 2),
        win_length=frame_size,
        window=scipy.signal.windows.hamming
    )
    delta = librosa.feature.delta(mfccs)
    delta_2 = librosa.feature.delta(mfccs, order=2)
    
    # Compute mean for each feature
    mfccs_mean = np.mean(mfccs, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta_2, axis=1)
    
    return np.hstack([mfccs_mean, delta_mean, delta2_mean])


def process_voices(audio_voices, frame_size, sr=sampling_rate, n_mfcc=13):
    features = []
    
    for voice in audio_voices:
        feature = extract_features(voice, frame_size, sr, n_mfcc)
        features.append(feature)
    
    return np.array(features)


# In[196]:


X_train_unscaled = process_voices(training_voices, get_frame_size())
X_test_unscaled = process_voices(testing_voices, get_frame_size())


# In[197]:


print(f'Train: {X_train_unscaled.shape}, Test: {X_test_unscaled.shape}')

print('--------')
print('Example Speach (No.5) MFCC, and Delta Features:\n')
print(X_train_unscaled[5])
print('--------')
print('Label:', Y_train[5])


# ## 2. Scaling Data
# Since the delta, and delta-delta features are extracted alongisde the MFCCs, then scaling is important since each vector has a different scale range.
# 
# **Mean-variance normalization** was applied using the `StandardScaler()` function. This will normalize data points so that all points have a mean of *zero*, and a variance of *one*.

# In[198]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.fit_transform(X_test_unscaled)


# In[199]:


X_train[5]


# # Training Phase
# ## 1. Supervised Models Training`

# ### Train Simple Models (Before Hyper-parameter tuning)

# In[200]:


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)

# Example with Random Forest
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(X_train, Y_train)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, Y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)
y_pred_random_forest = rfc_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
print('SVM Report:')
print(classification_report(Y_test, y_pred_svm))

print('Random Forest Report:')
print(classification_report(Y_test, y_pred_random_forest))

print('KNN Report:')
print(classification_report(Y_test, y_pred_knn))


# ### Hyper-parameter Tuning

# In[155]:


def tune_model(model, params: dict, cv=5, verbose=2):
    mdoel_grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
    mdoel_grid_search.fit(X_train, Y_train)
    
    return mdoel_grid_search

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
rfc_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
}


best_svm_model = tune_model(SVC(), svm_param_grid)
best_rfc_model = tune_model(RandomForestClassifier(random_state=42), rfc_param_grid)
best_knn_model = tune_model(KNeighborsClassifier(), knn_param_grid)

# SVM
print(f"Best SVM parameters: {best_svm_model.best_params_}")
print(f"Best Score", best_svm_model.best_score_)

# RFC
print(f"Best Random Forest parameters: {best_rfc_model.best_params_}")
print(f"Best Score", best_rfc_model.best_score_)

# KNN
print(f"Best KNN parameters: {best_knn_model.best_params_}")
print(f"Best Score", best_knn_model.best_score_)


# ### SVM Confusion Matrix (best model)

# In[156]:


disp = ConfusionMatrixDisplay.from_estimator(
    svm_model,
    X_test,
    Y_test,
    cmap=plt.cm.Blues
)
plt.title('SVM Model Confusion Matrix')
plt.show()


# ### KNN Confusion Matrix (worst model)

# In[157]:


disp = ConfusionMatrixDisplay.from_estimator(
    knn_model,
    X_test,
    Y_test,
    cmap=plt.cm.Blues
)
plt.title('KNN Model Confusion Matrix')
plt.show()

