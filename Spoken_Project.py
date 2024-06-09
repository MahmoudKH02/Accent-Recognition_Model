#!/usr/bin/env python
# coding: utf-8

# In[94]:


import librosa
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.model_selection import train_test_split, GridSearchCV


# # Loading Data

# In[95]:


sampling_rate = 16000

def load_directory(directory):
    subdirectories = ["Hebron", "Nablus", "Jerusalem", "RamallahReef"]
    audio_list = []
    labels = []
    
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
    
    return audio_list, labels


# In[96]:


training_directory = './training_set'
testing_directory = './testing_set'

training_voices, training_labels = load_directory(training_directory)
testing_voices, testing_labels = load_directory(testing_directory)


# In[97]:


plt.figure()
plt.plot(training_voices[1])

plt.title(f'Sample Speach Signal of {training_labels[1]}')
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.show()


# In[110]:


def get_frame_size(sampling_rate=sampling_rate, frame_duration=20): # 20 ms default frame_duration
    return int(sampling_rate * (frame_duration / 1000))


# In[111]:


def extract_features(audio, frame_size, sr=sampling_rate, n_mfcc=13):

#     audio = librosa.effects.preemphasis(audio)

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        hop_length=int(frame_size // 2),
        win_length=frame_size,
        window=scipy.signal.windows.hamming
    )
    
    # Compute statistics for each feature
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    return np.hstack([mfccs_mean, mfccs_std])


def process_voices(audio_voices, frame_size, sr=sampling_rate, n_mfcc=13):
    features = []
    
    for voice in audio_voices:
        feature = extract_features(voice, frame_size, sr, n_mfcc)
        features.append(feature)
        
    return np.array(features)


# In[112]:


extracted_training_features = process_voices(training_voices, get_frame_size())
extracted_testing_features = process_voices(testing_voices, get_frame_size())


# In[115]:


extracted_training_features[0]


# In[117]:


scaler = StandardScaler()

X_training_scaled = scaler.fit_transform(extracted_training_features)
X_testing_scaled = scaler.fit_transform(extracted_testing_features)


# In[118]:


X_training_scaled[0]


# # Models

# In[119]:


svm_model = SVC(kernel='linear')
svm_model.fit(X_training_scaled, training_labels)

# Example with Random Forest
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(X_training_scaled, training_labels)

# Predict on the test set
y_pred_svm = svm_model.predict(X_testing_scaled)
y_pred_random_forest = rfc_model.predict(X_testing_scaled)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_training_scaled, training_labels)
y_pred_knn = knn_model.predict(X_testing_scaled)

# Evaluate the model
print('SVM Report:')
print(classification_report(testing_labels, y_pred_svm))

print('Random Forest Report:')
print(classification_report(testing_labels, y_pred_random_forest))

print('KNN Report:')
print(classification_report(testing_labels, y_pred_knn))


# # Hyper-parameter Tuning

# In[120]:


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
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# SVM
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, n_jobs=-1, verbose=2)
svm_grid_search.fit(X_training_scaled, training_labels)
best_svm_params = svm_grid_search.best_params_

# Random Forest
rfc_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rfc_param_grid, cv=5, n_jobs=-1, verbose=2)
rfc_grid_search.fit(X_training_scaled, training_labels)
best_rfc_params = rfc_grid_search.best_params_

# KNN
knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, n_jobs=-1, verbose=2)
knn_grid_search.fit(X_training_scaled, training_labels)
best_knn_params = knn_grid_search.best_params_

# SVM
print(f"Best SVM parameters: {best_svm_params}")
print(f"Best Score", svm_grid_search.best_score_)

# RFC
print(f"Best Random Forest parameters: {best_rfc_params}")
print(f"Best Score", rfc_grid_search.best_score_)

# KNN
print(f"Best KNN parameters: {best_knn_params}")
print(f"Best Score", knn_grid_search.best_score_)


# In[ ]:




