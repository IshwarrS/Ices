import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
# Keras
import keras
from keras import models
from keras import layers
from pandas import read_csv
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import soundfile as sf
import pickle

def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    print(featureMap)
    data[feature]=data[feature].map(featureMap)
    return data

# def map_label(data,feature):
#     featureMap=dict()
#     for i in sorted(data[feature].unique(),reverse=True):
#         if i == 'positive':
#             featureMap[i]=0
#         if i == 'negative':
#             featureMap[i] = 1
#     print(featureMap)
#     data[feature]=data[feature].map(featureMap)
#     return data

def map_gender(data,feature):
    featureMap=dict()
    for i in sorted(data[feature].unique(),reverse=True):
        if i == 'male':
            featureMap[i]=0
        if i == 'female':
            featureMap[i] = 1
    print(featureMap)
    data[feature]=data[feature].map(featureMap)
    return data

def map_smoker(data,feature):
    featureMap=dict()
    for i in sorted(data[feature].unique(),reverse=True):
        if i == 'yes':
            featureMap[i]=0
        if i == 'no':
            featureMap[i] = 1
    print(featureMap)
    data[feature]=data[feature].map(featureMap)
    return data

def map_reported_symptom(data):

    featureMap=dict()
    symptom = 'cough_filename,Body aches,Cough (dry),Cough (wet with mucus),Fever-chills or sweating,Headaches,Loss of taste,Loss of smell,New or worsening cough,Shortness of breath,Sore throat,Tightness in chest,Vomiting and diarrhoea,none'
    symptoms = symptom.split(',')
    data['combined_data'] = data['cough_filename'].str.cat(data['patient_reported_symptoms'], sep =",")
    data = data.sort_index()
    print(data['cough_filename'], data['combined_data'])
    for i in symptoms:
        featureMap[i] = 0

    symptom_list = []

    for i in data['combined_data']:
        s = i.split(',')

        for j in s:
            if 'mp3' in j:
                featureMap['cough_filename'] = j
            else:
                featureMap[j] = 1
        #print(featureMap)
        symptom_list.append(featureMap.copy())
        for k in symptoms:
            featureMap[k] = 0
    df = pd.DataFrame(symptom_list)
    print(df)
    df.to_csv('df.csv')
    data = pd.merge(left = data, right = df, left_on='cough_filename', right_on='cough_filename')
    print(data.shape)
    data = data.drop(['combined_data'], axis = 1)
    data = data.drop(['patient_reported_symptoms'], axis=1)
    return data

def map_age(data):
    featureMap = dict()
    for i in data['age']:
        featureMap[i] = get_age(i)
    data['age'] = data['age'].map(featureMap)
    return data

def get_age(age):

    age /= 10
    return int(age)

def remove_silence(x):
    print(librosa.get_duration(x))
    y = librosa.effects.split(x,top_db=30)
    l = []
    for i in y:
        l.append(x[i[0]:i[1]])
    x = np.concatenate(l,axis=0)
    print(librosa.get_duration(x))
    return x

def trim_audio(x, sr):
    print('Trimming...')
    print(librosa.get_duration(x))
    sf.write('trimmed.wav', x, sr)
    y, sr = librosa.load('trimmed.wav', mono=True, duration=5)
    print(librosa.get_duration(y))
    return y, sr


# generating a dataset
def generate_data():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate pitch magnitude'
    for i in range(1, 41):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    result = 'positive negative'.split()
    for r in result:
        for filename in os.listdir(f'./results/{r}'):
            soundname = f'./results/{r}/{filename}'
            print(soundname)
            y, sr = librosa.load(soundname, mono=True)
            y = remove_silence(y)
            y, sr = trim_audio(y, sr)

            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(pitches)} {np.mean(magnitudes)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {r}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

dataset1 = read_csv('labels.csv')
print(dataset1.shape)
dataset1 = map_reported_symptom(dataset1)
dataset1 = map_age(dataset1)
dataset1 = map_gender(dataset1, 'gender')
dataset1 = map_smoker(dataset1, 'smoker')
dataset1 = dataset1.drop(['date'], axis = 1)
dataset1 = dataset1.drop(['corona_test'], axis = 1)
dataset1.to_csv('symptommerged.csv')
# reading dataset from csv

generate_data()
dataset2 = pd.read_csv('data.csv')
dataset = pd.merge(left = dataset1, right = dataset2, left_on='cough_filename', right_on='filename')
dataset = dataset.drop(['filename'], axis=1)
dataset = dataset.drop(['cough_filename'], axis=1)
dataset = mapping(dataset, 'label')
dataset.to_csv('final_data.csv')


label_list = dataset.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(label_list)
print(y)

# normalizing
scaler = StandardScaler()
X = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))

# spliting of dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128,
                    validation_data = (X_train, y_train))

model.save('final_model')

reconstructed_model = keras.models.load_model("final_model")



#
# calculate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print('test_loss: ', test_loss)
print('test_acc: ', test_acc)



# predictions

print("Generate predictions")
predictions = model.predict(X_test)

# calculate accuracy


classes = np.argmax(predictions, axis = 1)
print(classes)
print(y_test)

# model.save('final_model')


reconstructed_model.fit(X_test, y_test,
epochs = 100,
         batch_size = 128,
validation_data = (X_test, y_test)
)

test_loss, test_acc = reconstructed_model.evaluate(X, y, batch_size=128)
print('test_loss: ', test_loss)
print('test_acc: ', test_acc)
# predictions

print("Generate predictions")
predictions = reconstructed_model.predict(X)

# calculate accuracy


classes = np.argmax(predictions, axis = 1)
print(y)
print(classes)
print(X.shape)
print(len(scaler.mean_), len(scaler.scale_))
np.savetxt("X.csv", X, delimiter=",")
np.savetxt("means.csv", scaler.mean_, delimiter=",")
np.savetxt("std.csv", scaler.scale_, delimiter=",")
reconstructed_model.save('recon_model')
