
"""

Original Author: Ishwar Kumar S (ishwarr18@gmail.com)
Date: AUG - 2020

AHEM - Machine Learning Prototype to detect Covid-19 just by a cough

This module will pre - process raw input via Flask and evaluate the data on a trained Sequential Keras model
to predict if a person has Covid-19 or not.

Please note that the accuracy will improve based on the amount of data used to train the model.
Currently the model is trained with symptoms and coughs
of 16 people(PCR tested for COVID-19), which is very limited.
Hence, the results may not be with high accuracy.
Average accuracy with 80% training data and 20% testing data is ~66%
However, the model can be trained with more data once it is available and can yield higher accuracy upon time.

Modules used - Librosa, Sklearn, Keras (Tensorflow), pandas, numpy, Flask, Javascript, CSS
"""
#import all required modules
import flask
from flask import Flask, render_template, request
import logging
from subprocess import run, PIPE
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from sklearn.preprocessing import LabelEncoder, StandardScaler

import keras
from keras import models
from keras import layers
from pandas import read_csv
import pandas as pd

import warnings
warnings.simplefilter("ignore")
import soundfile as sf
import itertools

#Regularizing patient reported symptoms to binaries
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
            if 'wav' in j:
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

#Returns a single digit value based on the age group
def map_age(data):
    featureMap = dict()
    for i in data['age']:
        featureMap[i] = get_age(i)
    data['age'] = data['age'].map(featureMap)
    return data

#Returns the first digit of age to determine age group
def get_age(age):

    age /= 10
    return int(age)

#Removes the Non-Cough sound from the audio
def remove_silence(x):
    print(librosa.get_duration(x))
    y = librosa.effects.split(x,top_db=30)
    l = []
    for i in y:
        l.append(x[i[0]:i[1]])
    x = np.concatenate(l,axis=0)
    print(librosa.get_duration(x))
    return x

#Trims the audio to 5 seconds length
def trim_audio(x, sr):
    print('Trimming...')
    print(librosa.get_duration(x))
    sf.write('trimmed.wav', x, sr)
    y, sr = librosa.load('trimmed.wav', mono=True, duration=5)
    print(librosa.get_duration(y))
    return y, sr

#Extracts multiple essential featured from the cough audio
def generate_cough_data():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate pitch magnitude'
    for i in range(1, 41):
        header += f' mfcc{i}'

    header = header.split()

    file = open('new_tmp.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    soundname = 'tmp_audio.wav'
    filename = 'tmp_audio.wav'
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
    file = open('new_tmp.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

#Normalize to standard scalar format using Means and Standard Deviations from the tranined model
def normalize_data(X):
    m = np.genfromtxt("means.csv", delimiter=",")
    m = m.reshape((1,64))
    s = np.genfromtxt("std.csv", delimiter=",")
    s = s.reshape((1,64))
    z = []
    for i, j, k in zip(X, m, s):
        z.append((i - j) / k)
    X = np.asarray(z)
    return X


#Generates a combined dataset of the age, gender, symptoms and various featured extracted from the cough audio
def generate_new_features(raw_data):
    dataset1 = pd.DataFrame.from_dict(raw_data)
    dataset1 = map_reported_symptom(dataset1)
    dataset1 = map_age(dataset1)
    dataset1.to_csv('new_symptom.csv')

    generate_cough_data()
    dataset2 = pd.read_csv('new_tmp.csv')
    dataset = pd.merge(left=dataset1, right=dataset2, left_on='cough_filename', right_on='filename')
    dataset = dataset.drop(['filename'], axis=1)
    dataset = dataset.drop(['cough_filename'], axis=1)
    dataset.to_csv('final_input.csv')
    return dataset

#Loads a keras trained model and evaluates the prediction for the input dataset
def predict_covid(dataset):

    model = keras.models.load_model('model/recon_model')

    #scaler = StandardScaler()
    X = np.array(dataset, dtype=float)
    print('Shape of X:', X.shape)
    #X_transformed = scaler.fit_transform(X.reshape((64, 1)))
    X = normalize_data(X)
    print('Shape of X after normalize:', X.shape)
    predictions = []
    predictions = model.predict(X)
    classes = np.argmax(predictions, axis=1)
    print(classes)
    if classes[0] == 1:
        return 'You are likely NOT having COVID-19:)'
    else:
        return 'You are likely having Covid-19!!'

#Creates a flask app and connects with the front end webpages created with CSS and JS
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():

    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))
    if flask.request.method == 'POST':

        age = flask.request.form['age']
        gender = flask.request.form['gender']
        smoker = flask.request.form['smoker']
        print(age, gender, smoker)
        reported_symptoms = flask.request.values.getlist('reported_symptoms')
        reported_symptoms = ','.join(reported_symptoms)
        file = request.files['file']
        file.save('tmp_audio.wav')
        raw_data = {'age':[int(age)], 'gender':[int(gender)], 'smoker':[int(smoker)], 'patient_reported_symptoms':[reported_symptoms], 'cough_filename':['tmp_audio.wav']}
        features = generate_new_features(raw_data)
        pred = predict_covid(features)
        return flask.render_template('result.html',
                                     result=pred,
                                     )
@app.route('/result', methods=['POST'])
def home():
    return (flask.render_template('index.html'))

#Lets run the code
if __name__ == '__main__':
    app.run(debug = True)
