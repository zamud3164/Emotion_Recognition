import librosa
import os
from tqdm import tqdm
import pickle
import pandas as pd
import math

print(librosa.__version__)


print("This script will extract the data from the generated data and save it as a CSV file.")
print("The path for the generated data on my laptop is: C:/Users/Zara Mudassar/OneDrive - OsloMet/Spring 1/4630/generated_data/")
print("Please locate the path to the generated data on your laptop to be able to input it below, see my path as an example.\n")
global_path = input("Enter the path to the data: ")

df_data = pd.read_csv('data/mosei_test_filtered.csv')
data_dir = global_path + 'autoencoders/'

sr = 16000
audio_vectors = {}

wav_file_path = data_dir

orig_wav_files = os.listdir(data_dir)
for orig_wav_file in tqdm(orig_wav_files):
    orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
    audio_vectors[orig_wav_file] = orig_wav_vector

with open('data/audio_vectors_autoencoded.pkl', 'wb') as f:
        pickle.dump(audio_vectors, f)

