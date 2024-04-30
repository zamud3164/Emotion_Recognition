import librosa
import os
from tqdm import tqdm
import pickle
import pandas as pd
import math


print("This script will extract the data from the IEMOCAP dataset and save it as a CSV file.")
print("The path for the IEMOCAP dataset on my laptop is: C:/Users/Zara Mudassar/OneDrive - OsloMet/Spring 1/4630/IEMOCAP/")
print("Please locate thepath to the IEMOCAP dataset on your laptop to be able to input it below, see my path as an example.\n")
global_path = input("Enter the path to the IEMOCAP dataset: ")


df_iemocap = pd.read_csv('data\df_iemocap.csv')
iemocap_dir = global_path + 'IEMOCAP_full_release/'


sr = 16000
audio_vectors = {}
for sess in range(1, 6):
    print('Session {} started.'.format(sess))
    wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        try:
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in df_iemocap[df_iemocap['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                start_frame = math.floor(start_time * sr)
                end_frame = math.floor(end_time * sr)
                truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                audio_vectors[truncated_wav_file_name] = truncated_wav_vector
        except:
            print('An exception occured for {}'.format(orig_wav_file))
    with open('data/audio_vectors/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
        pickle.dump(audio_vectors, f)

    print('Session {} done.'.format(sess))

