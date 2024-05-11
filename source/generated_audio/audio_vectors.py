import librosa
import os
from tqdm import tqdm
import pickle
import pandas as pd
import math

print(librosa.__version__)


print("This script will extract the data from the IEMOCAP dataset and save it as a CSV file.")
print("The path for the IEMOCAP dataset on my laptop is: C:/Users/Zara Mudassar/OneDrive - OsloMet/Spring 1/4630/generated_data/")
print("Please locate thepath to the IEMOCAP dataset on your laptop to be able to input it below, see my path as an example.\n")
global_path = input("Enter the path to the IEMOCAP dataset: ")

df_data = pd.read_csv('source/generated_audio/data/mosei_test_filtered.csv')
data_dir = global_path + 'gans/'
#columns = ['wav_file', 'label', 'audio_vector']
#df_list = []

# print number of rows in df_data
print(df_data.shape[0])


sr = 16000
audio_vectors = {}

#wav_file_path = 'source/generated_audio/audio/'
wav_file_path = data_dir

orig_wav_files = os.listdir(data_dir)
for orig_wav_file in tqdm(orig_wav_files):
    #feature_list = []
    #print(orig_wav_file)
    orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
    #print(orig_wav_file.split('.'))
    #orig_wav_file_0, orig_wav_file_1, file_name, file_format = orig_wav_file.split('.')
    #number, label_name, gan = file_name.split('_')
    #print(label_name)
    #feature_list.append(file_name)
    #feature_list.append(label_name)
    #feature_list.append(orig_wav_vector.tolist())
    #df_list.append(feature_list)
    audio_vectors[orig_wav_file] = orig_wav_vector

with open('source/generated_audio/data/audio_vectors_gan.pkl', 'wb') as f:
        pickle.dump(audio_vectors, f)


#df = pd.DataFrame(df_list, columns=columns)
#df.to_csv('audio_vectors_generated_audio.csv', index=False)

