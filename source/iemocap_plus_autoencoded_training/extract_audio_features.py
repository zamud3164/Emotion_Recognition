import pickle
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd
import json


#data_dir = 'data/'
df_data_path = 'data/mosei_test_filtered.csv'
audio_vectors_path = 'data/audio_vectors_autoencoded.pkl'

columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std', 'mfccs', 'mfccsstd', 'mfccmax', 'chroma', 'mel', 'contrast', 'zeroocr']

df_list = []

emotion_dict = {'anger': 0,
                'happy': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fear': 5,
                'surprised': 6,
                'neutral': 7,
                'xxx': 8,
                'oth': 8}


df_data = pd.read_csv(df_data_path)

sample_rate = 16000

audio_vectors = pickle.load(open(audio_vectors_path, 'rb'))

print(len(audio_vectors))

#for index, row in enumerate(audio_vectors_file):
for index, row in tqdm(df_data.iterrows()):
    wav_file_name = row['fname']
    label = emotion_dict[row['label']]
    y = audio_vectors[wav_file_name]

    feature_list = [wav_file_name, label]  # wav_file, label
    
    sig_mean = np.mean(abs(y))
    feature_list.append(sig_mean)  # sig_mean
    feature_list.append(np.std(y))  # sig_std

    rmse = librosa.feature.rms(y=y + 0.0001)[0]
    feature_list.append(np.mean(rmse))  # rmse_mean
    feature_list.append(np.std(rmse))  # rmse_std

    silence = 0
    for e in rmse:
        if e <= 0.4 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))
    feature_list.append(silence)  # silence

    y_harmonic = librosa.effects.hpss(y)[0]
    feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic 
    
    cl = 0.45 * sig_mean
    center_clipped = []
    for s in y:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
    feature_list.append(np.std(auto_corrs))  # auto_corr_std

    stft = np.abs(librosa.stft(y))

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)

    mfccsstd = np.std(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    mfccmax = np.max(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)    
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)    

    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sample_rate).T, axis=0)    

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)    

    zerocr = np.mean(librosa.feature.zero_crossing_rate(y))

    feature_list.append(mfccs.tolist())
    feature_list.append(mfccsstd.tolist())
    feature_list.append(mfccmax.tolist())
    feature_list.append(chroma.tolist())
    feature_list.append(mel.tolist())
    feature_list.append(contrast.tolist())

    feature_list.append(zerocr)
    


    df_list.append(feature_list)

        

# Create DataFrame from df_list
df_features = pd.DataFrame(df_list, columns=columns)
df_features.to_csv('data/audio_features_autoencoded_v0.csv', index=False)

print("\nAudio features extracted and saved!")