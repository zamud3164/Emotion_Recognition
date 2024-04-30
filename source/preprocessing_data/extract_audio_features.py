import pickle
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd


data_dir = 'data/'
df_iemocap_path = '{}df_iemocap.csv'.format(data_dir)
audio_vectors_path = '{}audio_vectors/audio_vectors_'.format(data_dir)

columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std', 'mfccs', 'mfccsstd', 'mfccmax', 'chroma', 'mel', 'contrast', 'zeroocr']

df_list = []

emotion_dict = {'ang': 0,
                'hap': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fea': 5,
                'sur': 6,
                'neu': 7,
                'xxx': 8,
                'oth': 8}


df_iemocap = pd.read_csv(df_iemocap_path)

sample_rate = 16000

for sess in (range(1, 6)):
        try:
            audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
        except pickle.UnpicklingError as e:
            print(f"Error loading pickle file for session {sess}: {e}")
            continue  # Skip to the next iteration of the loop
        for index, row in tqdm(df_iemocap[df_iemocap['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
            try:
                wav_file_name = row['wav_file']
                label = emotion_dict[row['emotion']]
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
                feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)

                # based on the pitch detection algorithm mentioned here:
                # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
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
            except Exception as e:
                print('Some exception occurred: {}'.format(e))

        print('Session {} done.'.format(sess))

# Create DataFrame from df_list
df_features = pd.DataFrame(df_list, columns=columns)
df_features.to_csv('data/audio_features_v0.csv', index=False)