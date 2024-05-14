import pandas as pd

# Combine audio data
audio_train = pd.read_csv('data/audio_train_autoencoded_v0.csv')
audio_test = pd.read_csv('data/audio_test_autoencoded_v0.csv')
audio_train_iemocap = pd.read_csv('data/audio_train_v0.csv')
audio_test_iemocap = pd.read_csv('data/audio_test_v0.csv')

audio_train = pd.concat([audio_train, audio_train_iemocap])
audio_test = pd.concat([audio_test, audio_test_iemocap])

audio_train.to_csv('data/audio_train_autoencoded_v0.csv', index=False)
audio_test.to_csv('data/audio_test_autoencoded_v0.csv', index=False)

# Combine text data
text_train = pd.read_csv('data/text_train_autoencoded.csv')
text_test = pd.read_csv('data/text_test_autoencoded.csv')
text_train_iemocap = pd.read_csv('data/text_train.csv')
text_test_iemocap = pd.read_csv('data/text_test.csv')

text_train = pd.concat([text_train, text_train_iemocap])
text_test = pd.concat([text_test, text_test_iemocap])

text_train.to_csv('data/text_train_autoencoded.csv', index=False)
text_test.to_csv('data/text_test_autoencoded.csv', index=False)