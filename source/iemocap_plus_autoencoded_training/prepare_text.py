import re
import os
import pandas as pd
import unicodedata

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)

file2transcriptions = {}

transcripts_path = 'data/mosei_test_filtered.csv'
transcript_df = pd.read_csv(transcripts_path)



#transcript_files = os.listdir(transcripts_path)
for i, f in enumerate(transcript_df['ASR']):
    file2transcriptions[transcript_df['fname'][i]] = f

print(len(file2transcriptions))

x_train_audio = pd.read_csv('data/audio_train_autoencoded_v0.csv')
x_test_audio = pd.read_csv('data/audio_test_autoencoded_v0.csv')

# Prepare text data
text_train = pd.DataFrame()
text_train['wav_file'] = x_train_audio['wav_file']
text_train['label'] = x_train_audio['label']
text_train['transcription'] = [normalizeString(file2transcriptions[code]) for code in x_train_audio['wav_file']]

text_test = pd.DataFrame()
text_test['wav_file'] = x_test_audio['wav_file']
text_test['label'] = x_test_audio['label']
text_test['transcription'] = [normalizeString(file2transcriptions[code]) for code in x_test_audio['wav_file']]

text_train.to_csv('data/text_train_autoencoded.csv', index=False)
text_test.to_csv('data/text_test_autoencoded.csv', index=False)

print(text_train.shape, text_test.shape)
print("\nText data extracted and splitted!")
