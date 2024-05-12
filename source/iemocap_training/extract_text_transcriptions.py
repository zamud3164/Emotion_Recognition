import re
import os
import pandas as pd
import unicodedata

print("This script will extract the data from the IEMOCAP dataset and save it as a CSV file.")
print("The path for the IEMOCAP dataset on my laptop is: C:/Users/Zara Mudassar/OneDrive - OsloMet/Spring 1/4630/IEMOCAP/")
print("Please locate thepath to the IEMOCAP dataset on your laptop to be able to input it below, see my path as an example.")
global_path = input("Enter the path to the IEMOCAP dataset: ")

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

for sess in range(1, 6):
    transcripts_path = global_path + 'IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
    transcript_files = os.listdir(transcripts_path)
    for f in transcript_files:
        with open('{}{}'.format(transcripts_path, f), 'r') as f:
            all_lines = f.readlines()

        for l in all_lines:
            audio_code = useful_regex.match(l).group()
            transcription = l.split(':')[-1].strip()
            file2transcriptions[audio_code] = transcription

print(len(file2transcriptions))

x_train_audio = pd.read_csv('data/audio_train_v0.csv')
x_test_audio = pd.read_csv('data/audio_test_v0.csv')

# Prepare text data
text_train = pd.DataFrame()
text_train['wav_file'] = x_train_audio['wav_file']
text_train['label'] = x_train_audio['label']
text_train['transcription'] = [normalizeString(file2transcriptions[code]) for code in x_train_audio['wav_file']]

text_test = pd.DataFrame()
text_test['wav_file'] = x_test_audio['wav_file']
text_test['label'] = x_test_audio['label']
text_test['transcription'] = [normalizeString(file2transcriptions[code]) for code in x_test_audio['wav_file']]

text_train.to_csv('data/text_train.csv', index=False)
text_test.to_csv('data/text_test.csv', index=False)

print(text_train.shape, text_test.shape)
print("\nText data extracted and splitted!")
