import re
import os
import pandas as pd

print("This script will extract the data from the IEMOCAP dataset and save it as a CSV file.")
print("The path for the IEMOCAP dataset on my laptop is: C:/Users/Zara Mudassar/OneDrive - OsloMet/Spring 1/4630/IEMOCAP/")
print("Please locate thepath to the IEMOCAP dataset on your laptop to be able to input it below, see my path as an example.")
global_path = input("Enter the path to the IEMOCAP dataset: ")

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

for sess in range(1, 6):
    emo_evaluation_dir = global_path + 'IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
    evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
    for file in evaluation_files:
        with open(emo_evaluation_dir + file) as f:
            content = f.read()
        info_lines = re.findall(info_line, content)
        for line in info_lines[1:]:  # the first line is a header
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
            start_time, end_time = start_end_time[1:-1].split('-')
            start_time, end_time = float(start_time), float(end_time)
            start_times.append(start_time)
            end_times.append(end_time)
            wav_file_names.append(wav_file_name)
            emotions.append(emotion)
    
    print('Session {} done.'.format(sess))


df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion'])

df_iemocap['start_time'] = start_times
df_iemocap['end_time'] = end_times
df_iemocap['wav_file'] = wav_file_names
df_iemocap['emotion'] = emotions

df_iemocap.tail()


df_iemocap.to_csv('data/df_iemocap.csv', index=False)