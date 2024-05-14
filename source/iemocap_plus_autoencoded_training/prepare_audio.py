import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/audio_features_autoencoded_v0.csv')
df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]

# change 7 to 2
df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})

# Parse JSON strings back into arrays
df['mfccs'] = df['mfccs'].apply(lambda x: json.loads(x))
df['mfccsstd'] = df['mfccsstd'].apply(lambda x: json.loads(x))
df['mfccmax'] = df['mfccmax'].apply(lambda x: json.loads(x))
df['chroma'] = df['chroma'].apply(lambda x: json.loads(x))
df['mel'] = df['mel'].apply(lambda x: json.loads(x))
df['contrast'] = df['contrast'].apply(lambda x: json.loads(x))


# Separate label column and drop unnecessary columns
y = df['label']
X = df.drop(['wav_file', 'label'], axis=1)

# Expand array columns into multiple single-value columns
array_columns = ['mfccs', 'mfccsstd', 'mfccmax', 'chroma', 'mel', 'contrast']
expanded_arrays = {}
for col in array_columns:
    expanded_arrays[col] = pd.DataFrame(X[col].tolist(), index=X.index).add_prefix(col+'_')

# Concatenate expanded single-value columns with remaining single-value columns
X_single_value = X.drop(array_columns, axis=1)
X_expanded = pd.concat([X_single_value] + list(expanded_arrays.values()), axis=1)
label_wav = df[['label', 'wav_file']]
X_expanded = pd.concat([label_wav, X_expanded], axis=1)

# Apply MinMaxScaler to scale the numeric values
scalar = MinMaxScaler()
X_expanded[X_expanded.columns[2:]] = scalar.fit_transform(X_expanded[X_expanded.columns[2:]])
X_expanded.head()

# Split the data into training and testing sets
X_train, X_test = train_test_split(X_expanded, test_size=0.2)

#X_expanded.to_csv('source/generated_audio/data/audio_test_gan_v0.csv', index=False)
X_train.to_csv('data/audio_train_autoencoded_v0.csv', index=False)
X_test.to_csv('data/audio_test_autoencoded_v0.csv', index=False)
print(X_expanded.shape)

print("\nAudio data prepared and splitted!")
