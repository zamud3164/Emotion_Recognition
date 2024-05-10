# Emotion_Recognition
This project, emotion recognition using multimodal data, was done in the course ACIT4630, in spring 2024. 

### Code
The code for emotion recognition is extracted from several sources and then fused together and modified to suit our task.  Preprocessing the IEMOCAP dataset and extracting features are from https://github.com/Demfier/multimodal-speech-emotion-recognition/tree/master. Setting up the training environment and the logic for saving checkpoints is from https://github.com/IliaZenkov/transformer-cnn-emotion-recognition. These works are also published. There has been made changes for example adding more features and removing unnecessary parts not needed for our specific task. The model architecture is not from these sources but set up by us on the basis of experiments. These two repositories we have used are cited in our report too. 

### Dataset
IEMOCAP is used for this project. IEMOCAP can be downloaded by following this link: https://hioa365-my.sharepoint.com/:f:/g/personal/zamud3164_oslomet_no/En_HHH04uxxBjLoIsivNEhEBuMV2j2kscpLw8YoEEBWm-g?e=Y1Cscr. This link will be available till mid june.
If IEMOCAP not available, the model can be trained and evaluated using the preprocessed files provided under the folder "data", for more details, see the section below.

### Running the project
To run this project from scratch, IEMOCAP is needed, see above for how to get IEMOCAP.
If IEMOCAP not available, the model can be trained and evaluated using the preprocessed files provided under the folder "data".
To skip the IEMOCAP preprocessing, skip steps 2-6. In step 7, we train the model. In step 8, we validate the model. There is already a model saved in the "savedModel" directory, therefore step 8 can be run without having to train a model yourself too (if needed).  

1. Clone the repository and locate the root folder. In the root folder, run "pip install -r requirements.txt" to install all the necessary dependencies.
   
2. Open and run the file "extract_data.py" located in the folder "preprocessing_data". This file will create df_iemocap.csv file in the data folder.

3. Open and run the "build_audio_vectors.py" file located in the folder "preprocessing_data". This file will look at each of the 5 sessions in the IEMOCAP dataset and extract all wav files as audio vectors from these 5 sessions and save them using pickle.

4. Using these extracted audio_vectors, we want to extract features from them. Open and run the file "extract_audio_features.py" located in the folder "preprocessing_data". This file will go through all the audio vectors pickle files, extract features, and save it together with their respective wav file name and emotion to a csv file called "audio_feature_v0.csv". This csv file will be used to further prepare the data and split it in the next step.

5. Using the csv file created in the last step, we further prepare the audio files and splits them so they are ready to be used for training. Open the "prepare_audio_features.py" located in the folder "preprocessing_data" and run it. This will then create two files, "audio_test_v0" and "audio_train_v0". These two files are also provided from before in the data folder.

6. Now that audio files are ready, we will prepare the text data files. Open and run the "extract_text_transcriptions.py" file in the folder "preprocessing_data". This file will extract text and then split it into training and testing. It will create two files, "text_train.csv" and "text_test.csv". These files are also provided from before in the data folder.

7. Now that we have prepared our training data, we will move over to the main part of the code, the training. Open the notebook "CNN.ipynb" and run the cells. In this notebook, we first load the data files, vectorize the text data, and fuse text and audio data. After fusing the data, we define our model architecture and other functions such as validation, training step, and saving checkpoints. We then instiate the model and run it for 100 epochs. Then we create the plots to see the loss plot and accuracy plot.

8. Run "validation.ipynb" to validate the trained model using test data. First the notebook will load the test data, both audio and text. Then it will vectorize the text data and then fuse both modlaities to a combined test data. Then it will use this data to evaluate the model. It will give the accuracy, precision, recall, and F1-score as well as a classification report.

