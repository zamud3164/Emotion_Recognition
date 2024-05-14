# Emotion_Recognition
This project, emotion recognition using multimodal data, was done in the course ACIT4630, in spring 2024. 

### Code
The code for emotion recognition is extracted from several sources and then fused together and modified to suit our task.  Preprocessing the IEMOCAP dataset and extracting features are from https://github.com/Demfier/multimodal-speech-emotion-recognition/tree/master. Setting up the training environment and the logic for saving checkpoints is from https://github.com/IliaZenkov/transformer-cnn-emotion-recognition. These works are also published. There has been made changes for example adding more features and removing unnecessary parts not needed for our specific task. The model architecture is not from these sources but set up by us on the basis of experiments. These two repositories we have used are cited in our report too. 

### Dataset
IEMOCAP is used for this project. IEMOCAP can be downloaded by following this link: https://hioa365-my.sharepoint.com/:f:/g/personal/zamud3164_oslomet_no/En_HHH04uxxBjLoIsivNEhEBuMV2j2kscpLw8YoEEBWm-g?e=Y1Cscr. This link will be available till mid june.
If IEMOCAP not available, the model can be trained and evaluated using the preprocessed files provided under the folder "data", for more details, see 'Running the project' section.

The generated samples used in this project can be downloaded from this link: https://hioa365-my.sharepoint.com/:f:/g/personal/zamud3164_oslomet_no/Evgk8vEz1ZNGp8sCh23U67cB5hg1RVBlb06lLAq-Vs9LaQ?e=x0TZ1x. Download and store it on your machine. When running the project, if one wants to preprocess the generated samples, it will ask for the path to this folder as input. In this folder, there are two folders, "gan" and "autoencoder". Just download the whole folder, given in the link, and give path to it. The preprocessing files will themselves, based on which of the wto in need, get it. The model can also be trained without having to preprocess any data yourself, because the finished preprocess data is available in the data folder. See section 'Running the project' section for details. 

### Saved Model
The trained models are too big and therefore it is not possible to push them to this github repo. Download the four trained models from this link: https://hioa365-my.sharepoint.com/:f:/g/personal/zamud3164_oslomet_no/ElfJL7rP-TpBo_R_0xNrwOsBdncHqtGWBeS1xejP4oBuvw?e=OshQ8p and save it in the savedModel directory of this project. Then the notebook CNN_v1 and CNN_v2 can be used to check the model performance without having to train the models yourself. To check a model without training, run all celles in the notebooks excluding the one cell for training. 

### Running the project
To run this project from scratch, IEMOCAP and the generated samples are needed, see above for how to get this data.
If data not available, the model can be trained and evaluated using the preprocessed files provided under the folder "data".
To skip the data preprocessing, skip all steps in both the instructions below and only run the last step, the step where the notebook is runned to train and evaluate the model. For the last step, there is already models saved in the "savedModel" directory, therefore is you do not want to train the model from the scratch, you can run all cells in the notebook excluding the one cell that does the training, and that way just evaluate the model. For this, you will need to download the saved models (see above) and place them in the savedModel directory.  

1. Clone the repository and locate the root folder. In the root folder, run "pip install -r requirements.txt" to install all the necessary dependencies. The project is structures as following: preprocessed data files in the "data" folder. "savedModel" folder to add the saved models too. "souce" folder containing the code for running this project. "source" is divided in two, one for training a model only on IEMOCAP and one for training on IEMOCAP plus generated samples. 

##### IEMOCAP plus generated samples
Generated samples are both from autoencoder (iemocap_plus_autoencoded_training) and from gan (iemocap_plus_gan_training). Open the folder you want too train with and follow the instructions below. The instructions are same for both, the only difference in the files is where the audio is taken from. iemocap_plus_gan gives better results and are used as benchmark for this project. 

1. Open the folder source --> iemocap_plus_generated_training.

2. Open and run the file "audio_vectors.py". It will ask for the path to the generated samples as input.

3.  Open and run "extract_audio_features.py" then prepare_audio.py". Then run "prepare_text.py".

4. Open and run "combine_data.py". This file combined the preprocessed iemocap data with the preprocessed generated samples data created from the steps above. If you have not yourself preprocessed the iemocap data, it will be provided in the right folder from before anyways. Now we are done with the preprocessing and can train the model!

5. Open and run the notebook "CNN_v2.ipynb" to train and evaluate the model. If you only want to evaluate an already savedModel you downloaded from the link given above, run all cells in this notebook expect for the one cell which does the actual training (it is the cell where number of epochs and the training is initialized).

#### IEMOCAP only
1. Open the folder source --> iemocap_training

2. Open and run "extract_data.py" and "build_audio_vectors.py". Both will ask for the IEMOCAP dataset path as input.

3. Open and run "extract_audio_features.py" first and then "prepare_audio_features.py". Then run "extract_text_transcriptions.py". Now the preprocessing is done and we can train and evaluate the model!
   
4. Open and run the notebook "CNN_v1.ipynb" to train and evaluate the model. If you only want to evaluate an already savedModel you downloaded from the link given above, run all cells in this notebook expect for the one cell which does the actual training (it is the cell where number of epochs and the training is initialized).

