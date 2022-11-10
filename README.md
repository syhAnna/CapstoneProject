# Capstone Project: Automation System for Email Response
## UCLA Master's Degree Capstone Project (Fall 2022)

Google Drive link for large data files that cannot be uploaded: https://drive.google.com/drive/folders/19sByFpcVTjR-kuzpwOMMMLZVITVhdFkJ?usp=share_link


## Directory Detail:
    .
    ├── requirements.txt                                                  # Project environment requirements file        
    │
    ├── DataPreproDataPreprocess+NewModelDraft.ipynb                      # Jupyter file contains data processing and tarined new model         
    │
    ├── Web application   
    │   ├── app.py                                                        # Smart Reply Suggestion Web application (new model)
    │   └── templates/index.html                                          # Web template html file 
    │
    ├── util files
    │   ├── utils.py                                                      # Helper functions for both baseline model and new model
    │   ├── preprocess.py                                                 # Clean and process both baseline and new model's raw data
    │   ├── vectorization.py                                              # Vectorize the cleaned data
    │   └── smart_reply.py                                                # The SmartReply class that can display the new model result
    │
    ├── model                  
    │   ├── Baseline model             
    │   │   ├── baseline_enc_dec.py                                       # Encoder and Decoder classes
    │   │   ├── baseline_model.py                                         # Construct and train the baseline LSTM model
    │   │   └── baseline_inference.py                                     # Model inference of trained baseline model
    │   │    
    │   └── New model
    │       ├── my_model_sim_matrix.py                                    # Construct the similarity matrix using Annoy measures similarity distance
    │       ├── my_model_clustering.py                                    # DBSCAN clustering
    │       ├── my_model.py                                               # Construct and train the new LSTM model
    │       └── my_model_inference.py                                     # Model inference of trained new model
    │
    └── data files                  
        ├── BaselineDataFiles            
        │   ├── glove.6B.300d.txt                                         # GloVe (300-dimensional) - pretrained word emdeddings
        │   ├── single_qna.csv                                            # Baseline model raw dataset: Amazon Question/Answer Dataset
        │   ├── processed_clean_single_qna.csv                            # Cleaned and processed dataset
        │   ├── qa_tokenizer.pickle                                       # Tokenized question-and-answer messages data
        │   ├── enc_train.pickle, enc_val.pickle, enc_test.pickle         # Splitted dataset for encoder training
        │   ├── dec_train.pickle, dec_val.pickle, dec_test.pickle         # Splitted dataset for decoder training 
        │   ├── baseline_encoder_weights.h5, baseline_decoder_weights.h5  # Files contained trained baseline model weights
        │   └── true_ans_lst.pickle, pred_ans_lst.pickle                  # The inference output of trained baseline model
        │    
        └── MyModelDataFiles
            ├── topical_chat.csv                                          # New model raw dataset: Chatbot Dataset Topical Chat
            ├── cleaned_topical_chat.csv                                  # Cleaned dataset
            ├── input_texts.pickle, target_texts.pickle                   # Vectorized input and target texts
            ├── input_words_set.pickle, target_words_set.pickle           # The set of unique input and target words of input and target texts
            ├── input_sequences.pickle, target_sequences.pickle           # Encoded and padded input and target seuqences
            ├── my_model_input_annoy.ann, my_model_target_annoy.ann       # The Annoy index of input and target texts 
            ├── input_sim_matrix.pickle, target_sim_matrix.pickle         # The input adnd target texts similarity matrices constructed using Annoy index
            ├── target_dbscan_0008.pickle                                 # The DBSCAN object with eps=0.008
            └── my_lstm_model_0008.hdf5                                   # File contained trained new model object


## Getting Started:
The instructions of running the project on your local machine. Install Python 3.10 on your machine.
### Setting up the project:
* Clone the repository:
```
git clone https://github.com/syhAnna/CapstoneProject.git
```
* Navigate to the repository
```
cd CapstoneProject
```
* Install the requirments.txt using:
```
pip install -r requirements.txt
```
### Usage:
* Run new model in terminal (sample output is on the left side of image below):
```
python3 smart_reply.py
```
* Run new model Web Application (sample output is on the right side of image below):
```
python3 app.py
```
Then, point your browser to http://localhost:5000/ (http://127.0.0.1:5000/)
### Preview (Terminal Output & Web App Display): 
![alt text](https://github.com/syhAnna/CapstoneProject/blob/main/imgs/sample_web.png?raw=true)


## Requirements:
* Download the large files in Google Drive (https://drive.google.com/drive/folders/19sByFpcVTjR-kuzpwOMMMLZVITVhdFkJ?usp=share_link) into the correspond directories in this depository
* Run 'smart_reply.py' file to play with trained new model, in order to run successfully, pass the required file paths in 'my_model_inference.py':
    * let DBSCAN_FNAME = path to file 'target_dbscan_0008.pickle'
    * let MODEL_FNAME = path to file 'my_lstm_model_0008.hdf5'


## Data: 
* Raw Data
    * (Baseline: 'single_qna.csv') Amazon Question/Answer Dataset: https://www.kaggle.com/datasets/praneshmukhopadhyay/amazon-questionanswer-dataset
    * (New model: 'topical_chat.csv') Chatbot Dataset Topical Chat: https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat
* Data Preprocessing
    *  Data cleaning: extend the abbreviation; remove html, number, emoji, punctuations; transfer all uppercase to lowercase
    *  Munipulate the columns (add / remove), remove the rows have NaN values
* Data Manipulation
    *  Data tokenization, vectorization, padding
    *  Baseline: pretrained word emdeddings - GloVe (300-dimensional) - glove.6B.300d.txt
    *  New model: use Annoy (Approximate Nearest Neighbors Oh Yeah) to contruct input and target similarity matrices


## Model Construction:
* Baseline model: LSTM + Beam Search
* New model: Similarity Distance (Annoy) + DBSCAN clustering + LSTM


## Techniques:
* Baseline model: keras LSTM model, BLEU score measurement, GloVe pretrained word embeddings
* New model: Annoy similarity distance measurement technique, DBSCAN clustering, keras LSTM model

Research paper link:

PPT link: https://drive.google.com/file/d/1OCfDZdY8Oduq3OwGdU2ztc4frVGo1z_R/view?usp=share_link 


