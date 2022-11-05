# CapstoneProject: Automation System for Email Response
## UCLA Master Degree Capstone Project (Fall 2022)

Google Drive link for large data files that cannot be uploaded: https://drive.google.com/drive/folders/19sByFpcVTjR-kuzpwOMMMLZVITVhdFkJ?usp=share_link

### Directory Detail:
    .
    ├── DataPreproDataPreprocess+NewModelDraft.ipynb                    # Jupyter file contains data processing and tarined new model         
    │
    ├── util files
    │   ├── utils.py                                                    # Helper functions for both baseline model and new model
    │   ├── preprocess.py                                               # Clean and process both baseline and new model's raw data
    │   ├── vectorization.py                                            # Vectorize the cleaned data
    │   └── smart_reply.py                                              # The SmartReply class that can display the new model result
    │
    ├── model                  
    │   ├── Baseline model             
    │   │   ├── baseline_enc_dec.py                                     # Encoder and Decoder classes
    │   │   ├── baseline_model.py                                       # Construct and train the baseline LSTM model
    │   │   └── baseline_inference.py                                   # Model inference of trained baseline model
    │   │    
    │   └── New model
    │       ├── my_model_sim_matrix.py                                  # 
    │       ├── my_model_clustering.py
    │       ├── my_model.py
    │       └── my_model_inference.py
    │
    └── data files                  
        ├── BaselineDataFiles            
        │   ├── glove.6B.300d.txt
        │   ├── single_qna.csv
        │   ├── processed_clean_single_qna.csv
        │   ├── qa_tokenizer.pickle
        │   ├── enc_train.pickle, enc_val.pickle, enc_test.pickle
        │   ├── dec_train.pickle, dec_val.pickle, dec_test.pickle
        │   ├── baseline_encoder_weights.h5, baseline_decoder_weights.h5  # Files contained trained baseline model weights
        │   └── true_ans_lst.pickle, pred_ans_lst.pickle
        │    
        └── MyModelDataFiles
            ├── topical_chat.csv
            ├── cleaned_topical_chat.csv
            ├── input_texts.pickle, target_texts.pickle
            ├── input_words_set.pickle, target_words_set.pickle
            ├── input_sequences.pickle, target_sequences.pickle
            ├── input_sim_matrix.pickle, target_sim_matrix.pickle
            ├── target_dbscan_0008.pickle
            ├── my_model_input_annoy.ann, my_model_target_annoy.ann
            ├── one_hot_labels
            └── my_lstm_model_0008.hdf5


### Requirements:
* Run 'smart_reply.py' file to play with trained new model
    * In order to run  
    * ...
Example: 


## Data 
* Raw Data
    * (Baseline: 'single_qna.csv') Amazon Question/Answer Dataset: https://www.kaggle.com/datasets/praneshmukhopadhyay/amazon-questionanswer-dataset
    * (New model: 'topical_chat.csv') Chatbot Dataset Topical Chat: https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat
* Data Preprocessing
    *  Data cleaning: extend the abbreviation; remove html, number, emoji, punctuations; transfer all uppercase to lowercase
    *  Munipulate the columns (add / remove), remove the rows have NaN values
* Data Manipulation
    *  Data tokenization, vectorization
    *  Baseline: GloVe (300-dimensional) - glove.6B.300d.txt
    *  New model: use Annoy (Approximate Nearest Neighbors Oh Yeah) to contruct input and target similarity matrices

## Model Construction
* Baseline model:
* New model:


## Techniques
* Raw Data:
* COnstruct similarity 
* Model:


