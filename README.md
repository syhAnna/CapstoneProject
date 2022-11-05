# CapstoneProject: 
UCLA Master Degree Capstone Project

Google Drive link for large data files that cannot be uploaded: https://drive.google.com/drive/folders/19sByFpcVTjR-kuzpwOMMMLZVITVhdFkJ?usp=share_link

### Directory Detail:
    .
    ├── DataPreproDataPreprocess+NewModelDraft.ipynb                    # Jupyter file contains data processing and tarined new model         
    │
    ├── util files
    │   ├── utils.py                                                    # Helper functions for both 
    │   ├── preprocess.py
    │   ├── vectorization.py
    │   └── smart_reply.py
    │
    ├── model                  
    │   ├── Baseline model             
    │   │   ├── baseline_enc_dec.py
    │   │   ├── baseline_model.py
    │   │   └── baseline_inference.py
    │   │    
    │   └── New model
    │       ├── my_model_sim_matrix.py
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
* Files 'data_generation.py' and 'data_preprocessing.py' need to be run on IBM virtial machine with SROM installed
    * IBM Virtual Machine Setup Instruction: VM_setup.md
    * SROM Installation Instruction: srom_installation.md
* Files 'datase_ppl_model.ipynb' and 'ppl_instance_model.ipynb' need to be run on Colab
* Files 'visualization.ipynb' need to be run on IBM Watson Studio


## Data 
* Data Generation
    *  For each dataset run SROM to generate best set of pipelines
    *  Rerun the set of generated pipelines on the test dataset, record the predictions as {0, 1} = {True prediction, False prediction}
* Data Preprocessing
    *  Use generated raw data to construct 3D tensors
* Data Analysis
    *  Visualize data use PCA, NMF, TSNE, etc.
    *  Visualize factors use histogram

## Model Construction
* Fastai Collaborative Filtering
    *  One dataset case: add regularization & instance similarity
    *  Multiple datasets case: add regularization & dataset similarity

## Techniques
* Dataset space: Penn Machine Learning Benchmarks
* Data generation: SROM (AutoClassification)
* Model: Fastai Collaborative Filtering
* Matrix Factorization: PCA, NMF
* Similarity measurement:
    * One dataset: cosine similarity, Euclidean distance
    * Multiple datasets: Canonical Correlation Analysis (CCA) score
* Data visualization: Clustering (TSNE, Dendrogram), Histogram, Linear Regression, Decis

