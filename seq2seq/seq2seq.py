# Import Packages
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# data preprocessing
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.text import Tokenizer
from scipy import spatial

root_path = '/content/drive/MyDrive/CapstoneProject/SmartReply'

file1 = open(root_path + '/input_texts.pickle', 'rb')
input_texts = pickle.load(file1)
file1.close()

file2 = open(root_path + '/target_texts.pickle', 'rb')
target_texts = pickle.load(file2)
file2.close()

file3 = open(root_path + '/input_words_set.pickle', 'rb')
input_words_set = pickle.load(file3)
file3.close()

file4 = open(root_path + '/target_words_set.pickle', 'rb')
target_words_set = pickle.load(file4)
file4.close()

input_words = sorted(list(input_words_set))
target_words = sorted(list(target_words_set))
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])
max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i) for i, word in enumerate(target_words)])

#saving
root_path = '/content/drive/MyDrive/CapstoneProject/SmartReply/seq2seq'
with open(root_path + '/input_token_index.pickle', 'wb') as handle:
    pickle.dump(input_token_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#saving
with open(root_path + '/target_token_index.pickle', 'wb') as handle:
    pickle.dump(target_token_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length), dtype="float32"
) # (86659, 47)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length), dtype="float32"
) # (86659, 9)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
) # (86659, 9, 20902)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.0