import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import string
import pickle
import time

from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

EMBED_DIM = 300
UNITS = 1024
BATCH_SIZE = 512
EPOCHS = 20
EPOCHS_PER_CHECKPOINT = 5  # saving checkpoint every 5 epochs
BEAM_IDX = 3

BASELINE_ROOT = './BaselineDataFiles'
MODEL_ROOT = './MyModelDataFiles'
BASELINE_DATA = BASELINE_ROOT + '/single_qna.csv'
MODEL_DATA = MODEL_ROOT + '/topical_chat.csv'
PRETRAINED_VEC = "./glove.6B.300d.txt"


#########################################################
# Helper function for timer and data saving and loading #
#########################################################
def timer(start_time=None):
    """
    Measure the block's execution time using the clock
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def saveData(obj, filename_path):
    """
    file_type could be: .csv or .pickle
    """
    file_type = '.' + filename_path.split('.')[-1]
    if file_type == '.csv':
        obj.to_csv(filename_path, index=False)
    if file_type == '.pickle':
        with open(filename_path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Data saved successully.')


def loadData(filename_path):
    """
    file_type could be: .csv or .pickle
    """
    obj = None
    file_type = '.' + filename_path.split('.')[-1]
    if file_type == '.csv':
        obj = pd.read_csv(filename_path)
    if file_type == '.pickle':
        f = open(filename_path, 'rb')
        obj = pickle.load(f)
        f.close()
    return obj


######################################
# Helper functions for data cleaning #
######################################
def extend(text):
    """
    Extend the abbreviations.
    """
    text = re.sub(r"won't", "will not", str(text))
    text = re.sub(r"can\'t", "can not", str(text))

    text = re.sub(r"n\'t", " not", str(text))
    text = re.sub(r"\'re", " are", str(text))
    text = re.sub(r"\'s", " is", str(text))
    text = re.sub(r"\'d", " would", str(text))
    text = re.sub(r"\'ll", " will", str(text))
    text = re.sub(r"\'t", " not", str(text))
    text = re.sub(r"\'ve", " have", str(text))
    text = re.sub(r"\'m", " am", str(text))
    return text


def remove_html(text):
    """
    Removes HTML.
    """
    return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', str(text))


def remove_punctuation_lower(text):
    """
    Remove punctuation and change all letters to lower case
    """
    return ''.join(' ' if c in string.punctuation else c for c in str(text)).lower()


# https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    """
    Remove all emoji.
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def preprocess(corpus):
    """
    Cleans and removes unwanted characters from the corpus.
    """
    preprocessed = list()
    for message in corpus:
        msg = remove_html(message)
        msg = extend(msg)
        msg = remove_punctuation_lower(msg)
        msg = msg.replace('\\r', ' ')
        msg = msg.replace('\\"', ' ')
        msg = msg.replace('\\n', ' ')
        # clean numbers
        msg = re.sub('[0-9]+', ' ', str(msg))
        msg = ' '.join(msg.split())
        preprocessed.append(msg.strip())
    return preprocessed


#######################################################
# Tokenization (Vectorization), Padding and Embedding #
#######################################################
def get_tokenizer(txt):
    """
    Vectorizing a text corpus
    """
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(txt)
    return tokenizer


def encode_pad_seq(tokenizer, length, txt):
    """
    Encode and pad sequences
    """
    # Integer encode sequences
    X = tokenizer.texts_to_sequences(txt)
    # post pad sequences with 0 values
    return pad_sequences(X, maxlen=length, padding='post')


def word_id_mapping(tokenizer):
    """
    Get both word-to-ID and ID-to-word mappings
    """
    vocab = tokenizer.word_index
    word2id, id2word = dict(), dict()
    for k, v in vocab.items():
        word2id[k], id2word[v] = v, k
    return word2id, id2word


def token_embed(tokenizer):
    """
    Use the pre-trained word-embedding vectors: “glove.6B.300d.text”
    Create token-embedding mapping
    """
    pretrained_vec = open(PRETRAINED_VEC)
    embeddings_index = dict()
    for i, line in enumerate(pretrained_vec):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    vocab = tokenizer.word_index
    vocab_size = len(vocab) + 1
    embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


###################################
# Helper functions for inferences #
###################################
def remove_indicator(msg):
    msg = msg.replace('<bos>', '')
    msg = msg.replace('<eos>', '')
    return msg.strip()


######################
# Plotting functions #
######################
def plot_loss(loss_train, loss_val):
    epochs = range(EPOCHS)
    plt.plot(epochs, loss_train, 'b', label='Training loss')
    plt.plot(epochs, loss_val, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Crossentropy loss')
    plt.legend()
    plt.show()


###################
# AutoReply Class #
###################
class AutoReply():
    def __init__(self):
        # a flag to check whether to end the conversation
        self.end_chat = False
        # greet while starting
        self.welcome()

    def welcome(self):
        print("Initializing AutoReply ...")
        # some time to get user ready
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        # give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice([
            "Hey! Nice to meet you",
            "Hello, it's my pleasure meeting you",
            "Hi, let's chat!"
        ])
        print("AutoReply >>  " + greeting)

    def user_input(self):
        # receive input from user
        text = input("User    >> ")
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # turn flag on
            self.end_chat = True
            # a closing comment
            print('AutoReply >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting AutoReply ...')
        else:
            # continue chat, preprocess input text encode the new user input,
            # add the eos_token and return a tensor in Pytorch
            self.input = text

    def bot_response(self):
        response = respond(self.input)
        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        # print bot response
        print('AutoReply >>  ' + response)

    # in case there is no response from model
    def random_response(self):
        response = respond(self.input)
        # iterate over history backwards to find the last token
        while response == '':
            response = respond(self.input)
        # if it is a question, answer suitably
        if response.strip() == '?':
            reply = np.random.choice(["I don't know",
                                      "I am not sure"])
        else:  # not a question? answer suitably
            reply = np.random.choice(["Great",
                                      "Fine. What's up?",
                                      "Okay"])
        return reply
