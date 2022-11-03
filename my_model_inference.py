import tensorflow as tf
import numpy as np
import random
from utils import loadData, remove_indicator, get_tokenizer
from utils import MODEL_ROOT
from tensorflow.keras.preprocessing.sequence import pad_sequences

N_RESPONSE = 5
DBSCAN_FNAME = '/target_dbscan_0008.pickle'
MODEL_FNAME = '/my_lstm_model_0008.hdf5'

input_texts = loadData(MODEL_ROOT + '/input_texts.pickle')
target_texts = loadData(MODEL_ROOT + '/target_texts.pickle')
# load trained model and DBSCAN result
model = tf.keras.models.load_model(MODEL_ROOT + MODEL_FNAME)
db = loadData(MODEL_ROOT + DBSCAN_FNAME)


def get_responses(seed_text, n):
    # print("Input -", seed_text)
    # print("----------------------")
    responses = list()

    input_tokenizer = get_tokenizer(input_texts)
    token_list = input_tokenizer.texts_to_sequences([seed_text])[0]
    max_seq_len = max([len(x) for x in input_tokenizer.texts_to_sequences(input_texts)])
    token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        predictions = model.predict(token_list, verbose=10)
        predicted_indices = predictions.argsort()[0][::-1][:n]
        # print(predicted_indices)

    for predicted_index in predicted_indices:
        score = 0
        if predicted_index == len(set(db.labels_)) - 1:
            # print("Predicting outside clusters")
            predicted_index = -1
        else:
            score = predictions[0][predicted_index]

        # randomly pick 1 index
        possible_response = np.where(db.labels_ == predicted_index)[0]
        response_index = random.sample(possible_response.tolist(), 1)[0]
        responses.append([remove_indicator(target_texts[response_index]).replace("\t", "").replace("\n", ""), score])

    # for i, response in enumerate(responses):
    #     print("Response", (i + 1), "->", response[0], " -> Score :", response[1])

    return responses


# # test predictions
# get_responses(input_texts[74], N_RESPONSE)   # 'hi how are you'
# get_responses(input_texts[19], N_RESPONSE)   # 'do you like comic books'
# get_responses(input_texts[109], N_RESPONSE)  # 'do you know much about the bible'
# get_responses(input_texts[70], N_RESPONSE)   # 'have a good evening'
# get_responses("what is your name", N_RESPONSE)
# get_responses("goodbye", N_RESPONSE)
