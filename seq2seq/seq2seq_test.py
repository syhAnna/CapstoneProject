# Import Packages
import numpy as np
import pickle
from keras.models import load_model
from math import log
from process import process

"""
Read out data and model from files
"""
root_path = '/Users/yuhan/Desktop/project/CapstoneProject/processed_data'
file1 = open(root_path + '/input_texts.pickle', 'rb')
input_texts = pickle.load(file1)
file1.close()
max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])

root_path = '/Users/yuhan/Desktop/project/CapstoneProject/seq2seq'
file2 = open(root_path + '/input_token_index.pickle', 'rb')
input_token_index = pickle.load(file2)
file2.close()

file3 = open(root_path + '/target_token_index.pickle', 'rb')
target_token_index = pickle.load(file3)
file3.close()
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# read out saved model
root_path = '/Users/yuhan/Desktop/project/CapstoneProject/seq2seq/models'
encoder_model = load_model(root_path + "/encoder_model.hdf5")
decoder_model = load_model(root_path + "/decoder_model.hdf5")


def respond(text):
    input_seq = np.zeros(
        (1, max_encoder_seq_length), dtype="float32"
    )

    for t, word in enumerate(text.split()):
        input_seq[0, t] = input_token_index[word]

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['bos']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'eos' or
                len(decoded_sentence) > 50):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [h, c]
    return decoded_sentence


# test
print("Q1: how are you\nA1: ", respond("how are you"))
print("Q2: good morning\nA2: ", respond("good morning"))
print("Q3: good bye\nA3: ", respond("good bye"))

"""
Try more test
"""
for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    decoded_sentence = respond(input_texts[seq_index])
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

eos_token = target_token_index['eos']
print('EOS token: ', eos_token)


# """
# Generate beam text
# """
# def generate_beam_text(seed_text, next_words, beam_search_n, break_at_eos):
#     distributions_scores_states = [[list(), 0.0, [None, None]]]
#     decoder_states_value = None
#     for _ in range(next_words):
#         sequence_temp_candidates = list()
#         for i in range(len(distributions_scores_states)):
#             input_seq = np.zeros(
#                 (1, max_encoder_seq_length), dtype="float32"
#             )
#             # Generate empty target sequence of length 1.
#             target_seq = np.zeros((1, 1))
#             seq, score, states_values = distributions_scores_states[i]
#
#             if len(distributions_scores_states) == 1:
#                 for t, word in enumerate(process(seed_text).split()):
#                     input_seq[0, t] = input_token_index[word]
#
#                 # Encode the input as state vectors.
#                 decoder_states_value = encoder_model.predict(input_seq)
#                 # Populate the first character of target sequence with the start character.
#                 target_seq[0, 0] = target_token_index['bos']
#             else:
#                 target_seq[0, 0] = seq[-1]
#                 decoder_states_value = states_values
#
#                 candidate_sentence = ""
#                 for token_index in seq:
#                     if token_index == eos_token:
#                         break
#
#                     word = reverse_target_char_index[token_index]
#                     candidate_sentence += word + " "
#                 print("score :", score, " | ", candidate_sentence)
#
#             output_tokens_distribution, h, c = decoder_model.predict([target_seq] + decoder_states_value)
#
#             # Update states
#             decoder_states_value = [h, c]
#             predicted_distribution = output_tokens_distribution[0][0]
#
#             for j in range(len(predicted_distribution)):
#                 if predicted_distribution[j] > 0:
#                     candidate = [seq + [j], score - log(predicted_distribution[j]), decoder_states_value]
#                     if break_at_eos and j == eos_token:
#                         continue
#                     else:
#                         sequence_temp_candidates.append(candidate)
#
#         # 2. score and sort all candidates
#         ordered = sorted(sequence_temp_candidates, key=lambda tup: tup[1])
#         distributions_scores_states = ordered[:beam_search_n]
#         print("-----")
#
#
# """
# Test the generate beam text function
# """
# generate_beam_text("i wonder if they met how that would go from there", 5, 5, False)
# generate_beam_text("do you like comic books", 4, 5, False)
# generate_beam_text("thanks", 5, 5, False)
# generate_beam_text("hi do you like to dance", 5, 5, False)


