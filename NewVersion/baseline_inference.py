import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import loadData, word_id_mapping, remove_indicator
from utils import UNITS, BEAM_IDX, BASELINE_ROOT
from baseline_enc_dec import MAXLEN_Q, MAXLEN_A

qa_tokenizer = loadData(BASELINE_ROOT + '/qa_tokenizer.pickle')
qa_vocab_len = len(qa_tokenizer.word_index) + 1
word2id, id2word = word_id_mapping(qa_tokenizer)


def beam_search(inputs, encoder, decoder, max_length_inp, max_length_target, special_input=False, beam_index=BEAM_IDX):
    """
    Beam Search
    inputs (type, shape) = (np.array, (1, max_length_inp))
    """
    if special_input:
        inputs = tf.zeros((1, max_length_inp), dtype=tf.dtypes.int32)
        sentence = 'All zeros'
    else:
        sentence = ''
        for i in inputs[0]:
            if i == 0:  # post padding
                break
            sentence += (id2word[i] + ' ')
        inputs = tf.convert_to_tensor(inputs)

    start = [word2id['<bos>']]
    result = [[start, 0.0]]
    enc_hidden = (tf.zeros((1, UNITS)), tf.zeros((1, UNITS)))
    enc_outputs = encoder(inputs, enc_hidden)
    enc_output, enc_states = enc_outputs[0], enc_outputs[1:]
    dec_state_h, dec_state_c = enc_states
    dec_input = tf.expand_dims([word2id['<bos>']], 0)

    while len(result[0][0]) < (max_length_target - 1):  # '<bos>' is already added
        temp = []
        for s in result:
            predictions, dec_state_h, dec_state_c = decoder(dec_input, (dec_state_h, dec_state_c), enc_output)
            for w in np.argsort(predictions[0])[-beam_index:]:  # Get the top BEAM_IDX preds
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += predictions[0][w]
                temp.append([next_cap, prob])
        result = temp
        # Sorting according to the probabilities
        result = sorted(result, reverse=False, key=lambda l: l[1])
        result = result[-beam_index:]
        predicted_id = result[-1] # max prob
        prd_id = predicted_id[0][-1]
        if prd_id != word2id['<eos>']:
            # Decoder input = word predicted with highest prob
            dec_input = tf.expand_dims([prd_id], 0)
        else:
            break

    result = result[-1][0]
    final_result = []
    for i in [id2word[i] for i in result]:
        if i != '<eos>':
            final_result.append(i)
        else:
            break

    final_result = ' '.join(final_result[1:])
    return final_result, sentence


def predict(encoder, decoder, dataset_input, dataset_output, special_input=False):
    start, end = 0, len(dataset_input)
    pred_ans_lst, true_ans_lst = list(), list()
  
    for j in tqdm(range(start, end)):
        actual_ans = ''
        input, output = dataset_input[j], dataset_output[j]
        input = np.expand_dims(input, 0)
        pred_ans, _ = beam_search(input, encoder, decoder, MAXLEN_Q, MAXLEN_A, special_input=special_input)
        for i in output:
            if i == 0:  # post padding
                break
            actual_ans += (id2word[i] + ' ')
        hypo = remove_indicator(pred_ans).split(' ')
        reference = remove_indicator(actual_ans).split(' ')
        references = [reference]  # list of references for 1 sentence
        true_ans_lst.append(references)  # list of references for all sentences in corpus
        pred_ans_lst.append(hypo)  # list of hypo corresponds to list of references
    return pred_ans_lst, true_ans_lst
