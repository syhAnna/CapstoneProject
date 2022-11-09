from utils import saveData, loadData
from utils import get_tokenizer, encode_pad_seq
from sklearn.model_selection import train_test_split

SEED = 42
BASELINE_ROOT = './BaselineDataFiles'
MODEL_ROOT = './MyModelDataFiles'
BASELINE_DATA = BASELINE_ROOT + '/processed_clean_single_qna.csv'
MODEL_INPUT_TXT = MODEL_ROOT + '/input_texts.pickle'
MODEL_TARGET_TXT = MODEL_ROOT + '/target_texts.pickle'

baseline_df = loadData(BASELINE_DATA)
input_texts = loadData(MODEL_INPUT_TXT)
target_texts = loadData(MODEL_TARGET_TXT)


###########################
# Vectorize baseline data #
###########################
# X = (Question, QA), y = Answer; train:validation = 8:2
enc_train, enc_test, qa_train, qa_test, dec_train, dec_test = train_test_split(baseline_df['Question'],
                                                                               baseline_df['QA'],
                                                                               baseline_df['Answer'],
                                                                               test_size=0.005, random_state=SEED)
enc_train, enc_val, qa_train, qa_val, dec_train, dec_val = train_test_split(enc_train,
                                                                            qa_train,
                                                                            dec_train,
                                                                            test_size=0.2, random_state=SEED)
qa_tokenizer = get_tokenizer(baseline_df['QA'])

enc_train = encode_pad_seq(qa_tokenizer, baseline_df['QuestionLength'].max(), enc_train)
enc_val = encode_pad_seq(qa_tokenizer, baseline_df['QuestionLength'].max(), enc_val)
enc_test = encode_pad_seq(qa_tokenizer, baseline_df['QuestionLength'].max(), enc_test)
# Answer Length + 2 for '<bos>' and '<eos>'
dec_train = encode_pad_seq(qa_tokenizer, baseline_df['AnswerLength'].max()+2, dec_train)
dec_val = encode_pad_seq(qa_tokenizer, baseline_df['AnswerLength'].max()+2, dec_val)
dec_test = encode_pad_seq(qa_tokenizer, baseline_df['AnswerLength'].max()+2, dec_test)

# Save Data
saveData(qa_tokenizer, BASELINE_ROOT + '/qa_tokenizer.pickle')

saveData(enc_train, BASELINE_ROOT + '/enc_train.pickle')
saveData(enc_val, BASELINE_ROOT + '/enc_val.pickle')
saveData(enc_test, BASELINE_ROOT + '/enc_test.pickle')

saveData(dec_train, BASELINE_ROOT + '/dec_train.pickle')
saveData(dec_val, BASELINE_ROOT + '/dec_val.pickle')
saveData(dec_test, BASELINE_ROOT + '/dec_test.pickle')

########################
# Vectorize model data #
########################
input_tokenizer, target_tokenizer = get_tokenizer(input_texts), get_tokenizer(target_texts)
input_tokens, target_tokens = input_tokenizer.texts_to_sequences(input_texts), \
                              target_tokenizer.texts_to_sequences(target_texts)
input_maxlen, target_maxlen = max([len(x) for x in input_tokens]), \
                              max([len(x) for x in target_tokens])
input_sequences = encode_pad_seq(input_tokenizer, input_maxlen, input_texts)
target_sequences = encode_pad_seq(target_tokenizer, target_maxlen, target_texts)

# Save Data
saveData(input_sequences, MODEL_ROOT + '/input_sequences.pickle')
saveData(target_sequences, MODEL_ROOT + '/target_sequences.pickle')





