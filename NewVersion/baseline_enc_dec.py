import tensorflow as tf
from utils import token_embed, loadData
from utils import BASELINE_ROOT

BASELINE_CLEAN_DATA = BASELINE_ROOT + '/processed_clean_single_qna.csv'
TOKENIZER = BASELINE_ROOT + '/qa_tokenizer.pickle'

baseline_df = loadData(BASELINE_CLEAN_DATA)
MAXLEN_Q = baseline_df['QuestionLength'].max()
MAXLEN_A = baseline_df['AnswerLength'].max() + 2
embedding_matrix = token_embed(loadData(TOKENIZER))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=MAXLEN_Q,
                                                   weights=[embedding_matrix], trainable=False)
        self.lstm = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units)), tf.zeros((self.batch_size, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=MAXLEN_A,
                                                   weights=[embedding_matrix], trainable=False)
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units)), tf.zeros((self.batch_size, self.dec_units))

