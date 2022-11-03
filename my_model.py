import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from utils import loadData, get_tokenizer, encode_pad_seq
from utils import MODEL_ROOT

EMBEDDING_DIM = 512
DROP_RATE = 0.2
EPOCH = 25
LEARNING_RATE = 0.001
DBSCAN_FNAME = '/target_dbscan_0008.pickle'


def get_one_hot_labels(db):
    """
    According to the clustering result from DBSCAN to
    encode categorical features as a one-hot numeric array
    """
    labels = np.array(db.labels_, copy=True)
    labels[labels == -1] = len(set(db.labels_)) - 1
    encoder_labels = OneHotEncoder().fit(labels.reshape(-1, 1))
    return encoder_labels.transform(labels.reshape(-1, 1))


def create_model(max_seq_len, total_words, one_hot_labels):
    """
    Create the LSTM model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(total_words, EMBEDDING_DIM, input_length=max_seq_len),
            tf.keras.layers.LSTM(EMBEDDING_DIM),
            tf.keras.layers.Dropout(DROP_RATE),
            tf.keras.layers.Dense(one_hot_labels.shape[1], activation='softmax')
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


if __name__ == '__main__':
    # read out data
    input_texts = loadData(MODEL_ROOT + '/input_texts.pickle')
    dbscan = loadData(MODEL_ROOT + DBSCAN_FNAME)
    # get one-hot encoded labels
    one_hot_labels = get_one_hot_labels(dbscan)

    #################################
    # Construct and train the model #
    #################################
    input_tokenizer = get_tokenizer(input_texts)
    max_seq_len = max([len(x) for x in input_tokenizer.texts_to_sequences(input_texts)])
    input_seq = np.array(encode_pad_seq(input_tokenizer, max_seq_len, input_texts))

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        all_words = len(input_tokenizer.word_index) + 1
        model = create_model(max_seq_len, all_words, one_hot_labels)
        model.summary()
    history = model.fit(input_seq, one_hot_labels.todense(), epochs=EPOCH, verbose=1)
    # Save model
    model.save(MODEL_ROOT + "/my_lstm_model_0008.hdf5")

    ##############################
    # plot the loss and accuracy #
    ##############################
    fig, ax1 = plt.subplots()
    color1, color2 = 'r', 'b'
    ax1.set_xlabel('Epoch')

    ax1.set_ylabel('accuracy', color=color1)
    ax1.plot(range(EPOCH), history.history['accuracy'], color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('loss', color=color2)
    ax2.plot(range(EPOCH), history.history['loss'], color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.show()
