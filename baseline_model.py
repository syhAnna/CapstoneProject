import os
import tensorflow as tf
from datetime import datetime
from utils import loadData, saveData, word_id_mapping, timer, plot_loss
from utils import EMBED_DIM, BATCH_SIZE, EPOCHS, UNITS, BASELINE_ROOT, EPOCHS_PER_CHECKPOINT
from baseline_enc_dec import Encoder, Decoder
from baseline_inference import predict
from nltk.translate.bleu_score import corpus_bleu

LEARNING_RATE = 0.001

qa_tokenizer = loadData(BASELINE_ROOT + '/qa_tokenizer.pickle')
qa_vocab_len = len(qa_tokenizer.word_index) + 1
word2id, id2word = word_id_mapping(qa_tokenizer)

enc_train, enc_val, enc_test = loadData(BASELINE_ROOT + '/enc_train.pickle'), \
                               loadData(BASELINE_ROOT + '/enc_val.pickle'), \
                               loadData(BASELINE_ROOT + '/enc_test.pickle')
dec_train, dec_val, dec_test = loadData(BASELINE_ROOT + '/dec_train.pickle'), \
                               loadData(BASELINE_ROOT + '/dec_val.pickle'), \
                               loadData(BASELINE_ROOT + '/dec_test.pickle')


######################
# Create the dataset #
######################
def create_dataset(enc_dataset, dec_dataset):
    steps_per_epoch = len(enc_dataset) // BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((enc_dataset, dec_dataset)).shuffle(len(enc_dataset))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset, steps_per_epoch


dataset_train, steps_per_epoch_train = create_dataset(enc_train, dec_train)
dataset_val, steps_per_epoch_val = create_dataset(enc_val, dec_val)
sample_input_batch, sample_target_batch = next(iter(dataset_train))


#######################
# Encoder and Decoder #
#######################
encoder = Encoder(qa_vocab_len, EMBED_DIM, UNITS, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_state_h, sample_state_c = encoder(sample_input_batch, sample_hidden)

decoder = Decoder(qa_vocab_len, EMBED_DIM, UNITS, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)


##############################
# Loss func, optimizer & log #
##############################
def loss_func(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_obj(target, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# checkpoints
checkpoint_dir = BASELINE_ROOT + '/baseline_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# logging
if not os.path.exists('baseline_logs'):
    os.makedirs('baseline_logs')
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'baseline_logs/gradient/' + current_time + '/baseline'
summary_writer = tf.summary.create_file_writer(log_dir)

#################
# Training Step #
#################
@tf.function
def train_step(input, target, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_outputs = encoder(input, enc_hidden)
        enc_output, enc_states = enc_outputs[0], enc_outputs[1:]
        dec_state_h, dec_state_c = enc_states
        dec_input = tf.expand_dims([word2id['<bos>']] * BATCH_SIZE, 1)

        # Teacher forcing
        for t in range(1, target.shape[1]):
            # enc_output -> decoder
            pred, dec_state_h, dec_state_c = decoder(dec_input, (dec_state_h, dec_state_c), enc_output)
            loss += loss_func(target[:, t], pred)
            # teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)
    batch_loss = (loss / int(target.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def train(EPOCHS):
    loss_train, loss_val = list(), list()

    for epoch in range(EPOCHS):
        # Training
        start_time = timer()
        print("###### Training ######")
        enc_hidden_train = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (input, target)) in enumerate(dataset_train.take(steps_per_epoch_train)):
            batch_loss = train_step(input, target, enc_hidden_train)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch_train))

        temp_loss_train = total_loss / steps_per_epoch_train
        loss_train.append(temp_loss_train)
        with summary_writer.as_default():
            tf.summary.scalar('loss', temp_loss_train, step=epoch)
        timer(start_time)

        # Validation
        start_time = timer()
        print("###### Validation ######")
        enc_hidden_val = encoder.initialize_hidden_state()
        total_loss_val = 0
        for (batch, (input, target)) in enumerate(dataset_val.take(steps_per_epoch_val)):
            batch_loss = train_step(input, target, enc_hidden_val)
            total_loss_val += batch_loss
            if batch % 40 == 0:
                print('Epoch {} Batch {} Val-Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        print('Epoch {} Val-Loss {:.4f}'.format(epoch + 1, total_loss_val / steps_per_epoch_val))

        # saving checkpoint every EPOCHS_PER_CHECKPOINT
        if (epoch + 1) % EPOCHS_PER_CHECKPOINT == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        temp_loss_val = total_loss_val / steps_per_epoch_val
        loss_val.append(temp_loss_val)
        with summary_writer.as_default():
            tf.summary.scalar('val-loss', temp_loss_val, step=epoch)

        if epoch == EPOCHS - 1:
            for (batch, (input, target)) in enumerate(dataset_val.take(1)):
                tf.summary.trace_on(graph=True, profiler=True)
                batch_loss = train_step(input, target, enc_hidden_val)
                with summary_writer.as_default():
                    tf.summary.trace_export(name="step", step=0, profiler_outdir=log_dir)
        timer(start_time)
    return loss_train, loss_val


start_time = timer()
loss_train, loss_val = train(EPOCHS)
timer(start_time)

# save model and weights
encoder.save_weights(BASELINE_ROOT + '/baseline_encoder_weights.h5')
decoder.save_weights(BASELINE_ROOT + '/baseline_decoder_weights.h5')
# plot
plot_loss(loss_train, loss_val, EPOCHS)
# restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

############
# Evaluate #
############
# prediction
start_time = timer()
pred_ans_lst, true_ans_lst = predict(encoder, decoder, enc_test, dec_test)
timer(start_time)
# Calculate BLEU Score
start_time = timer()
print('BLEU-1: %f' % corpus_bleu(true_ans_lst, pred_ans_lst, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(true_ans_lst, pred_ans_lst, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(true_ans_lst, pred_ans_lst, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(true_ans_lst, pred_ans_lst, weights=(0.25, 0.25, 0.25, 0.25)))
timer(start_time)

# save pred_ans_lst and true_ans_lst
saveData(pred_ans_lst, BASELINE_ROOT + '/pred_ans_lst.pickle')
saveData(true_ans_lst, BASELINE_ROOT + '/true_ans_lst.pickle')




