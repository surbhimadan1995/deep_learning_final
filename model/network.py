print ('=== start ===')

import tensorflow as tf
import pdb
import numpy as np
import batch
import random

WORD_EMBED_SIZE = 10
NUM_CLASSES = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 1

print ('=== cleaning data ===')
print ('=== getting training data ===')
VOCAB_SIZE, _, int_docs, labels, word_ids = batch.get_imdb_data()

print ('=== getting testing data ===')
test_int_docs, test_labels = batch.get_imdb_test_data(word_ids)



test_pairs = list(zip(test_int_docs, test_labels))
random.shuffle(test_pairs)
#pdb.set_trace()
TEST_NUM_BATCHES = len(test_pairs)

train_pairs = list(zip(int_docs, labels))
random.shuffle(train_pairs)
NUM_BATCHES = len(train_pairs)

#################################################################
# Helper functions
#################################################################

def noisy_weight_variable(shape, uniform_range=None):
    if uniform_range:
        initial = tf.random_uniform(shape, minval=uniform_range[0], maxval=uniform_range[1])
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def noisy_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



"""
x should be reshaped to [BATCH_SIZE, X_HEIGHT, X_WIDTH, INPUT_CHANNELS]
w should be shape [FILTER_HEIGHT, FILTER_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS]

where FILTER_WIDTH by FILTER_HEIGHT is the size of the kernel which is EMBED_SIZE by MAX_SENTENCE_LEN

strides can stay as [1,1,1,1]
"""
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


"""
What is the value of k ? k things we want to keep --> but how many ? (a fraction of the total)
"""
def k_max_pool(x, k):
    reshaped_x = tf.transpose(x, perm=[0, 3, 1, 2])
    top_values, _ = tf.nn.top_k(reshaped_x, k, sorted=False)
    pooled_x = tf.transpose(top_values, perm=[0, 2, 3, 1])
    return pooled_x

    # return tf.nn.max_pool(x, ksize=[1, WORD_EMBED_SIZE, k, 1], strides=[1, 1, k, 1], padding='SAME')
    # values, _ = tf.nn.top_k(x, k) # by default, sorted=True
    # return values

#################################################################
# Model setup
#################################################################

# input shape = [num_sentences, max_words_per_sentence, EMBED_SIZE]
# because its X_WIDTH by X_HEIGHT by INPUT_CHANNELS


# x and y have to be of type float32 :( because I think that's the required dtype for conv2d
# throwing shade at tensorflow.

x = tf.placeholder(tf.int32, shape=[None, None])
y = tf.placeholder(tf.float32, shape=[NUM_CLASSES])


word_embedding_matrix = noisy_weight_variable([VOCAB_SIZE, WORD_EMBED_SIZE], uniform_range=[-1,1])

#################################################################
# Network
# NHWC = batch_size, height, width, channel
#################################################################

print ('=== prepping network ===')

CONV_WORD_WIDTH = 5
CONV_WORD_IN_CHANNELS = 1
CONV_WORD_OUT_CHANNELS = 6
HOWEVER_MANY = -1
WORD_K = 4

word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix, x)
word_embeddings = tf.nn.dropout(word_embeddings, 0.4)
transposed_word_embeddings = tf.transpose(word_embeddings, perm=[0, 2, 1])
reshaped_word_embeddings = tf.expand_dims(transposed_word_embeddings, 3)
# shape = [None, 10, None, 1] (NHWC)
word_feature_maps = noisy_weight_variable([WORD_EMBED_SIZE, CONV_WORD_WIDTH, CONV_WORD_IN_CHANNELS, CONV_WORD_OUT_CHANNELS])

word_conv_output = conv2d(reshaped_word_embeddings, word_feature_maps)
word_pooling_output = k_max_pool(word_conv_output, WORD_K)
word_tanh_output = tf.tanh(word_pooling_output)
# shape = [None, WORD_EMBED_SIZE, k, CONV_WORD_OUT_CHANNELS]


SENTENCE_K = 2
layer_1_shape = word_tanh_output.get_shape().as_list()
SENTENCE_EMBED_SIZE = np.prod(layer_1_shape[1:])
CONV_SENTENCE_WIDTH = 5
CONV_SENTENCE_IN_CHANNELS = 1
CONV_SENTENCE_OUT_CHANNELS = 15




reshaped_sentence_embeddings = tf.reshape(word_tanh_output, [HOWEVER_MANY, SENTENCE_EMBED_SIZE])
transposed_sentence_embeddings = tf.transpose(reshaped_sentence_embeddings, perm=[1, 0])
transposed_sentence_embeddings = tf.nn.dropout(transposed_sentence_embeddings, 0.6)
expanded_sentence_embeddings = tf.expand_dims(tf.expand_dims(transposed_sentence_embeddings, 0), 3)



#reshaped_sentence_embeddings = tf.reshape(word_tanh_output, [1, SENTENCE_EMBED_SIZE, HOWEVER_MANY, CONV_SENTENCE_IN_CHANNELS])
# shape = [1, 240, None, 1] (NHWC)
sentence_feature_maps = noisy_weight_variable([SENTENCE_EMBED_SIZE, CONV_SENTENCE_WIDTH, CONV_SENTENCE_IN_CHANNELS, CONV_SENTENCE_OUT_CHANNELS])

sentence_conv_output = conv2d(expanded_sentence_embeddings, sentence_feature_maps)
sentence_pooling_output = k_max_pool(sentence_conv_output, SENTENCE_K)
sentence_tanh_output = tf.tanh(sentence_pooling_output)

# shape = [1, 240, 2, 15]


document_embedding = tf.reshape(sentence_tanh_output, [1, HOWEVER_MANY])
document_embedding = tf.nn.dropout(document_embedding, 1)
document_embedding_size = document_embedding.get_shape()[1].value

W = noisy_weight_variable([document_embedding_size, NUM_CLASSES])
b = noisy_bias_variable([NUM_CLASSES])




probs = tf.nn.softmax(tf.matmul(document_embedding, W) + b)
probs = tf.reshape(probs, [HOWEVER_MANY])

cross_entropy = -tf.reduce_sum(y * tf.log(probs))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
num_correct_predictions = tf.equal(tf.argmax(probs, 0), tf.argmax(y, 0))
accuracy = tf.reduce_mean(tf.cast(num_correct_predictions, tf.float32))

# gradients = tf.train.GradientDescentOptimizer(LEARNING_RATE).compute_gradients(cross_entropy, [vsp])
gradients = tf.gradients(cross_entropy, reshaped_sentence_embeddings)

session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

# train on documents
print ('=== training ===')
print ('training on', NUM_BATCHES, 'batches')

acc_accum = 0
counter = 1

for ep in range(NUM_EPOCHS):
    for i in range(NUM_BATCHES):
        doc, label = train_pairs[i]

        if len(doc) < SENTENCE_K or len(doc[0]) < WORD_K: continue

        train_step.run(feed_dict={x: doc, y: label})
             
        
        print("====== gradient values =====")
        
        ind = tf.argmin(probs, dimension=0)
        index = ind.eval(feed_dict={x: doc, y: label})
        # pdb.set_trace()
        list_ = [0, 0]
        list_[index] = 1

        #pdb.set_trace()
        grad_vals = session.run(gradients, feed_dict={x: doc, y: list_})  

        pdb.set_trace()

        acc_accum += accuracy.eval(feed_dict={x: doc, y: label})
        counter += 1

        if counter % 100 == 0:
            print ('step', counter, 'accuracy:', acc_accum / counter)

print('final train step', counter, 'accuracy:', acc_accum / counter)

print("=== testing ===")
print ('testing on', TEST_NUM_BATCHES, 'batches')
test_acc_accum = 0
test_counter = 1

for i in range(TEST_NUM_BATCHES):
    doc, label = test_pairs[i]
    if len(doc) < SENTENCE_K or len(doc[0]) < WORD_K: continue

    test_acc_accum += session.run([accuracy], feed_dict={x: doc, y: label})[0]
    test_counter += 1
    if test_counter % 100 == 0:
        print ('step', test_counter, 'accuracy:', test_acc_accum / test_counter)

print ('final test step', test_counter, 'accuracy:', test_acc_accum / test_counter)






