import tensorflow as tf
import pdb


WORD_EMBED_SIZE = 10
NUM_CLASSES = 2
VOCAB_SIZE = 42



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

#final_counts, word_ids, docs = batch.process({'../data/train_pos','../data/train_neg'})
#int_docs = batch.convert_docs_to_ints(word_ids, docs)
def get_batch():
    pass



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
    return tf.nn.max_pool(x, ksize=[1, WORD_EMBED_SIZE, k, 1], strides=[1, 1, k, 1], padding='SAME')
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

word_embedding_matrix= noisy_weight_variable([VOCAB_SIZE, WORD_EMBED_SIZE], uniform_range=[-1,1])

#################################################################
# Network
# NHWC = batch_size, height, width, channel
#################################################################


WORD_CONV_WIDTH = 5
WORD_CONV_CHANNELS = 6
HOWEVER_MANY = -1
K_WIDTH = 4

word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix, x)
transposed_word_embeddings = tf.transpose(word_embeddings, perm=[0, 2, 1])
reshaped_word_embeddings = tf.expand_dims(transposed_word_embeddings, 3)

word_feature_maps = noisy_weight_variable([WORD_EMBED_SIZE, WORD_CONV_WIDTH, 1, WORD_CONV_CHANNELS])
word_conv_output = conv2d(reshaped_word_embeddings, word_feature_maps)
pooling_output = k_max_pool(word_conv_output, K_WIDTH)
tanh_output = tf.tanh(pooling_output)


# TO FUTURE US:
# we need the output of layer 1 to be a 2D tensor of shape [num_sentences, sentence_embedding_size]
# then do layer 2 + softmax



# [HOWEVER_MANY, WORD_EMBED_SIZE, WORD_CONV_WIDTH, WORD_CONV_CHANNELS]



W_conv1 = noisy_weight_variable([5, 5, 1, 32])





'''
weights = noisy_weight_variable([])
biases = noisy_bias_variable([NUM_CLASSES])
'''

"""
~Questions~:
- loss function for sentiment
- A 3D tensorof 2D tensors where each tensor is a sentence
- Per document sentence length padding

- How does conv work ?
- How does k max pooling ?

"""
