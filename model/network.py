import tensorflow as tf
import batch

fname_pos = '../data/train_pos.txt'

EMBED_SIZE = 50
NUM_CLASSES = 2

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

def get_batch():
    # TODO
    pass
#################################################################
# Model setup
#################################################################

# input shape = [num_sentences, max_words_per_sentence, embedding_length]

x = tf.placeholder(tf.int32, shape=[None, None, EMBED_SIZE])
y = tf.placeholder(tf.int32, shape=[NUM_CLASSES])

e_words = noisy_weight_variable([vocab_size, embedding_size], uniform_range=[-1,1])

w_words = noisy_weight_variable([])
b_words = noisy_bias_variable([EMBED_SIZE])
embd_words = tf.nn.embedding_lookup(e_words, x)





"""
~Questions~:
- loss function for sentiment
- A 3D tensorof 2D tensors where each tensor is a sentence
- Per document sentence length padding

- How does conv work ?
- How does k max pooling ? 









"""





