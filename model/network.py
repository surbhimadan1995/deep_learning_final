import tensorflow as tf
import batch

fname_pos = '../data/train_pos.txt'

embedding_length = 50
sentiment_analysis_options = 2

###############################################################################
#NOTE FOR BEVERLY: we want our input to be of type (((string list) list) list)
# which is in total a list of documents, where each inner list is
# a list of sentences and each inner list is a list of words
# ex.
# The documents "a b c. d. e f." and "g. h i j." would turn into:
# [[["a", "b", "c"], ["d"], ["e", "f"]], [["g"], ["h", "i", "j"]]]
###############################################################################


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

def get_batch()

#################################################################
# Model setup
#################################################################

# input shape = [num_sentences, max_words_per_sentence, embedding_length]

xs = tf.placeholder(tf.int32, shape=[None, None, embedding_length])
ys = tf.placeholder(tf.int32, shape=[sentiment_analysis_options])

Ewords = noisy_weight_variable([vocab_size, embedding_size], uniform_range=[-1,1])

Wwords = noisy_weight_variable([])
bwords = noisy_bias_variable([embedding_size])
embd_words = tf.nn.embedding_lookup(Ewords, xs)


