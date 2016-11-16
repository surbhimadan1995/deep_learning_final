'''
Steps to read in file with one review per line (i.e one document per line)

- Remove HTML markup
- Break list of docs into list of sentences and each sentence into list of words
- Map any numeric token in text to 'NUMBER'
- Map any symbol token that is not .!? to 'SYMBOL'
- Map any < 5 frequency word token to 'UNKNOWN'
- We should have 29493 vocab words by the end of this cleaning

'''
import sys
import re
from collections import defaultdict
import pdb
import numpy as np

'''
Returns list of lines for the file passed in
'''
def read_data(file_name):
    with open(file_name) as f:
        docs = f.readlines()
    return docs

def basic_tokenizer(sentence, word_split=re.compile("([.,!?\"':;)(])")):
    '''
    Source: Sidd, Piazza
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    '''
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]


'''
Args:
    list of docs where each doc is a list of tokens (i.e. strings)

Returns:
    list of docs
        * where each doc is a list of sentences split at either token .?!
        * where each sentence is a list of word tokens
'''
def split_docs(tokenized_docs, delim=re.compile("\.?\??!?")):
    final_docs = []
    for doc in tokenized_docs:
        #pdb.set_trace()
        split_doc = re.split(delim, doc)
        split_doc = [basic_tokenizer(s) for s in split_doc]
        #split_doc = [[w.lower() for w in sentence.strip().split()] for sentence in split_doc]
        final_docs.append(split_doc)
    return final_docs


'''
Gets initial word-freq map for the docs passed in
'''
def get_word_counts(docs):
    counts = defaultdict(lambda:0)
    for doc in docs:
        for sentence in doc:
            for word in sentence:
                counts[word] += 1
    return counts


'''
Args:
    docs - list of strings to clean

Returns
    html_free_docs - list of strings devoid of html markup
'''
def strip_html(docs):
    return [clean_html(doc) for doc in docs]


'''
Copied from NLTK package.
Remove HTML markup from the given string.
Args:
    html: the stiring to be cleaned of HTML markup

Returns:
    cleaned string
'''
def clean_html(html):
    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', '', html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r'(?s)<!--(.*?)-->[\n]?', '', cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r'(?s)<.*?>', ' ', cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r'&nbsp;', ' ', cleaned)
    cleaned = re.sub(r'  ', ' ', cleaned)
    cleaned = re.sub(r'  ', ' ', cleaned)
    return cleaned.strip()

NUMBER = 'NUMBER'
SYMBOL = 'SYMBOL'
STOP = 'STOP'
UNKNOWN = 'UNKNOWN'
PUNCTUATION_LIST = ['.', '!', '?']
SYMBOL_LIST = ['(', '[', ',', '\"', '\'', ':', ';', ')', ']']


'''
Args:
    file_names - list of file names where each file contains movie reviews on each line

Returns:
    (final_counts, word_ids, final_docs)
        * where final_counts is final word freq including unknowmn, symbol, number
        * where word_ids are umappings of words to unique ints
        * where final_docs are the same format as tokenized docs but including
        unknown, symbol, number and STOPs
'''
def process(file_names):
    docs = []
    for name in file_names:
        docs = docs + read_data(name)
    html_free_docs = strip_html(docs)
    tokenized_docs = split_docs(html_free_docs)
    counts = get_word_counts(tokenized_docs)
    return clean(counts, tokenized_docs)

'''
Args:
    counts: initial word frequencies
    list of documents
        * where each doc is a list of sentences
        * where each sentence is a list of words

Returns:
    (final_counts, word_ids, final_docs)
    * where final_counts is final word freq including unknowmn, symbol, number
    * where word_ids are umappings of words to unique ints
    * where final_docs are the same format as tokenized docs but including
        unknown, symbol, number and STOPs
'''
def clean(counts, tokenized_docs):
    final_docs = []
    final_counts = defaultdict(lambda:0)
    word_ids = {}
    curr_id = 1

    for doc in tokenized_docs:
        final_doc = []
        for sentence in doc:
            final_sentence = [STOP]
            for word in sentence:
                if word.isdigit():
                    word = NUMBER
                elif word in SYMBOL_LIST:
                    word = SYMBOL
                elif counts[word] < 5:
                    word = UNKNOWN
                if word not in word_ids:
                    word_ids[word] = curr_id
                    curr_id += 1
                final_counts[word] += 1
                final_sentence.append(word)
            final_sentence.append(STOP)
            final_doc.append(final_sentence)
        final_docs.append(final_doc)
    return (final_counts, word_ids, final_docs)


'''
Args:
    word_ids: map of words to unique integer ids
    list of documents
        * where each document is a list of sentences
        * where each sentence is a list of words

Returns:
     list of documents
        * where each document is a list of sentences
        * where each sentence is a list of words in their integer form
'''
def convert_docs_to_ints(word_ids, docs):
    STOP = 0
    PADDING_ID = 0 # which is also a stop

    int_docs = []
    for doc in docs:
        max_sentence_length = max(len(sentence) for sentence in doc)
        int_doc = []
        for sentence in doc:
            int_sentence = []
            for word in sentence:
                if word in word_ids:
                    int_sentence.append(word_ids[word])
                else:
                    int_sentence.append(STOP)
            int_sentence += [PADDING_ID] * (max_sentence_length - len(int_sentence))
            int_doc.append(int_sentence)
        int_docs.append(int_doc)

    return int_docs

'''
    gets imdb movie review data
    returns vocab size, documents (words), int_docs (words in int form), labels (of one-hot vectors)
        ** labels:  [0, 1] represents positive review
                    [1, 0] represents negative review
'''
POS_REVIEWS_TRAIN = '../data/train_pos'
NEG_REVIEWS_TRAIN = '../data/train_neg'
# POS_REVIEWS_TRAIN = '../data/beverly_pos'
# NEG_REVIEWS_TRAIN = '../data/beverly_neg'
def get_imdb_data():
    positive = read_data(POS_REVIEWS_TRAIN)
    negative = read_data(NEG_REVIEWS_TRAIN)

    # [0, 1] represents positive review, [1, 0] represents negative review
    labels = [[0, 1]] * len(positive) + [[1, 0]] * len(negative)

    final_counts, word_ids, docs = process([POS_REVIEWS_TRAIN,NEG_REVIEWS_TRAIN])
    int_docs = convert_docs_to_ints(word_ids, docs)
    # add 1 to len(word_ids) to account for STOP
    return len(word_ids)+1, docs, int_docs, labels
'''
POS_REVIEWS_TEST = '../data/test_pos'
NEG_REVIEWS_TEST = '../data/test_neg'
def get_imdb_test_data(word_ids):
    positive = read_data(POS_REVIEWS_TEST)
    negative = read_data(NEG_REVIEWS_TEST)

    # [0, 1] represents positive review, [1, 0] represents negative review
    labels = [[0, 1]] * len(positive) + [[1, 0]] * len(negative)

    docs = positive + negative

    html_free_docs = strip_html(docs)
    tokenized_docs = split_docs(html_free_docs)
    int_docs = convert_docs_to_ints(word_ids, tokenized_docs)

    return int_docs, labels
'''
if __name__=='__main__':
    '''
    if len(sys.argv) <= 1:
        sys.exit('Usage: python3 batch.py <file_name> ...')
    file_names = []
    for i in range(1, len(sys.argv)):
        file_names.append(sys.argv[i])
    final_counts, word_ids, docs = process(file_names)
    int_docs = convert_docs_to_ints(word_ids, docs)
    '''
    vocab_size, docs, int_docs, labels, word_ids = get_imdb_data()
    test_int_docs, test_labels = get_imdb_test_data(word_ids)
    pdb.set_trace()
