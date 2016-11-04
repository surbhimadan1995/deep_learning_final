'''
Steps to read in file with one review per line (i.e one document per line)

- Remove HTML markup
- Break list of docs into list of sentences and each sentence into list of words
- Map any numeric token in text to 'NUMBER'
- Map any symbol token that is not .!? to 'SYMBOL'
- Map any < 5 frequency word token to 'UNKNOWN'
- We should have 29493 vocab words by the end of this cleaning

'''
import nltk
import sys
import re
from collections import defaultdict

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


def get_initial_counts(doc, counts):
    for word in doc:
        counts[word] += 1

'''
Args:
    docs - list of strings to clean

Returns
    html_free_docs - list of strings devoid of html markup
'''
def strip_html(docs):
    return [clean_html(doc) for doc in docs]


def clean_html(html):
    '''
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    '''

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
    tokenized_docs - list of documents where each document is a list of tokenized words

Returns:
    (list of documents, dictionary of words to # occurrences)
        * where each document is a list of sentences
        * where each sentence is a list of words
'''
def final_clean(initial_counts, tokenized_docs):
    final_counts = defaultdict(lambda: 0)
    documents_to_return = []
    for doc in tokenized_docs:
        sentence_to_return = [STOP] # can get rid of stop, just make this not contain a STOP
        for word in doc:
            if word.isdigit():
                word = NUMBER
            elif word in SYMBOL_LIST:
                word = SYMBOL
            elif initial_counts[word] < 5:
                word = UNKNOWN

            final_counts[word] += 1
            sentence_to_return.append(word)
            if word in PUNCTUATION_LIST:
                sentence_to_return.append(STOP) # can get rid of stop, just make this not add a STOP
                documents_to_return.append(sentence_to_return)
                sentence_to_return = [STOP] # can get rid of stop, just make this not contain a STOP

    # print(len(final_counts)) vocab size is 29324
    return (documents_to_return, final_counts)


'''
Args:
    file_names - list of file names where each file contains movie reviews on each line

Returns:
    (list of documents, dictionary of words to # occurrences)
        * where each document is a list of sentences
        * where each sentence is a list of words
'''
def process(file_names):
    docs = []
    for file_name in file_names:
        docs  = docs + read_data(file_name)
    html_free_docs = strip_html(docs)
    tokenized_docs = [basic_tokenizer(review) for review in html_free_docs]
    initial_counts = defaultdict(lambda: 0)
    [get_initial_counts(doc, initial_counts) for doc in tokenized_docs]
    return final_clean(initial_counts, tokenized_docs)

if __name__=='__main__':
    if len(sys.argv) <= 1:
<<<<<<< HEAD
        sys.exit("Usage: python3 batch.py <file_name>")
    file_name = sys.argv[1]
    docs = read_data(file_name)
    html_free_docs = strip_html(docs)
=======
        sys.exit('Usage: python3 batch.py <file_name> ...')
    file_names = []
    for i in range(1, len(sys.argv)):
        file_names.append(sys.argv[i])
    process(file_names)
>>>>>>> 1d88d16bd6a3f9962999efa4a9cfd62c863399d2
