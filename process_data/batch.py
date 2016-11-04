"""
Steps to read in file with one review per line (i.e one document per line)

- Remove HTML markup
- Break list of docs into list of sentences and each sentence into list of words
- Map any numeric token in text to "NUMBER"
- Map any symbol token that is not .!? to "SYMBOL"
- Map any < 5 frequency word token to "UNKNOWN"
- We should have 29493 vocab words by the end of this cleaning

"""
import nltk
import tensorflow as tf
import sys
import re

def read_data(file_name):
    with open(file_name) as f:
        docs = f.readlines()
    return docs


"""
Args:
    docs - list of strings to clean

Returns
    html_free_docs - list of strings devoid of html markup
"""
def strip_html(docs):
    return [clean_html(doc) for doc in docs]





def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()



if __name__=='__main__':
    if len(sys.argv) <= 1:
        sys.exit("Usage: python3 batch.py <file_name>")
    file_name = sys.argv[1]
    docs = read_data(file_name)
    html_free_docs = strip_html(docs)
