# Import Packages
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def clean(text):
    text = text.lower().replace('\n', ' ').replace('-', ' ').replace(':', ' ') \
        .replace(',', '').replace('"', '').replace("...", ".").replace("..", ".") \
        .replace("!", ".").replace("?", "").replace(";", ".").replace(":", " ")

    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii", 'ignore')

    text = " ".join(text.split())

    # more clean
    text = text.replace(r'&amp;?', r'and')
    text = text.replace(r'&lt;', r'<')
    text = text.replace(r'&gt;', r'>')

    # clean url http://x.x.x.x/xxx
    text = re.sub(r'http(s)?:\/\/([\w\.\/])*', ' ', text)
    # clean numbers
    text = re.sub('[0-9]+', ' ', text)
    # clean special char
    text = re.sub(r'[!"#$%&()*+,-./:;=?@\\^_`"~\t\n\<\>\[\]\{\}]', ' ', text)
    # convert multiple conitues blank char to one blank char
    text = re.sub(r' +', ' ', text)

    return text


# https://www.kaggle.com/code/andreshg/nlp-glove-bert-tf-idf-lstm-explained
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


# https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def process(text):
    # Clean puntuation, urls, and so on
    text = clean(text)
    # Remove stopwords: is it necessary to remove stop word? (try)
    text = remove_stopwords(text)
    # Remove emojis
    text = remove_emoji(text)

    return text.strip()