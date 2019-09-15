import re 
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
import time

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def load_stop_words(STOP_WORDS_PATH =  "stopwords.txt"):
    '''
    load stop words
    '''
    stopwords = []
    with open(STOP_WORDS_PATH) as file:
        for line in file:
            stopwords.append(line.replace("\n", "").lower())
    return stopwords

def remove_contraction(text):
    '''
    open sentences
    '''
    words = text.split()
    new = []
    for word in words:
        if word in contraction_dict:
            new.append(contraction_dict[word])
        else:
            new.append(word)
    return " ".join(new)

def remove_words_below_n(words, n=3):
    '''
    remove words with length less than n
    '''
    new = []
    for word in words:
        if len(word) >= n:
            new.append(word)
    return new

def remove_stop_words(text):
    '''
    remove stop words
    '''
    words = text.split()
    stopwords = load_stop_words()
    new = []
    for word in words:
        if word not in stopwords:
            new.append(word)
    return new

def lemmatize(words):
    '''
    to perform lemmatization
    '''
    lemmatizer = WordNetLemmatizer() 
    new = []
    for word in words:
        new.append( lemmatizer.lemmatize(word) )
    return " ".join(new)

def stem(words):
    '''
    to do stemming
    '''
    ps = PorterStemmer() 
    new = []
    for word in words:
        new.append( ps.stem(word) )
    return new

def preprocess(text, n=3):
    '''
    cleans text
    '''
    text = text.lower() # conver to lower case characters
    text = remove_contraction(text) # convert 'nt 'st 's to normal forms
    text = re.sub('[^a-z ]', '', text) # remove non-alphabetic characters
    text = remove_stop_words(text) # remove stop words
    text = remove_words_below_n(text, n) # remove words which are of length smaller than 3 characters
    text = stem(text) # perform stemming
    text = lemmatize(text) # perform lemmatizing
    return text


def process_doc(doc,n=3):
    '''
    processes list of text samples
    '''
    start_time = time.time()
    print("Pre-processing...")
    newtexts = []
    for text in doc:
        newtexts.append(preprocess(text, n))
    return newtexts
    print("Completed Pre-processing at time:",time.time()-start_time)


if __name__ == "__main__":
    sample = "I've had many times but this one is a good chan4ge fo4r u ok ? haha"
    out = [sample,sample]
    out = process_doc(out)
    print(out)
