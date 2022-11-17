import re
import joblib
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def contractions(s):
    s = re.sub(r"won't", "will not", s)
    s = re.sub(r"would't", "would not", s)
    s = re.sub(r"could't", "could not", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"can\'t", "can not", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    return s


def purifier(x):
    x = " ".join(x.lower() for x in str(x).split())
    # x = BeautifulSoup(x, features="html.parser").get_text() #To be used for removing HTML Tags and Text Extraction
    # x = re.sub(r"http\S+", "", x)                           #Remove URLs from reviews
    x = contractions(x)
    x = " ".join([re.sub("[^A-Za-z]+","", x) for x in nltk.word_tokenize(x)])
    x = re.sub(" +", " ", x)
    x = " ".join([x for x in x.split() if x not in stop])
    x = " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)])
    return x


loaded_model = joblib.load("review classifier model.joblib")
vectorizer = loaded_model[0]


def sentiment_func(reviews=[""]):   #You can send multiple reviews here in a list
    x_test_1 = [purifier(i) for i in reviews]
    tf_x_test = vectorizer.transform(x_test_1)
    y_test_pred = loaded_model[1].predict(tf_x_test)
    return ["+ve" if i == 1 else "-ve" for i in y_test_pred]

if __name__ == "__main__":
    text = input("\nEnter text:\n")
    print(sentiment_func([text]))

