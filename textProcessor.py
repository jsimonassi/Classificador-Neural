import string
import unicodedata
import wget as wget
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer


class TextProcessor:
    def __init__(self, text):
        self.text = text

    def process(self):
        self.__change_to_utf8()
        return self.text

    def normalize(self):
        self.__to_lower()
        self.__remove_punctuation()
        self.__remove_whitespace()
        return self.text

    def tokenize(self):
        self.__remove_stopwords()
        self.__lemmatize()
        return self.text

    # Process Methods
    def __change_to_utf8(self):
        self.text = ''.join(ch for ch in unicodedata.normalize('NFKD', self.text))

    # Normalize Methods
    def __remove_punctuation(self):
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))

    def __remove_numbers(self):
        # remove numbers
        self.text = ''.join([i for i in self.text if not i.isdigit()])

    def __remove_whitespace(self):
        # remove whitespace
        self.text = self.text.strip()

    def __to_lower(self):
        # convert to lower case
        self.text = self.text.lower()

    # Tokenize Methods
    def __remove_stopwords(self):
        self.text = ' '.join([word for word in self.text.split() if word not in stopwords.words('portuguese')])

    def __lemmatize(self):
        try:
            open("lemmatization-pt.txt")
        except FileNotFoundError:
            url = "https://raw.githubusercontent.com/michmech/lemmatization-lists/master/lemmatization-pt.txt"
            wget.download(url, 'lemmatization-pt.txt')

        # Convert to dictionary
        lmztpt = {}
        dic = open("lemmatization-pt.txt")
        for line in dic:
            txt = line.split()
            lmztpt[txt[1]] = txt[0]

        return ' '.join([TextProcessor.__portuguese_mess(word, lmztpt, RSLPStemmer()) for word in self.text.split()])

    @staticmethod
    def __portuguese_mess(word, lmztpt, ptstemmer):
        # Lemmatize wherever possible
        if word in lmztpt.keys():
            return lmztpt.get(word)
        else:
            return ptstemmer.stem(word)

