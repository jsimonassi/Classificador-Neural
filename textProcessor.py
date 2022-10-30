import string
import unicodedata
import wget as wget
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer


class TextProcessor:
    def __init__(self, text):
        self.text = text

    def process(self):
        """Chama todos os métodos privados de processamento de texto."""
        self.__change_to_utf8()
        return self.text

    def normalize(self):
        """Chama todos os métodos privados de normalização de texto."""
        self.__to_lower()
        self.__remove_punctuation()
        self.__remove_whitespace()
        return self.text

    def tokenize(self):
        """Chama todos os métodos privados de tokenização de palavras"""
        self.__remove_stopwords()
        self.__lemmatize()
        return self.text

    # Process Methods
    def __change_to_utf8(self):
        """Converte o texto para UTF-8"""
        self.text = ''.join(ch for ch in unicodedata.normalize('NFKD', self.text))

    # Normalize Methods
    def __remove_punctuation(self):
        """Remove pontuação"""
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))

    def __remove_numbers(self):
        """Remove números"""
        self.text = ''.join([i for i in self.text if not i.isdigit()])

    def __remove_whitespace(self):
        """Remove espaços em branco"""
        self.text = self.text.strip()

    def __to_lower(self):
        """Converte o texto para minúsculo"""
        self.text = self.text.lower()

    # Tokenize Methods
    def __remove_stopwords(self):
        """Remove as stopwords"""
        self.text = ' '.join([word for word in self.text.split() if word not in stopwords.words('portuguese')])

    def __lemmatize(self):
        """Realiza a lematização com base no arquivo txt.
        Implementado desta forma pois não encontrei na lib uma solução em pt-br"""
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
        """Função auxiliar que realiza a lematização, se a palavra estiver no dicionário.
        Caso contrário, realiza o stemming"""
        if word in lmztpt.keys():
            return lmztpt.get(word)
        else:
            return ptstemmer.stem(word)

