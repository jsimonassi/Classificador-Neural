from bagOfWords import BagOfWords
from dataReader import DataReader
from textProcessor import TextProcessor
from neuralNetowrkManager import NeuralNetworkManager
import nltk

# Coluna de polaridade da tabela de dataset
POLARITY_COLUMN = 4
# Número de linhas que serão lidas do arquivo csv
LINES_TO_READ = 100

if __name__ == '__main__':

    # Faz o download dos conteúdos utilizados pela biblioteca NLTK
    nltk.download('rslp')
    nltk.download('stopwords')
    nltk.download('punkt')

    # Lê o arquivo de dados
    dataset = DataReader('./DataSet/b2w.csv').read()
    if dataset.empty:
        print('Aborting')
        exit(1)

    # Processa o texto
    processed_data = []
    for i in range(LINES_TO_READ):
        processed_text = TextProcessor(dataset.iloc[i, 1]).process()
        normalized_text = TextProcessor(processed_text).normalize()
        tokenized_text = TextProcessor(normalized_text).tokenize()
        processed_data.append(tokenized_text)

    # Separa os dados utilizando a técnica de Bag of Words
    x, y = BagOfWords.get_xy_data(dataset, processed_data, POLARITY_COLUMN, LINES_TO_READ)
    # Cria o objeto de rede neural
    my_network = NeuralNetworkManager(x, y)

    # Treina a rede neural
    my_network.train_classifier()

    # Executa a predição
    my_network.execute_prediction()

    # Recupera a precisão dos acertos
    accuracy = my_network.get_accuracy()

    print("Accuracy: ", accuracy)
