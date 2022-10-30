from bagOfWords import BagOfWords
from dataReader import DataReader
from textProcessor import TextProcessor
from neuralNetowrkManager import NeuralNetworkManager
import nltk
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# Coluna de polaridade da tabela de dataset
POLARITY_COLUMN = 4
# Número de linhas que serão lidas do arquivo csv
LINES_TO_READ = 5000


def plot_confusion_matrix(conf_matrix):
    df_cm = pd.DataFrame(conf_matrix, range(len(conf_matrix)), range(len(conf_matrix[0])))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()


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
    my_network = NeuralNetworkManager(x, y, cross_validation=True)

    # Treina a rede neural
    my_network.train_classifier_with_logistic_regression()

    # Executa a predição
    my_network.execute_prediction()

    # Recupera a precisão dos acertos
    accuracy = my_network.get_accuracy()

    print("Accuracy: ", accuracy)

    plot_confusion_matrix(my_network.get_confusion_matrix())
