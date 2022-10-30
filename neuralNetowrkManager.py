from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class NeuralNetworkManager:

    def __init__(self, x, y, cross_validation=True):
        """Inicializa o objeto de gerenciamento da rede neural"""
        if cross_validation:
            skf = StratifiedKFold(n_splits=5)
            for train_index, test_index in skf.split(x, y):
                self.x_train, self.x_test = x[train_index], x[test_index]
                self.y_train, self.y_test = y[train_index], y[test_index]
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)
        self.classifier = None
        self.y_pred = None

    def train_classifier_with_naive_bayes(self):
        """Treina o classificador utilizando o algoritmo Naive Bayes"""
        self.classifier = GaussianNB()
        self.classifier.fit(self.x_train, self.y_train)

    def train_classifier_with_logistic_regression(self):
        """Treina o classificador utilizando o algoritmo Logistic Regression"""
        self.classifier = LogisticRegression()
        self.classifier.fit(self.x_train, self.y_train)

    def train_classifier_with_decision_tree(self):
        """Treina o classificador utilizando o algoritmo Decision Tree"""
        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(self.x_train, self.y_train)

    def execute_prediction(self):
        """Executa a predição com base no classificador instanciado"""
        self.y_pred = self.classifier.predict(self.x_test)

    def get_accuracy(self):
        """Recupera a precisão dos acertos"""
        if self.y_pred is None:
            raise Exception('You must execute prediction before get accuracy')
        return accuracy_score(self.y_test, self.y_pred)

    def get_confusion_matrix(self):
        """Recupera a matriz de confusão"""
        if self.y_pred is None:
            raise Exception('You must execute prediction before get confusion matrix')
        return confusion_matrix(self.y_test, self.y_pred)

