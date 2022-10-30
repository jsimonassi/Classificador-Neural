from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class NeuralNetworkManager:

    def __init__(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)
        self.classifier = GaussianNB()
        self.y_pred = None

    def train_classifier(self):
        self.classifier.fit(self.x_train, self.y_train)

    def execute_prediction(self):
        self.y_pred = self.classifier.predict(self.x_test)

    def get_accuracy(self):
        if self.y_pred is None:
            raise Exception('You must execute prediction before get accuracy')
        return accuracy_score(self.y_test, self.y_pred)
