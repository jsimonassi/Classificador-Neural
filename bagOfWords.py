import math
from sklearn.feature_extraction.text import CountVectorizer


class BagOfWords:

    @staticmethod
    def get_xy_data(dataset, processed_data, polarity_column, lines_to_read):
        matrix = CountVectorizer(max_features=1000)
        x = matrix.fit_transform(processed_data).toarray()
        y = dataset.iloc[:lines_to_read, polarity_column]

        for i in range(len(y)):
            if math.isnan(y[i]):
                y[i] = -1

        return x, y
