class ClassifierWrapper:

    def __init__(self, classifier, backend):
        self.classifier = classifier
        self.backend = backend

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.classifier.predict(X)
