from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


class GermanCreditExperiment:

    def __init__(self):
        self.models = [RandomForestClassifier()]
        self.model_reports = {}

        self.model_names = []
        for model in self.models:
            self.model_names.append(model.__class__.__name__)

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):

        for model in self.models:
            model.fit(X_train, y_train)

        self._get_performance_report(X_test, y_test)

    def _get_performance_report(self, X_test, y_test):

        for model, model_name in zip(self.models, self.model_names):
            y_pred = model.predict(X_test)
            self.model_reports[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
            }
