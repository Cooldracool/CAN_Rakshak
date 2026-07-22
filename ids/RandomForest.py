from common_imports import accuracy_score, joblib
from sklearn.ensemble import RandomForestClassifier
from ids.base import IDS

class RandomForest(IDS):
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)

    def train(self, train_dataset_dir=None, cfg=None, **kwargs):

        cfg = cfg or {}

        self.load_dataset(
            train_dataset_dir,
            cfg["file_name"],
            "train"
        )

        self.rf.fit(self.X, self.Y)

    def test(self, test_dataset_dir=None, cfg=None, **kwargs):

        cfg = cfg or {}

        self.load_dataset(
            test_dataset_dir,
            cfg["file_name"],
            "test"
        )

        Y_pred = self.predict(self.X)

        return Y_pred, self.Y

    def save(self, path):
        joblib.dump(self.rf, path)

    def predict(self, X_test):
        rf_preds = self.rf.predict(X_test)
        return rf_preds

    def load(self, path):
        self.rf = joblib.load(path)

    def extract_features(self):
        pass