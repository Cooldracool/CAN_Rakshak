from ids.base import IDS
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
class DecisionTree(IDS):
    def __init__(self):
        self.dt = DecisionTreeClassifier(max_depth = 4)

    def train(self, train_dataset_dir=None, cfg=None, **kwargs):

        cfg = cfg or {}

        self.load_dataset(
            train_dataset_dir,
            cfg["file_name"],
            "train"
        )

        self.dt.fit(self.X, self.Y)

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
        joblib.dump(self.dt, path)
    
    def predict(self, X_test):
        dt_preds = self.dt.predict(X_test)
        return dt_preds

    def load(self, path):
        self.dt = joblib.load(path)

    def extract_features(self):
        pass

