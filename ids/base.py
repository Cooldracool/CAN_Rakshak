from common_imports import ABC, abstractmethod
from common_imports import os, np, csv


class IDS(ABC):

    def load_dataset(self, dataset_dir, file_name, mode):
        prefix = file_name.replace(".csv", "").replace(".log", "")

        features_csv = os.path.join(
            dataset_dir,
            f"{prefix}_{mode}_features.csv"
        )

        labels_csv = os.path.join(
            dataset_dir,
            f"{prefix}_{mode}_labels.csv"
        )

        self.X = np.loadtxt(features_csv, delimiter=",")

        labels = []
        with open(labels_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                labels.append(int(row[1]))

        self.Y = np.array(labels)

    @abstractmethod
    def train(self, X_train=None, Y_train=None, **kwargs):
        """
        Train the model using the provided training dataset.

        :param train_dataset: The dataset used for training the model.
        :param val_dataset: Optional validation dataset to evaluate model performance during training.
        :param kwargs: Additional keyword arguments for the training process.
        """
        pass

    @abstractmethod
    def test(self, X_test=None, Y_test=None, **kwargs):
        """
        Test the model using the provided test dataset.

        :param test_dataset: The dataset used for testing the model.
        :param kwargs: Additional keyword arguments for the testing process.
        """
        pass

    @abstractmethod
    def predict(self, X_test=None, **kwargs):
        """
        Predict the output using the model for the given input features.

        :param features: Input features for which to predict the output.
        :return: The predicted output.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save the trained model to the specified path.

        :param path: The file path where the model will be saved.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load a trained model from the specified path.

        :param path: The file path from which the model will be loaded.
        """
        pass