from features.feature_extractors.base import FeatureExtractor
from utilities import *
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import csv

class Stat(FeatureExtractor):
    def __init__(self, cfg=None):
        super().__init__(cfg or {})
        self.X, self.Y = self.extract_features(cfg=cfg or {})
        print("Features extracted using Stat feature extractor")
        print("X shape : ", self.X.shape)
        print("Y shape : ", self.Y.shape)

    def extract_features(self, file_path=None, cfg=None):
        cfg = cfg or {}
        dir_path     = cfg.get('dir_path', '')
        dataset_name = cfg.get('dataset_name', '')
        file_name    = cfg.get('file_name', '')
        model_name   = cfg.get('model_name', '')
        mode         = cfg.get('mode', '')

        print("Extracting features")
        dataset_path = os.path.join(dir_path, "..", "datasets", dataset_name)
        modified_dataset_path = os.path.join(dataset_path, "modified_dataset")
        file_path = os.path.join(modified_dataset_path, file_name.replace(".log", ".csv"))
        print("File path : ", file_path)
        df = self.read_attack_data(file_path)

        X, Y = df.drop(columns=['flag', 'timestamp']).values, df['flag'].values
        scalar_path = os.path.join(modified_dataset_path, model_name + "scalar.pkl")

        """if mode == "train":
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, scalar_path)

        if mode == "test":
            modified_dataset_path = os.path.join(dataset_path, "train")
            scalar_path = os.path.join(modified_dataset_path, model_name + "scalar.pkl")
            scaler = joblib.load(scalar_path)
            
            # Scale features
            
        X = scaler.transform(X)"""

        # Create output folder names
        features_dir = os.path.join(self.features_path, "Stat")
        os.makedirs(features_dir, exist_ok=True)

        prefix = file_name.replace(".csv", "").replace(".log", "")

        

        # Convert labels
        label_map = {
            'B': 0,
            'D': 1,
            'F': 2,
            'G': 3
        }
        Y = df['flag'].map(label_map).astype(int).values

        # Save full dataset
        features_csv = os.path.join(features_dir, prefix + "_features.csv")
        labels_csv = os.path.join(features_dir, prefix + "_labels.csv")

        np.savetxt(features_csv, X, delimiter=",")

        # Save Y
        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label"])

            for i, label in enumerate(Y):
                writer.writerow([i, int(label)])

        print(f"Saved features -> {features_csv}")
        print(f"Saved labels   -> {labels_csv}")
        # ---------------------------------------------------------
        return X, Y
        
    def read_attack_data(self,data_path):

        columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
            'data5', 'data6', 'data7', 'flag']

        data = pd.read_csv(data_path, names = columns,skiprows=1)
        #data = shift_columns(data)
        
        data = data.replace(np.nan, '00')
        
        data_cols = [
            'data0','data1','data2','data3',
            'data4','data5','data6','data7'
        ]

        data['can_id'] = data['can_id'].apply(hex_to_dec)

        for col in data_cols:
            data[col] = data[col].apply(hex_to_dec)

        # don't compute DATA
        # don't compute IAT
        # don't drop data0...data7

        return data
