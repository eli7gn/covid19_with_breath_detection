"""
Created on 28 Dec 2023
@author: Elisheva Guahnich
This audio classifier uses an XGBoost model.
Download the audios from https://drive.google.com/drive/folders/1BGkigZWDt2GhGkDD14YoplgJmeuheWd6?usp=drive_link 
"""
import os
import pandas as pd
import numpy as np
import librosa  # pip install librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb  # pip install xgboost

DIRPATH = ""  # Set your directory path here


class AudioClassifier:
    def __init__(self):
        self.target_names = ["positive", "negative"]
        self.dataset = os.path.join(
            DIRPATH, "data/breath_coswara"
        )  # DIRPATH + "data/breath_coswara"
        self.process_dataset()

    def extract_features(self, file_name):
        try:
            audio, sample_rate = librosa.load(file_name)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None
        return mfccs_processed

    def process_dataset(self):
        features = []
        subfolders = ["positive", "negative"]

        for label, subfolder in enumerate(subfolders):
            folder_path = os.path.join(self.dataset, subfolder)
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):  # Check for .wav files
                    file_path = os.path.join(folder_path, file)
                    data = self.extract_features(file_path)
                    if data is not None:
                        features.append([data, label])

        # Convert into a Panda dataframe
        featuresdf = pd.DataFrame(features, columns=["feature", "class_label"])
        print(featuresdf)

        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        self.train_with_xgboost(x_train, y_train, x_test, y_test)
        self.train_with_random_forest(x_train, y_train, x_test, y_test)
        self.train_with_gradient_boosting(x_train, y_train, x_test, y_test)
        self.train_with_kneighbors(x_train, y_train, x_test, y_test)
        self.train_with_gaussian_nb(x_train, y_train, x_test, y_test)
        self.evaluate_combined_recall(x_train, y_train, x_test, y_test)

    def train_with_xgboost(self, x_train, y_train, x_test, y_test):
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(x_train, y_train)
        y_pred = xgb_model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "XGBoost")

    def train_with_logistic_regression(self, x_train, y_train, x_test, y_test):
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "Logistic Regression")

    def train_with_random_forest(self, x_train, y_train, x_test, y_test):
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "Random Forest")

    def train_with_gradient_boosting(self, x_train, y_train, x_test, y_test):
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "Gradient Boosting")

    def train_with_kneighbors(self, x_train, y_train, x_test, y_test):
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "KNeighbors")

    def train_with_gaussian_nb(self, x_train, y_train, x_test, y_test):
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "Gaussian Naive Bayes")

    def train_with_svc(self, x_train, y_train, x_test, y_test):
        model = SVC()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        self.evaluate_model(y_test, y_pred, "Support Vector Classifier")

    def evaluate_model(self, y_test, y_pred, model_name):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        print(
            f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}"
        )

        cm = confusion_matrix(y_test, y_pred)
        print(f"{model_name} Confusion Matrix:")
        print(cm)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def evaluate_combined_recall(self, x_train, y_train, x_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "KNeighbors": KNeighborsClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Support Vector Classifier": SVC(
                probability=True
            ),  # Ensure probability=True for SVC
            "XGBoost": xgb.XGBClassifier(),
        }

        # Store predictions from each model
        predictions = np.zeros_like(y_test)

        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            # If a model predicts positive, update the corresponding entry in predictions
            predictions = np.logical_or(predictions, y_pred).astype(int)

        # Now, predictions array will have 1 (positive) if any model predicted positive
        combined_recall = recall_score(y_test, predictions, average="binary")
        print(f"Combined Recall of All Models: {combined_recall:.2f}")
        self.evaluate_model(y_test, predictions, "Combined models")


# Runs when the script is executed directly
if __name__ == "__main__":
    classifier = AudioClassifier()
    print("Audio classifier has been created and trained.")