import pandas as pd
import pickle
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Data_Model:

    def __init__(self, csv_file, model_file):
        # Initialize imported dataset and model
        # load data set and model
        iris_data = self._load_iris_dataset(csv_file)
        self._model = self._load_pickle_model(model_file)

        self._df =  iris_data.select_dtypes(include='number')

        self._X = iris_data.drop(['species'], axis=1) 
        self._y = iris_data['species']
        self._feature_names = self._X.columns

        self._y = pd.Categorical(self._y).codes
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y, test_size=0.20, random_state=0)

        label_encoder = LabelEncoder()
        self._y_true = label_encoder.fit_transform(self._y)
        self._y_pred = self._model.predict(self._X)
        self._y_proba = self._model.predict_proba(self._X)

        self._summary_explainer = shap.KernelExplainer(self._model.predict, self._X_train, silent=True)
        self._force_explainer = shap.KernelExplainer(self._model.predict_proba, self._X_train, silent=True)
        self._summary_shap_values = self._summary_explainer.shap_values(self._X_test, silent=True)
        self._force_shap_values = self._force_explainer.shap_values(self._X_test, silent=True)

        self._feature_importance = self._get_shap_averages()

    def _load_iris_dataset(self, csv_file):
        return pd.read_csv(csv_file)
    
    def _load_pickle_model(self, model_file):
        with open(model_file, "rb") as model_fp:
            model = pickle.load(model_fp)
        return model

    def calculate_mean(self):
        """Function to calculate the mean of the dataset"""
        return self._df.mean()

    def calculate_median(self):
        """Function to calculate the median of the dataset"""
        return self._df.median()

    def calculate_std(self):
        """Function to calculate the standard deviation of the dataset"""
        return self._df.std()

    def calculate_variance(self):
        """Function to calculate the variance of the dataset"""
        return self._df.var()

    def calculate_mode(self):
        """Function to calculate the mode of the dataset"""
        return self._df.mode().iloc[0]
    
    def calculate_accuracy(self):
        """Function to calculate accuracy score"""
        return accuracy_score(self._y_true, self._y_pred)

    def calculate_f1(self):
        """Function to calculate F1 score"""
        return f1_score(self._y_true, self._y_pred, average='weighted')

    def calculate_roc_auc(self):
        """Function to calculate AUC-ROC score"""
        return roc_auc_score(self._y_true, self._y_proba, multi_class='ovr')

    def calculate_precision(self):
        """Function to calculate precision score"""
        return precision_score(self._y_true, self._y_pred, average='weighted')

    def calculate_recall(self):
        """Function to calculate recall score"""
        return recall_score(self._y_true, self._y_pred, average='weighted')

    def generate_confusion_matrix(self):
        """Function to generate confusion matrix"""
        return confusion_matrix(self._y_true, self._y_pred)
    
    def _get_shap_averages(self):
        shap_values = self._summary_explainer(self._X)

        # Compute mean absolute SHAP values per feature
        feature_importance = {
            feature: np.abs(shap_values.values[:, i]).mean()
            for i, feature in enumerate(self._X.columns)
        }

        # Sort features by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

        return sorted_importance
    
    def shap_summary_plot(self):
        """Generate a SHAP summary plot to visualize feature importance and the impact of each feature on model predictions."""
        shap.summary_plot(self._summary_shap_values, self._X_test,feature_names=self._feature_names)
        return self._feature_importance

    def shap_force_plot(self):
        """Generate a SHAP force plot to visualize the contribution of individual features for a specific prediction."""
        fp = shap.force_plot(self._force_explainer.expected_value[0], self._force_shap_values[..., 0], self._X_test, feature_names=self._feature_names)
        shap.save_html('plots/force_plot.html', fp)
        return ""