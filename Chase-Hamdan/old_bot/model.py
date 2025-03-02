import pandas as pd
import pickle
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

        self._explainer = shap.KernelExplainer(self._model.predict, self._X_train, silent=True)
        self._shap_values = self._explainer.shap_values(self._X_test, silent=True)

    def _load_iris_dataset(self, csv_file):
        return pd.read_csv(csv_file)
    
    def _load_pickle_model(self, model_file):
        with open(model_file, "rb") as model_fp:
            model = pickle.load(model_fp)
        return model

    # Function to calculate the mean of the dataset
    def calculate_mean(self):
        return self._df.mean()

    # Function to calculate the median of the dataset
    def calculate_median(self):
        return self._df.median()

    # Function to calculate the standard deviation of the dataset
    def calculate_std(self):
        return self._df.std()

    # Function to calculate the variance of the dataset
    def calculate_variance(self):
        return self._df.var()

    # Function to calculate the mode of the dataset
    def calculate_mode(self):
        return self._df.mode().iloc[0]
    
    # Function to calculate accuracy score
    def calculate_accuracy(self):
        return accuracy_score(self._y_true, self._y_pred)

    # Function to calculate F1 score
    def calculate_f1(self):
        return f1_score(self._y_true, self._y_pred, average='weighted')

    # Function to calculate AUC-ROC score
    def calculate_roc_auc(self):
        return roc_auc_score(self._y_true, self._y_proba, multi_class='ovr')

    # Function to calculate precision score
    def calculate_precision(self):
        return precision_score(self._y_true, self._y_pred, average='weighted')

    # Function to calculate recall score
    def calculate_recall(self):
        return recall_score(self._y_true, self._y_pred, average='weighted')

    # Function to generate confusion matrix
    def generate_confusion_matrix(self):
        return confusion_matrix(self._y_true, self._y_pred)
    
    def shap_summary_plot(self):
        shap.summary_plot(self._shap_values, self._X_test,feature_names=self._feature_names)

    def shap_dependence_plot(self):
        shap.dependence_plot(0, self._shap_values, self._X_test, feature_names=self._feature_names)

    def shap_force_plot(self):
        shap.force_plot(self._explainer.expected_value[0], self._shap_values[..., 0], self._X_test, feature_names=self._feature_names)

