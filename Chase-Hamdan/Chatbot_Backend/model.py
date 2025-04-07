import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import shap
import numpy as np

class Data_Model:
    def __init__(self, data_csv: str, model_pickle: str):
        """
        Initialize the Data_Model with a dataset and a trained model.

        Args:
            data_csv (str): Path to the CSV file containing the Iris dataset.
            model_pickle (str): Path to the pickled trained classification model.
        """
        self._data = self._load_data_csv(data_csv)
        self._model = self._load_pickle_model(model_pickle)

        self._feature_names = self._data.drop(['species'], axis=1).columns.to_list()
        self._target_names = self._data['species'].unique().tolist()

        self._X = self._data.drop(['species'], axis=1)
        self._y = pd.Categorical(self._data['species']).codes
        self._y_pred = self._model.predict(self._X)

        self._explainer = shap.Explainer(self._model, feature_names=self._feature_names)
        self._explanation = self._explainer(self._X)

    def _load_data_csv(self, csv_file):
        """Load a CSV file into a pandas DataFrame."""
        return pd.read_csv(csv_file)
    
    def _load_pickle_model(self, model_file):
        """Load a pickled model from file."""
        with open(model_file, "rb") as model_fp:
            model = pickle.load(model_fp)
        return model

    ### Descriptive Statistic Functions ###

    def get_feature_names(self):
        """
        Get the list of feature names used in the dataset.

        Args:

        Returns:
            str: A stringified list of feature names.
        """
        return str(self._feature_names)
    
    def get_flower_types(self):
        """
        Get the list of unique flower species in the dataset.
        
        Args:

        Returns:
            str: A stringified list of flower species.
        """
        return str(self._target_names)

    def data_header(self):
        """
        Get the first few rows of the dataset to inspect formatting.

        Args:

        Returns:
            str: The first 5 rows of the dataset.
        """
        return str(self._data.head())
    
    def describe_data(self, species='null'):
        """
        Get summary statistics of the dataset including count, mean, standard deviation, min, first quartile, median, third quartile, and max for each feature. Optionally specify a species to only include data for that species
        
        Args:
            species (str): Optionally filter the data to only include this species. Enum['setosa', 'versicolor', 'virginica', 'null']

        Returns:
            str: Statistical description of all numerical features.
        """
        df = self._data
        if species != 'null':
            df = df[df['species'] == species]
        return str(df.describe())
    
    def plot_descriptive(self, feature:str, species='null', separate_species:bool=False, plot_type:str='histogram'):
        """
        Plot a histogram or boxplot for a given feature, optionally filtered by species.

        Args:
            feature (str): The feature to visualize. Enum['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
            species (str, optional): Filter the plot to only include this species. Enum['setosa', 'versicolor', 'virginica', 'null']
            separate_species (bool): Whether to color plots by species.
            plot_type (str): Type of plot ('histogram' or 'boxplot'). Enum['histogram', 'boxplot']

        Returns:
            None
        """
        df = self._data
        hue = None
        plots = []
        
        if species != 'null':
            df = df[df['species'] == species]

        if separate_species:
            hue = 'species'

        if plot_type == 'histogram':
            sns.histplot(data=df, x=feature, hue=hue, kde=True, palette='Set2')
            plt.title(f'Histogram of {feature} by Species')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plots.append(f"plots/histogram_{feature}.png")
            plt.savefig(plots[0], format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()

        elif plot_type == 'boxplot':
            sns.boxplot(x='species', y=feature, data=df, palette='Set2')
            plt.title(f'Box Plot of {feature} by Species')
            plt.savefig('test.png')
            plots.append(f"plots/boxplot_{feature}.png")
            plt.savefig(plots[0], format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            print("Invalid plot type. Choose either 'histogram' or 'boxplot'.")

        return 'plot displayed', plots

    ### Model Explainability Functions ###

    def predict(self, sepal_length:float, sepal_width:float, petal_length:float, petal_width:float, feature_importance:bool=False):
        """
        Predict the species of an iris flower given its features, optionally with SHAP explanation for feature importance if specified.

        Args:
            sepal_length (float): Sepal length in cm.
            sepal_width (float): Sepal width in cm.
            petal_length (float): Petal length in cm.
            petal_width (float): Petal width in cm.
            feature_importance (bool): Whether to calculate SHAP feature importance values and show visualization.

        Returns:
            str: Predicted species and confidence score, plus SHAP values if requested.
        """
        test_array = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        predicted = self._model.predict(test_array)[0]
        prob = self._model.predict_proba(test_array)[0][predicted] * 100
        result = f'Predicted {str(self._target_names[predicted])} with {prob:.3f}% confidence'
        plots = []
        if feature_importance:
            result += "\nFeature Importance SHAP values:\n"
            explanation = self._explainer(test_array)
            for i in range(len(self._feature_names)):
                result += f"  {str(self._feature_names[i])}: {explanation[0, :, predicted].values[i]}\n"
            plt.title(self._target_names[predicted])
            shap.plots.waterfall(explanation[0, :, predicted], show=False)
            plots.append(f"plots/predicted_feature_importance_{str(self._target_names[predicted])}.png")
            plt.savefig(plots[0], format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()
        return result, plots
        
    def shap_feature_importance(self, species:str='null', explanation_needed:bool=True, plot_type="bar"):
        """
        Plot global SHAP feature importances for the model and return SHAP values for each feature, 
        either for a specific species or for all species.

        Args:
            species (str): Species name to filter the SHAP explanation (e.g., 'setosa'). If not provided, SHAP values for all species will be plotted. Enum['setosa', 'versicolor', 'virginica', 'null']
            explanation_needed (bool): Whether to generate and display the SHAP explanation. Default is True.
            plot_type (str): Type of SHAP plot ('bar' or 'beeswarm'). Default is 'bar'. Enum['bar', 'beeswarm']

        Returns:
            str: A string summary of SHAP values for the specified species or all species, 
                with feature importance values for each feature.
        """
        result = ''
        plots = []
        avg_abs_shap_values = (np.sum(np.abs(self._explanation.values), axis=0) / len(self._explanation.values))

        if species != 'null':
            i = self._target_names.index(species)
            plt.title(species)
            shap.plots.bar(self._explanation[:, :, i], show=False)
            plots.append(f"plots/feature_importance_{species}.png")
            plt.savefig(plots[0], format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()
            result += f"Shap Values for {species} classification\n"
            for j in range(len(self._feature_names)):
                result += f'  {self._feature_names[j]}: {avg_abs_shap_values[j, i]}\n'
        else:
            for i in range(len(self._target_names)):
                plt.title(self._target_names[i])
                shap.plots.bar(self._explanation[:, :, i], show=False)
                plots.append(f"plots/feature_importance_{self._target_names[i]}.png")
                plt.savefig(plots[i], format='png', bbox_inches='tight', pad_inches=0.1)
                plt.close()
                result += f"Shap Values for {self._target_names[i]} classification\n"
                for j in range(len(self._feature_names)):
                    result += f'  {self._feature_names[j]}: {avg_abs_shap_values[j, i]}\n'
        
        return result, plots

    def get_confusion_matrix(self):
        """
        Get the confusion matrix of the model's predictions.

        Args:

        Returns:
            str: String representation of the confusion matrix.
        """
        return str(confusion_matrix(self._y, self._y_pred))
    
    def get_accuracy_score(self):
        """
        Get the accuracy score of the model on the dataset.
        
        Args:

        Returns:
            str: Accuracy as a float.
        """
        return str(accuracy_score(self._y, self._y_pred))
    
    def get_classification_report(self):
        """
        Get the classification report for model predictions.
        
        Args:

        Returns:
            str: Text summary of precision, recall, F1-score per class.
        """
        return classification_report(self._y, self._y_pred, target_names=self._target_names)
