import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler, label_binarize


# Load the dataset from a CSV file

def load_iris_dataset():
    return pd.read_csv("./data/iris.csv")

# Function to calculate the mean of the dataset
def calculate_mean(df):
    return df.mean()

# Function to calculate the median of the dataset
def calculate_median(df):
    return df.median()

# Function to calculate the standard deviation of the dataset
def calculate_std(df):
    return df.std()

# Function to calculate the variance of the dataset
def calculate_variance(df):
    return df.var()

# Function to calculate the mode of the dataset
def calculate_mode(df):
    return df.mode().iloc[0]

# Function to display descriptive statistics
def display_statistics(df):
    print("Mean:")
    print(calculate_mean(df))
    print("\nMedian:")
    print(calculate_median(df))
    print("\nStandard Deviation:")
    print(calculate_std(df))
    print("\nVariance:")
    print(calculate_variance(df))
    print("\nMode:")
    print(calculate_mode(df))

# Load the KNN model from a pickle file
def load_knn_model():
    with open("knn_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model


# Function to calculate accuracy score
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Function to calculate F1 score
def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

# Function to calculate AUC-ROC score
def calculate_roc_auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba, multi_class='ovr')

# Function to calculate precision score
def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')

# Function to calculate recall score
def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')

# Function to generate confusion matrix
def generate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Display all model statistics
def display_model_statistics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)  # For AUC-ROC score if the model supports probability prediction
    
    print(f"Accuracy: {calculate_accuracy(y, y_pred)}")
    print(f"F1 Score: {calculate_f1(y, y_pred)}")
    print(f"Precision: {calculate_precision(y, y_pred)}")
    print(f"Recall: {calculate_recall(y, y_pred)}")
    print(f"AUC-ROC Score: {calculate_roc_auc(y, y_proba)}")
    print(f"Confusion Matrix:\n{generate_confusion_matrix(y, y_pred)}")

if __name__ == "__main__":
    # Replace 'iris.csv' with your file path
    iris_data = load_iris_dataset()
    model = load_knn_model()
    X = iris_data.drop(['Id','Species'], axis=1)  # Assuming 'species' is the target label column
    y = iris_data['Species']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #y = scaler.transform(y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Exclude the target label column if present
    numeric_data = iris_data.select_dtypes(include='number')
    
    # Display statistics for numeric columns
    display_statistics(numeric_data)
    display_model_statistics(model, X, y_encoded)
