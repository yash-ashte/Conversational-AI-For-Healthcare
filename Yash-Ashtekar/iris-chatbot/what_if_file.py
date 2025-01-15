import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler, label_binarize


# Load the dataset from a CSV file

def load_iris_dataset():
    return pd.read_csv("./data/iris.csv")
#from explain.actions.utils import convert_categorical_bools




def is_numeric(feature_name, temp_dataset):
    return feature_name in temp_dataset['numeric']


def is_categorical(feature_name, temp_dataset):
    return feature_name in temp_dataset['cat']


def get_numeric_updates(parse_text, i):
    """Gets the numeric update information."""
    update_term = parse_text[i+2]
    update_value = float(parse_text[i+3])
    return update_term, update_value


def update_numeric_feature(temp_data, feature_name, update_term, update_value):
    """Performs the numerical update."""
    new_dataset = temp_data.copy()

    if update_term == "increase":
        new_dataset[feature_name] += update_value
        parse_op = f"{feature_name} is increased by {str(update_value)}"
    elif update_term == "decrease":
        new_dataset[feature_name] -= update_value
        parse_op = f"{feature_name} is decreased by {str(update_value)}"
    elif update_term == "set":
        new_dataset[feature_name] = update_value
        parse_op = f"{feature_name} is set to {str(update_value)}"
    else:
        raise NameError(f"Unknown update operation {update_term}")

    return new_dataset, parse_op


def what_if_operation(temp_data, feature_name, update_term, update_value, is_numeric, is_categorical):
    """The what if operation."""
    feature_name = feature_name.strip().replace("'", "")
    update_term = update_term.strip().replace("'", "")
    # The temporary dataset to approximate
    temp_dataset = temp_data

    # The feature name to adjust
    feature_name = feature_name

    # Numerical feature case. Also putting id in here because the operations
    # are the same
    if is_numeric:
        #$update_term, update_value = get_numeric_updates(parse_text, i)
        temp_dataset, parse_op = update_numeric_feature(temp_dataset,
                                                             feature_name,
                                                             update_term,
                                                             update_value)
    elif is_categorical:
        # handles conversion between true/false and 1/0 for categorical features
        #categorical_val = convert_categorical_bools(parse_text[i+2])
        temp_dataset[feature_name] = update_term
        #parse_op = f"{feature_name} is set to {str(categorical_val)}"
    elif feature_name == "id":
        # Setting what if updates on ids to no effect. I don't think there's any
        # reason to support this.
        return "What if updates have no effect on id's!", 0
    else:
        raise NameError(f"Parsed unknown feature name {feature_name}")

    #rocessed_ids = list(conversation.temp_dataset.contents['X'].index)
    #conversation.temp_dataset.contents['ids_to_regenerate'].extend(processed_ids)

    #conversation.add_interpretable_parse_op("and")
    #conversation.add_interpretable_parse_op(parse_op)

    return temp_dataset

from descript import load_knn_model
from sklearn.preprocessing import StandardScaler

def what_if_for_id(temp_data, feature_name, row_id, new_value):
    """
    Updates a specific feature for a given row ID, returns the updated dataset,
    and prints the model's prediction for the modified entry.
    
    Parameters:
        temp_data (pd.DataFrame): The dataset.
        row_id (int): The ID of the row to update.
        feature_name (str): The feature to update.
        new_value (float or str): The new value for the feature.
    
    Returns:
        pd.DataFrame: Updated dataset with the specified feature modified for the given ID.
    """
    # Check if the feature exists in the dataset
    if feature_name not in temp_data.columns:
        raise ValueError(f"Feature '{feature_name}' does not exist in the dataset.")
    
    # Locate the row by ID
    if row_id not in temp_data['Id'].values:
        raise ValueError(f"ID '{row_id}' does not exist in the dataset.")
    
    # Update the feature value for the specified ID
    temp_data.loc[temp_data['Id'] == row_id, feature_name] = new_value
    
    # Load the trained KNN model
    model = load_knn_model()
    
    # Prepare the row for prediction
    row_to_predict = temp_data[temp_data['Id'] == row_id].drop(['Id', 'Species'], axis=1)
    
    # Apply scaling if the model requires it
    scaler = StandardScaler()
    numeric_features = temp_data.drop(['Id', 'Species'], axis=1).select_dtypes(include='number')
    scaled_data = scaler.fit_transform(numeric_features)  # Fit and scale the original data for the model
    row_scaled = scaler.transform(row_to_predict)  # Scale the specific row to match model expectations
    
    # Predict the species for the updated row
    predicted_species_code = model.predict(row_scaled)[0]
    species_name = pd.Categorical(temp_data['Species']).categories[predicted_species_code]
    
    print(f"Updated row prediction: {species_name}")

    temp_data =temp_data.drop(['Id','Species'], axis=1)    
    return temp_data
