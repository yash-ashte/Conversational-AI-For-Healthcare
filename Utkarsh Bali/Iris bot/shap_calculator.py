import json
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# Load the Iris dataset
iris_data = pd.read_csv('iris.csv')
X = iris_data.drop(['Id', 'Species'], axis=1)
y = iris_data['Species']

# Train an XGBoost model
model = xgb.XGBClassifier(random_state=42)
model.fit(X, y)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)

def calculate_shap_values(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_df)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Create a dictionary of feature names and their corresponding SHAP values
    shap_dict = {feature: shap_values[0][i].tolist() for i, feature in enumerate(feature_names)}
    
    return shap_dict

if __name__ == "__main__":
    # Read input data from stdin
    input_json = sys.stdin.read()
    input_data = json.loads(input_json)
    
    # Calculate SHAP values
    shap_values = calculate_shap_values(input_data)
    
    # Print the result as JSON
    print(json.dumps(shap_values))
