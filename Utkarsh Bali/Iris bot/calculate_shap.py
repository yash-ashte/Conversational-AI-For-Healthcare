import sys
import json
import shap
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

# Split the dataset into training and testing sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost Classifier on the training data
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Get user input from command line arguments
user_input = json.loads(sys.argv[1])
user_input_array = np.array(user_input).reshape(1, -1)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(user_input_array)

# Output SHAP values as JSON
print(json.dumps(shap_values.values.tolist()))
