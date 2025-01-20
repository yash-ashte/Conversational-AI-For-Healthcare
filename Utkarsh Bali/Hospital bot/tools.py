from datetime import timedelta
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap
import io
import base64
import firebase_admin
from firebase_admin import credentials, storage
import os


firebase_credentials = credentials.Certificate("sum1-b6b1d-firebase-adminsdk-4otox-249a1600f3.json")
firebase_admin.initialize_app(firebase_credentials, {
    "storageBucket": "sum1-b6b1d.appspot.com"  # Replace with your Firebase storage bucket
})

# Load data
file_path = "hospital_dataset.csv"
data = pd.read_csv(file_path)

# Define columns
categorical_columns = ['gender', 'class', 'adm source', 'dis location', 'specialty', 'year', 'month',
                       'day of week']
numeric_columns = ['LOS', 'age', 'num of transfers', 'Charlson', 'vanWalraven', 'Time to Readmission']
target_column = 'Readm Indicator'

# Drop unnecessary columns
data.drop(['Case ID'], axis=1, inplace=True)
data.drop(['primary icd'], axis=1, inplace=True)
data.drop(['secondary diag code'], axis=1, inplace=True)
data.drop(['dept OU'], axis=1, inplace=True)

# Handle missing values
data = data.dropna(subset=[target_column])

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Scale numeric variables
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Split data into features (X) and target (y)
X = data.drop(target_column, axis=1)
y = data[target_column]

# Ensure the target variable contains valid values
if y.isnull().any():
    print("Error: The target variable still contains missing values!")
else:
    print("Target variable cleaned successfully.")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print(f"Model Accuracy: {accuracy:.2%}")

# Get predicted probabilities for ROC curve
y_prob = model.predict_proba(X_test)[:, 1]  # Only for the positive class

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# SHAP Analysis
explainer_xgb = shap.Explainer(model, X_test)
shap_values_xgb = explainer_xgb(X)

# custom functions
def get_model_accuracy():
    return f"{accuracy:.2%}"

def get_roc_auc():
    return f"{roc_auc:.2f}"

def get_feature_importance():
    """Generates the SHAP summary plot and uploads it to Firebase Storage."""
    try:
        # Step 1: Generate and save the SHAP summary plot
        file_name = "shap_summary_plot.png"
        shap.summary_plot(shap_values_xgb, X, show=False)
        plt.savefig(file_name)
        plt.close()

        # Step 2: Upload the image to Firebase and get the signed URL
        image_url = upload_to_firebase(file_name)

        # Clean up local file
        os.remove(file_name)

        return image_url
    except Exception as e:
        return f"Error generating feature importance plot: {str(e)}"

def upload_to_firebase(file_path):
    """Uploads an image to Firebase Storage and returns a signed URL with expiration."""
    try:
        # Get Firebase Storage bucket
        bucket = storage.bucket()
        blob = bucket.blob(os.path.basename(file_path))

        # Upload the file
        blob.upload_from_filename(file_path)

        # Generate a signed URL (expiring after 5 minutes)
        signed_url = blob.generate_signed_url(expiration=timedelta(minutes=5))
        return signed_url
    except Exception as e:
        raise Exception(f"Failed to upload to Firebase: {e}")

def explain_instance(instance_index):
    """Generates and uploads an explanation plot."""
    if instance_index is not None:
        try:
            # Step 1: Convert index and get instance data
            instance_index = int(instance_index)
            instance_data = X.iloc[[instance_index]]

            # Step 2: Generate SHAP values for the instance
            shap_values_instance = explainer_xgb(instance_data)

            # Step 3: Generate and save the SHAP waterfall plot
            file_name = f"shap_waterfall_plot_instance_{instance_index}.png"
            shap.waterfall_plot(shap_values_instance[0], show=False)
            plt.savefig(file_name)
            plt.close()

            # Step 4: Upload to Firebase and get a signed URL
            image_url = upload_to_firebase(file_name)

            # Clean up local file
            os.remove(file_name)

            return image_url
        except Exception as e:
            return f"Error explaining instance: {str(e)}"
    else:
        return "Instance index is required."

def get_feature_impact(feature_name):
    """Generates SHAP dependence plot for a feature and uploads it to Firebase Storage."""
    if feature_name in X.columns:
        try:
            # Step 1: Determine feature index
            feature_index = list(X.columns).index(feature_name)

            # Step 2: Generate and save the SHAP dependence plot
            file_name = f"shap_dependence_plot_{feature_name}.png"
            shap.dependence_plot(feature_index, shap_values_xgb.values, X, show=False)
            plt.savefig(file_name)
            plt.close()

            # Step 3: Upload the image to Firebase and get the signed URL
            image_url = upload_to_firebase(file_name)

            # Clean up local file
            os.remove(file_name)

            return image_url
        except Exception as e:
            return f"Error generating feature impact plot: {str(e)}"
    else:
        return f"Feature '{feature_name}' not found in dataset."

TOOL_MAP = {"get_model_accuracy": get_model_accuracy,
            "get_roc_auc": get_roc_auc,
            "get_feature_importance": get_feature_importance,
            "get_feature_impact": get_feature_impact,
            "explain_instance": explain_instance}
