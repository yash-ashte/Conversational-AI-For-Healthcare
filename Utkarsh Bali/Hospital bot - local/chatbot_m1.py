import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import shap
import numpy as np
import pickle
import seaborn as sns
import json
import ollama

file_path = "hospital_dataset.csv"
data = pd.read_csv(file_path)

# Define columns
categorical_columns = ['gender', 'class', 'adm source', 'dis location', 'specialty', 'year', 'month', 'day of week']
numeric_columns = ['LOS', 'age', 'num of transfers', 'Charlson', 'vanWalraven', 'Time to Readmission']
target_column = 'Readm Indicator'

# Drop unnecessary columns
data.drop(['Case ID', 'primary icd', 'secondary diag code', 'dept OU'], axis=1, inplace=True)

# Handle missing values in target
data = data.dropna(subset=[target_column])

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# -----------------------------------------
# Save a copy of data BEFORE scaling for plots
# -----------------------------------------
raw_data = data.copy()

# Scale numeric variables on 'data' for modeling purposes
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Split data into features (X) and target (y)
X = data.drop(target_column, axis=1)
y = data[target_column]

if y.isnull().any():
    print("Error: The target variable still contains missing values!")
else:
    print("Target variable cleaned successfully.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# ROC and AUC
y_prob = model.predict_proba(X_test)[:, 1]  # For the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

explainer_xgb = shap.Explainer(model, X_test)
shap_values_xgb = explainer_xgb(X)

def get_model_accuracy():
    return f"{accuracy:.2%}"

def get_roc_auc():
    return f"{roc_auc:.2f}"

def get_feature_importance():
    """Generate a SHAP summary plot to display feature importance."""
    shap.summary_plot(shap_values_xgb, X)
    return "Successfully generated SHAP summary plot."

def explain_instance(instance_index):
    if instance_index is not None:
          instance_index = int(input("Enter the index of the instance to explain: "))
          instance_data = X.iloc[[instance_index]]
          shap_values_instance = explainer_xgb(instance_data)
          shap.waterfall_plot(shap_values_instance[0])
          return "successfully generated plot"

def get_feature_impact(feature_name):
    if feature_name in X.columns:
        feature_index = list(X.columns).index(feature_name)
        shap.dependence_plot(feature_index, shap_values_xgb.values, X)
        return "succesfully generated plot"

def get_shap_value(instance_index):
    try:
        instance_index = int(instance_index)
    except Exception:
        return "Instance index must be an integer."
    if instance_index < 0 or instance_index >= len(X):
        return "Invalid instance index."
    instance_data = X.iloc[[instance_index]]
    shap_values_instance = explainer_xgb(instance_data)
    return str(dict(zip(X.columns, shap_values_instance.values[0])))

def get_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    return "Confusion matrix plotted."

def get_classification_report():
    report = classification_report(y_test, y_pred)
    print(report)
    return report

def get_precision_score():
    prec = precision_score(y_test, y_pred, average='binary')
    return f"Precision: {prec:.2f}"

def get_recall_score():
    rec = recall_score(y_test, y_pred, average='binary')
    return f"Recall: {rec:.2f}"

def get_f1_score():
    f1 = f1_score(y_test, y_pred, average='binary')
    return f"F1 Score: {f1:.2f}"


def plot_feature_distribution(feature_name):
    """
    Plot the distribution of a given feature using raw (pre-scaled) data.
    """
    if feature_name not in raw_data.columns:
        return f"Feature {feature_name} not found in dataset."
    plt.figure(figsize=(8, 6))
    plt.hist(raw_data[feature_name], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {feature_name} (Raw Data)")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.show()
    return f"Plotted distribution for {feature_name}."

def plot_data_correlation():
    """
    Plot a heatmap of the correlation matrix for features using raw (pre-scaled) data.
    """
    corr_matrix = raw_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap (Raw Data)")
    plt.show()
    return "Correlation heatmap plotted."

def get_learning_curve():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()
    return "Learning curve plotted."

def save_model(file_name="xgb_model.pkl"):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    return f"Model saved to {file_name}."

def load_model(file_name="xgb_model.pkl"):
    global model
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return f"Model loaded from {file_name}."

def predict_readmission(new_data):
    """
    Predict readmission for a new patient.
    :param new_data: Dictionary with keys matching the feature columns.
    :return: A string with predicted probability and predicted class.
    """
    df_new = pd.DataFrame([new_data])
    
    # Process categorical columns
    for col in categorical_columns:
        if col not in df_new.columns:
            df_new[col] = 0
        else:
            df_new[col] = label_encoders[col].transform(df_new[col].astype(str))
    
    # Process numeric columns
    for col in numeric_columns:
        if col not in df_new.columns:
            df_new[col] = 0
        else:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)
    
    df_new = df_new.reindex(columns=X.columns, fill_value=0)
    
    # Scale numeric columns all at once (using model-scaled data)
    df_new[numeric_columns] = scaler.transform(df_new[numeric_columns])
    
    pred_prob = model.predict_proba(df_new)[0][1]
    pred_class = model.predict(df_new)[0]
    return f"Predicted readmission probability: {pred_prob}, Predicted class: {int(pred_class)}"

def get_patient_profile(patient_index):
    try:
        patient_index = int(patient_index)
    except Exception:
        return "Patient index must be an integer."
    if patient_index < 0 or patient_index >= len(X):
        return "Invalid patient index."
    patient_record = raw_data.iloc[patient_index]  # using raw_data for original values
    shap_values_patient = explainer_xgb(X.iloc[[patient_index]])
    print("Patient Record:")
    print(patient_record)
    shap.waterfall_plot(shap_values_patient[0])
    return "Patient profile and SHAP explanation generated."

def analyze_readmission_trends(by_column):
    if by_column not in raw_data.columns:
        return f"Column {by_column} not found in the dataset."
    trend_data = raw_data.groupby(by_column)[target_column].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=by_column, y=target_column, data=trend_data, palette="viridis")
    plt.title(f"Readmission Rate by {by_column.capitalize()} (Raw Data)")
    plt.ylabel("Average Readmission Rate")
    plt.xlabel(by_column.capitalize())
    plt.xticks(rotation=45)
    plt.show()
    return f"Readmission trends analyzed by {by_column}."

def plot_readmission_distribution():
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=raw_data, palette="pastel")
    plt.title("Distribution of Readmission Indicator (Raw Data)")
    plt.xlabel("Readmission")
    plt.ylabel("Count")
    plt.show()
    return "Readmission distribution plotted."

def get_top_risk_factors(top_n=5):
    mean_abs_shap = np.abs(shap_values_xgb.values).mean(axis=0)
    risk_factors = sorted(zip(X.columns, mean_abs_shap), key=lambda x: x[1], reverse=True)[:top_n]
    return str(dict(risk_factors))

def compare_patient_to_average(patient_index):
    try:
        patient_index = int(patient_index)
    except Exception:
        return "Patient index must be an integer."
    if patient_index < 0 or patient_index >= len(raw_data):
        return "Invalid patient index."
    patient_data = raw_data.iloc[patient_index]
    avg_data = raw_data[numeric_columns].mean()
    comparison = pd.DataFrame({
        "Patient": patient_data[numeric_columns],
        "Average": avg_data
    })
    comparison.plot(kind="bar", figsize=(10, 6))
    plt.title("Patient vs. Average Feature Values (Raw Data)")
    plt.ylabel("Value")
    plt.show()
    return comparison.to_string()

def get_feature_statistics(feature_name):
    if feature_name not in raw_data.columns:
        return f"Feature {feature_name} not found in dataset."
    stats = raw_data[feature_name].describe()
    plt.figure(figsize=(8, 6))
    sns.histplot(raw_data[feature_name], kde=True, color='coral')
    plt.title(f"Distribution and Statistics of {feature_name} (Raw Data)")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.show()
    return str(stats.to_dict())

def get_model_parameters():
    return str(model.get_params())

response = None
def run(model: str, question: str):
    client = ollama.Client()
    messages = [{"role": "user", "content": question}]
    
    tools = [
         {
                "type": "function",
                "function": {    
                    "name": "get_model_accuracy",
                    "description": "Retrieves the accuracy of the model",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }

                },
            },
            {
                "type": "function",
                "function": {    
                    "name": "get_roc_auc",
                    "description": "Calculates the ROC AUC score and returns it",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }

                },
            },
            {
                "type": "function",
                "function": {    
                    "name": "explain_instance",
                    "description": "Explains a particular instance's output using SHAP values and generates a waterfall plot.",
                    "parameters": {
                        "type": "object",
                        "required": [
                            "instance_index"
                        ],
                        "properties": {
                            "instance_index": {
                                "type": "number",
                                "description": "The index of the instance to explain"
                            }
                        },
                    }

                },
            },
            {
                "type": "function",
                "function": {    
                    "name": "get_feature_importance",
                    "description": "Generates a summary plot of feature importance using SHAP values.",
                    "parameters": {
                        "type": "object",
                        "required": [],
                        "properties": {},
                    }
                },
            },
            {
                "type": "function",
                "function": {    
                    "name": "get_feature_impact",
                    "description": "Generates a SHAP dependence plot for a given feature",
                    "parameters": {
                        "type": "object",
                        "required": [
                            "feature_name"
                        ],
                        "properties": {
                            "feature_name": {
                                "type": "string",
                                "description": "The name of the feature to analyze"
                            }
                        },
                    }

                },
            },
        {
            "type": "function",
            "function": {    
                "name": "get_shap_value",
                "description": "Returns SHAP values for a specified instance as a dictionary mapping features to SHAP values.",
                "parameters": {
                    "type": "object",
                    "required": ["instance_index"],
                    "properties": {
                        "instance_index": {
                            "type": "number",
                            "description": "The index of the instance."
                        }
                    },
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_confusion_matrix",
                "description": "Computes and plots the confusion matrix for the test set predictions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_classification_report",
                "description": "Prints and returns the classification report for the test set predictions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_precision_score",
                "description": "Returns the precision score for the test set predictions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_recall_score",
                "description": "Returns the recall score for the test set predictions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_f1_score",
                "description": "Returns the F1 score for the test set predictions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "plot_feature_distribution",
                "description": "Plots the distribution of a specified feature from the dataset.",
                "parameters": {
                    "type": "object",
                    "required": ["feature_name"],
                    "properties": {
                        "feature_name": {
                            "type": "string",
                            "description": "The name of the feature to plot."
                        }
                    }
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "plot_data_correlation",
                "description": "Plots a heatmap of the correlation matrix for the features in the dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_learning_curve",
                "description": "Plots the learning curve of the model using the entire dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "save_model",
                "description": "Saves the trained model to disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Optional filename to save the model (default is 'xgb_model.pkl')."
                        }
                    },
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "load_model",
                "description": "Loads a model from disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Optional filename from which to load the model (default is 'xgb_model.pkl')."
                        }
                    },
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "predict_readmission",
                "description": "Predicts readmission for a new patient based on provided data.",
                "parameters": {
                    "type": "object",
                    "required": ["new_data"],
                    "properties": {
                        "new_data": {
                            "type": "object",
                            "description": "Dictionary with keys matching the feature columns."
                        }
                    }
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_patient_profile",
                "description": "Retrieves the original record and SHAP explanation summary for a patient.",
                "parameters": {
                    "type": "object",
                    "required": ["patient_index"],
                    "properties": {
                        "patient_index": {
                            "type": "number",
                            "description": "The index of the patient in the dataset."
                        }
                    }
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "analyze_readmission_trends",
                "description": "Analyzes and plots readmission rates grouped by a specified column.",
                "parameters": {
                    "type": "object",
                    "required": ["by_column"],
                    "properties": {
                        "by_column": {
                            "type": "string",
                            "description": "The column name by which to group (e.g., 'month', 'specialty')."
                        }
                    }
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "plot_readmission_distribution",
                "description": "Plots the distribution of the readmission indicator.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_top_risk_factors",
                "description": "Identifies the top risk factors for readmission based on average absolute SHAP values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "number",
                            "description": "Optional number of top factors to return (default is 5)."
                        }
                    },
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "compare_patient_to_average",
                "description": "Compares a patient's numeric feature values to the overall dataset average.",
                "parameters": {
                    "type": "object",
                    "required": ["patient_index"],
                    "properties": {
                        "patient_index": {
                            "type": "number",
                            "description": "The index of the patient."
                        }
                    }
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_feature_statistics",
                "description": "Returns descriptive statistics and plots the distribution for a specified feature.",
                "parameters": {
                    "type": "object",
                    "required": ["feature_name"],
                    "properties": {
                        "feature_name": {
                            "type": "string",
                            "description": "The name of the feature."
                        }
                    }
                }
            },
        },
        {
            "type": "function",
            "function": {    
                "name": "get_model_parameters",
                "description": "Returns the parameters of the trained XGBoost model.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
    ]
    
    response = client.chat(
        model=model,
        messages=messages,
        tools=tools,
    )
    
    # Add the model's response to the conversation history.
    messages.append(response["message"])
    
    # If the model did not use any function, output its response.
    if not response["message"].get("tool_calls"):
        print("The model didn't use a function. Its response was:")
        print(response["message"]["content"])
        return
    
    # Map tool function names to actual functions.
    available_functions = {
        "get_model_accuracy": get_model_accuracy,
        "get_roc_auc": get_roc_auc,
        "explain_instance": explain_instance,
        "get_feature_importance": get_feature_importance,
        "get_feature_impact": get_feature_impact,
        "get_shap_value": get_shap_value,
        "get_confusion_matrix": get_confusion_matrix,
        "get_classification_report": get_classification_report,
        "get_precision_score": get_precision_score,
        "get_recall_score": get_recall_score,
        "get_f1_score": get_f1_score,
        "plot_feature_distribution": plot_feature_distribution,
        "plot_data_correlation": plot_data_correlation,
        "get_learning_curve": get_learning_curve,
        "save_model": save_model,
        "load_model": load_model,
        "predict_readmission": predict_readmission,
        "get_patient_profile": get_patient_profile,
        "analyze_readmission_trends": analyze_readmission_trends,
        "plot_readmission_distribution": plot_readmission_distribution,
        "get_top_risk_factors": get_top_risk_factors,
        "compare_patient_to_average": compare_patient_to_average,
        "get_feature_statistics": get_feature_statistics,
        "get_model_parameters": get_model_parameters,
    }
    
    # Process each function call returned by the model.
    if response["message"].get("tool_calls"):
        for tool in response["message"]["tool_calls"]:
            func_name = tool["function"]["name"]
            function_to_call = available_functions.get(func_name)
            if not function_to_call:
                continue
            function_args = tool["function"].get("arguments", {})
            function_response = function_to_call(**function_args)
            # Add function response to conversation history.
            messages.append({
                "role": "tool",
                "content": function_response,
            })
    
    final_response = client.chat(model=model, messages=messages)
    print(final_response["message"]["content"])
    return final_response["message"]["content"]

question = "What is the model accuracy?"
run("llama3.2:1b", question)