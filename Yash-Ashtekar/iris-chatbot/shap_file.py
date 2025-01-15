import shap
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
import IPython
import matplotlib as plt

def load_knn_model():
    with open("knn_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

def shap_for_one(model, X_test, X_train, feature_names):
    shap.initjs()
    #background = shap.sample(X_train, 50)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test[0,:])
    fp = shap.force_plot(explainer.expected_value[0], shap_values[:,0], X_test[0,:], feature_names=feature_names)
    shap.save_html('shap_one.html', fp)
    #IPython.display.display(fp)
    #plt.show()

def shap_for_all(model, X_test, X_train, feature_names):
    shap.initjs()
    #background = shap.sample(X_train, 50)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    fp = shap.force_plot(explainer.expected_value[0], shap_values[...,0], X_test,feature_names=feature_names)
    shap.save_html('shap_all.html', fp)

def shap_summary(model, X_test, X_train, feature_names):
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test,feature_names=feature_names)

def shap_dependence(model, X_test, X_train, feature_names):
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.dependence_plot(0, shap_values, X_test, feature_names=feature_names)

def main():
    # Load your dataset (assuming it's CSV)
    iris_data = pd.read_csv('./data/iris.csv')

    # Prepare features (X) and labels (y)
    X = iris_data.drop(['Id', 'Species'], axis=1)  # Drop 'Id' and 'Species'
    y = iris_data['Species']
    
    # Encode labels if necessary (replace 'Species' with numeric values)
    y = pd.Categorical(y).codes
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Load the trained KNN model
    model = load_knn_model()

    # Set up SHAP and generate force plot for SHAP values
    shap_for_one(model, X_test, X_train,X.columns)
    shap_for_all(model, X_test, X_train,X.columns)
    shap_summary(model, X_test, X_train,X.columns)

if __name__ == "__main__":
    main()