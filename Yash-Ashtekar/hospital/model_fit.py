import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib as plt
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam

# Step 1: Parse the text file
def parse_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    pairs = data.split('--------------------\n')
    structured_data = []

    for pair in pairs:
        if not pair.strip():
            continue
        lines = pair.splitlines()
        #for line in lines[2:4]:
        #print(lines[2:4])
        #print("end")
        # Parse xi
        try:
            xi = np.fromstring(lines[1].split(': ')[1].strip("[]"), sep=' ')
        except (IndexError, ValueError):
            print(f"Skipping xi parsing due to format error in pair: {lines}")
            continue
        
        # Parse D (distance matrix)
        try:
            D_lines = [lines[2:4][0].split(': ')[1] + lines[2:4][1]]  # Extract only the numeric part
            #for row in D_lines:
            #    print(row)
            #print(D_lines)
            for row in D_lines:
                D_nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", row)
            D =  np.array(D_nums, dtype=float)
            #print(D)
            #D = np.array([list(map(float, row.strip("[]").split())) for row in D_lines])
        except (IndexError, ValueError):
            print(f"Skipping D parsing due to format error in pair: {lines}")
            continue
        
        # Parse K
        try:
            K = np.fromstring(lines[4].split(': ')[1].strip("[]"), sep=' ')
        except (IndexError, ValueError):
            print(f"Skipping K parsing due to format error in pair: {lines}")
            continue
        
        # Parse solution
        try:
            solution_lines = [lines[5:][0].split(': ')[1] + lines[5:][1]]
            for S in solution_lines:
                S_nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", S)
            solution =  np.array(S_nums, dtype=float)
            #solution = np.array([list(map(float, row.split())) for row in solution_lines])
        except (IndexError, ValueError):
            print(f"Skipping solution parsing due to format error in pair: {lines}")
            continue
        
        # Append to structured data
        structured_data.append({'xi': xi, 'D': D, 'K': K, 'solution': solution.flatten()})
        #print(f"Total processed pairs: {len(structured_data)}")

    return pd.DataFrame(structured_data)


# Load data
file_path = "output_pairs.txt"  # Path to your data file
data = parse_data(file_path)

#print(data.head)

# Step 2: Feature engineering
data['xi_1'], data['xi_2'] = data['xi'].apply(lambda x: x[0]), data['xi'].apply(lambda x: x[1])
data['D_1_1'], data['D_1_2'], data['D_2_1'], data['D_2_2'] = zip(*data['D'].apply(lambda x: x.flatten()))
data['K_1'], data['K_2'] = data['K'].apply(lambda x: x[0]), data['K'].apply(lambda x: x[1])

X = data[['xi_1', 'xi_2', 'D_1_1', 'D_1_2', 'D_2_1', 'D_2_2', 'K_1', 'K_2']]
y = np.vstack(data['solution'])

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sample = X.iloc[0:1]
print("sample to be predicted:\n")
print(sample)
print(f"Solution: {y[0]}")
print("\n")

#RANDOM FOREST MODEL
print("RANDOM FOREST MODE\n")

# Step 4: Model fitting
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Calculate R² score

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict a sample
  # Modify as needed
prediction = model.predict(sample)
print(f"Prediction: {prediction}")
print("----------------------------------------------------\n")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

shap.initjs()
#background = shap.sample(X_train, 50)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[0,:])
fp = shap.force_plot(explainer.expected_value[0], shap_values[:,0], X_test[0,:])#, feature_names=feature_names
#plt.show()
shap.save_html('shap_one.html', fp)

#shap.force_plot(explainer.expected_value[0], shap_values[...,0], X_test)
#shap.save_html('shap_all.html', fp)
print("here")
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
subset = X_test.sample(100, random_state=42)  # Reduce the size to 100 samples
shap_values_subset = explainer.shap_values(subset)

# Summary plot
shap.summary_plot(shap_values_subset, subset)
#shap.save_html('shap_summary.html', fp)
#shap_values = explainer.shap_values(X_test)
print("here2")
#shap.summary_plot(shap_values, X_test)

#shap_values = explainer.shap_values(X_test)
shap.dependence_plot(0, shap_values, subset)
print("here3")
#shap.save_html('shap_dependence.html', fp)
'''

print("LINEAR REGRESSION\n")
model = LinearRegression()  # or Ridge(), Lasso(), ElasticNet()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict a sample
prediction = model.predict(sample)
print(f"Prediction: {prediction}")
print("----------------------------------------------------\n")


print("DECISION TREE\n")
model = DecisionTreeRegressor()  # or GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict a sample
prediction = model.predict(sample)
print(f"Prediction: {prediction}")
print("----------------------------------------------------\n")
'''
'''
print("Neural Network\n")
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict a sample
prediction = model.predict(sample)
print(f"Prediction: {prediction}")
print("----------------------------------------------------\n")
print("MLP Classifier\n")
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
# If y_train is a DataFrame, convert to a 1D array
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.squeeze()

# Ensure y_train is 1D
y_train = np.ravel(y_train)
# Fit the model (X_train and y_train should be your training data and labels)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

hidden_layers = [100, 50]  # Number of neurons in each hidden layer
activation = 'relu'        # Activation function: 'relu', 'tanh', 'sigmoid', etc.
learning_rate = 0.001      # Learning rate for optimizer
batch_size = 32            # Batch size for training
epochs = 100               # Number of epochs for training

print("TENSORFLOW NEURAL NETWORK WITH CUSTOMIZABLE HYPERPARAMETERS\n")
print(f"Hyperparameters:\nHidden Layers: {hidden_layers}\nActivation: {activation}\nLearning Rate: {learning_rate}\nBatch Size: {batch_size}\nEpochs: {epochs}\n")
model = Sequential()
model.add(Dense(hidden_layers[0], activation=activation, input_dim=X_train.shape[1]))  # Input layer
for units in hidden_layers[1:]:
    model.add(Dense(units, activation=activation))
model.add(Dense(y_train.shape[1]))  # Output dimension matches the number of target variables

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict a sample
prediction = model.predict(sample)
print(f"Prediction: {prediction}")
print("----------------------------------------------------\n")
'''