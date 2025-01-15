import openai
import shap
import re
import queue
from descript import *
from shap_file import *
from what_if_file import *

output_queue = queue.Queue()

openai.api_key = 'sk-proj-WGnZ1zTTZ3XN2tmrL4UCT3BlbkFJAnikYbZs5LQVg7fxTaOp'

iris_data = load_iris_dataset()
df =  iris_data.select_dtypes(include='number')
#print(df.head())
model = load_knn_model()
X = iris_data.drop(['Id','Species'], axis=1)  # Assuming 'species' is the target label column
y = iris_data['Species']
cols = X.columns

y = pd.Categorical(y).codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#y = scaler.transform(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)





def execute(reply, df):
    result = ""
    if "what_if_operation" in reply or  "what_if_for_id" in reply:
        '''
        feature_name_mapping = {
        'sepal length': 'SepalLengthCm',
        'sepal width': 'SepalWidthCm',
        'petal length': 'PetalLengthCm',
        'petal width': 'PetalWidthCm'
        }'''
        pattern = r"(?<=\().+?(?=\))"
        parameters = re.search(pattern, reply)
        if parameters:
            param_list = [param.strip() for param in parameters.group(0).split(",")]
            #result += param_list

        feature_name = param_list[1].strip().replace("'", "")
        #update_term = update_term.strip().replace("'", "")
        '''if feature_name_user in feature_name_mapping:
            feature_name = feature_name_mapping[feature_name_user]
        else:
            raise ValueError(f"Unknown feature name: {param_list[1].lower()}")
        '''
        if  "what_if_operation" in reply:
            result += "What if:\n"
            print( "What if:\n")
            if (param_list[4].lower() in ['true', '1', 'yes', 'y']):
                is_numer = True
            else:
                is_numer = False
            if (param_list[5].lower() in ['true', '1', 'yes', 'y']):
                is_cater = True
            else:
                is_cater = False
            
            if is_numer:
                df = what_if_operation(df, feature_name, param_list[2], int(param_list[3]), is_numer, is_cater)
            elif is_cater:
                df = what_if_operation(df, feature_name, param_list[2], param_list[3], is_numer, is_cater)
            result += "updated dataset\n"
            print("updated dataset\n")
        elif "what_if_for_id" in reply:
            df = what_if_for_id(iris_data, feature_name, int(param_list[2]), float(param_list[3]))
    
    #if "what_if_for_id" in reply:
       # df = what_if_for_id(df, row_id, feature_name, new_value)

    if "calculate_mean" in reply:
        result += f"Mean:\n{calculate_mean(df)}\n"
        print(f"Mean:\n{calculate_mean(df)}\n")
    
    if "calculate_median" in reply:
        result += f"Median:\n{calculate_median(df)}\n"
        print(f"Median:\n{calculate_median(df)}\n")
    
    if "calculate_std" in reply:
        result += f"Standard Deviation:\n{calculate_std(df)}\n"
        print(f"Standard Deviation:\n{calculate_std(df)}\n")
    
    if "calculate_variance" in reply:
        result += f"Variance:\n{calculate_variance(df)}\n"
    
    if "calculate_mode" in reply:
        result += f"Mode:\n{calculate_mode(df)}\n"
    
    if "calculate_accuracy" in reply:
        result += f"Accuracy:\n{calculate_accuracy(y, y_pred)}\n"
        print(f"Accuracy:\n{calculate_accuracy(y, y_pred)}\n")
    
    if "calculate_f1" in reply:
        result += f"F1 Score:\n{calculate_f1(y, y_pred)}\n"
    
    if "calculate_precision" in reply:
        result += f"Precision:\n{calculate_precision(y, y_pred)}\n"
    
    if "calculate_recall" in reply:
        result += f"Recall:\n{calculate_recall(y, y_pred)}\n"
    
    if "calculate_roc_auc" in reply:
        #print("ROC AUC Score:")
        #print(calculate_roc_auc(y, y_proba))
        result += f"ROC AUC Score:\n{calculate_roc_auc(y, y_proba)}\n"
    
    if "generate_confusion_matrix" in reply:
        #print("Confusion Matrix:")
        #print(generate_confusion_matrix(y, y_pred))
        result += f"Confusionn Matrix:\n{generate_confusion_matrix(y, y_pred)}\n"
    
    if "shap_for_one" in reply:
        result += "Shap Plot has been saved to shap_one.html:\n"
        shap_for_one(model, X_test, X_train,cols)

    if "shap_for_all" in reply:
        result += "Shap Plot has been saved to shap_all.html:\n"
        shap_for_all(model, X_test, X_train,cols)
    
    if "shap_summary" in reply:
        result += "Shap plot displyed\n"
        shap_summary(model, X_test, X_train,cols)

    
    if "not" in reply:
        result += "Function not recognized or available.\n"
    
    return result





def run_chatbot(df):
    messages = [
        {"role": "user", "content": "You are a assistant who I can ask anything about the Iris dataset descriptive statistcs. Thi"\
            "I have written the following function definitions: calculate_mean(df),calculate_median(df),calculate_std(df),calculate_variance(df),calculate_mode(df),calculate_accuracy(y, y_pred),calculate_f1(y, y_pred),calculate_precision(y, y_pred),calculate_recall(y, y_pred),calculate_roc_auc(y, y_proba),generate_confusion_matrix(y, y_pred)."\
            "shap_for_one(model, X_test, X_train,X.columns),shap_for_all(model, X_test, X_train,X.columns),shap_summary(model, X_test, X_train,X.columns)."\
            "Another feature I want to implement is the 'what if' feature, if the user asks any such question, i have the following function definition: what_if_operation(df, feature_name_to_be_updated, update_term, update_value, is_numeric, is_categorical). update_term can either be 'increase','decrease; or 'set'."\
            "update_value is the value to be updated by and based of if the feature name has numeric or categorical data, set the booleans for is_numeric and is_categorical. Note that if is_categorical is true, update term can only be set and update value also has to be categorical."\
            "and what_if_for_id(temp_data, feature_name, row_id, new_value) for 1 specific row to be updated, For the feature names, make sure they are one of these: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm  "\
            "Based on the question at the start, give me what function to run. JUST GIVE THE FUNCTION NAME with df parameter. if the question doesn't ask for any of these, reply by saying Answer not available yet. If the question asks for more than 1 statistic, give me comma separated functions. If the question references one of the old questions, give me comma separated all the necessary old functions first adn then the new one."
        }
    ]

    #message = ""
    entry = ""
    while True:
        #if not output_queue.empty():
            #new_message = output_queue.get()
        
            #if new_message:
                #message += new_message
        entry = input("User : ")
        message = entry + ""
        if message.lower() in ["exit", "quit", "bye"]:
            print("Exiting the chat.")
            break
        if message:

            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            reply = chat.choices[0].message.content
            print(f"ChatGPT: ")
            print(reply)
            output = execute(reply, df)
            messages.append({"role": "assistant", "content": reply})
            #output_queue.put(output)
            #print(output)

def main():
    run_chatbot(df)

if __name__ == '__main__':
    main()
