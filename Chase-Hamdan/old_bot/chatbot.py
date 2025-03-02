import openai
import os
from model import Data_Model

cd = os.path.dirname(os.path.abspath(__file__))
IRIS_DATA_FILE = os.path.join(cd, "data", "iris_dataset.csv")
IRIS_KNN_MODEL_FILE = os.path.join(cd, "models", "iris_knn_model.pkl")
API_KEY_PATH = os.path.join(cd, "keys", "gpt_api_key.txt")

class Chatbot:
    def __init__(self):
        with open(API_KEY_PATH, "r") as key:
            openai.api_key = key.read().strip()
        self._iris_model = Data_Model(IRIS_DATA_FILE, IRIS_KNN_MODEL_FILE)
        self._function_list = [func for func in dir(Data_Model) if callable(getattr(Data_Model, func)) and not func.startswith('_')]
        self._content = "You are an assistant who I can ask anything about descriptive statistics on the Iris Dataset and a Trained KNN Model. Function Definitions: " + ",".join(self._function_list)
        self._CoT_Header = ""
        self._CoT_Response_Header = ""

    def _execute(self, reply):
        result = ""

        for func in self._function_list:
            if func in reply:
                if 'shap' in func:
                    result += f"\nShap Plot Displayed\n"
                    getattr(self._iris_model, func)()
                else:
                    result += f"\n{func}:\n{getattr(self._iris_model, func)()}\n"
        
        return result
    
    def test_execute(self, reply):
        print(self._execute(reply))
    
    def load_content(self, message):
        self._content = message

    def load_CoT_Header(self, header):
        self._CoT_Header = header

    def load_CoT_Response_Header(self, header):
        self._CoT_Response_Header = header
    
    def run(self, question):
        output = "Q: " + question + "\n"
        cot_question = self._CoT_Header + "Q: " + question + "\nA: Let's think this step by step. "
        messages = [{"role": "user", "content": self._content}, {"role": "user", "content": cot_question}]
        chat = openai.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
        
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"ChatGPT: {reply}\n")
        output += f"\nChatGpt: {reply}\n"

        function_response = self._execute(reply.strip().split("\n")[-1])

        print(f"FR:{function_response}")
        output += f"\nFR:{function_response}\n"

        if function_response == "": return output

        response_message = self._CoT_Response_Header + "Q: " + question +"\nFunction Response:\n" + function_response + "A: Let's think this step by step. "

        messages.append(
            {"role": "user", "content": response_message},
        )
        chat = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        output += f"ChatGpt: {reply}\n"
        return output




