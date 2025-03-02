from pydantic import BaseModel
from openai import OpenAI
import os
import json
from model import Data_Model

cd = os.path.dirname(os.path.abspath(__file__))
IRIS_DATA_FILE = os.path.join(cd, "data", "iris_dataset.csv")
IRIS_KNN_MODEL_FILE = os.path.join(cd, "models", "iris_knn_model.pkl")
TOOLS_FILE = os.path.join(cd, "tools", "tools.json")
API_KEY_PATH = os.path.join(cd, "keys", "gpt_api_key.txt")

class Chatbot:
    def __init__(self):
        self._iris_model = Data_Model(IRIS_DATA_FILE, IRIS_KNN_MODEL_FILE)
        with open(API_KEY_PATH, "r") as key:
            self._client = OpenAI(api_key=key.read().strip())
        with open(TOOLS_FILE, "r") as tools_file:
            self._tools = json.load(tools_file)
        self._system = "You are an assistant who can intelligently answer any question regarding the descriptive statistics on the Iris Dataset and a Trained KNN Model."
        self._CoT_Header = "Lets think this step by step."
        self._CoT_Response_Header = ""

    def _execute(self, name):
        result = ""

        if 'shap' in name:
            result += f"\nShap Plot Displayed\n{getattr(self._iris_model, name, None)()}\n"
        else:
            result += f"\n{name}:\n{getattr(self._iris_model, name, None)()}\n"
        
        return result
    
    class _output(BaseModel):
        how_to_solve_explanation: str
        answer: str

    def test_execute(self, reply):
        print(self._execute(reply))
    
    def load_system(self, message):
        self._system = message

    def load_CoT_Header(self, header):
        self._CoT_Header = header

    def load_CoT_Response_Header(self, header):
        self._CoT_Response_Header = header
    
    def run(self, question):

        messages = [{"role": "system", "content": self._system}, {"role": "user", "content": question + "\n" + self._CoT_Header},]

        completion = self._client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self._tools
        )

        messages.append(completion.choices[0].message)

        if completion.choices[0].message.tool_calls:
            for tool_call in completion.choices[0].message.tool_calls:
                name = tool_call.function.name

                result = self._execute(name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        format = self._output

        completion_2 = self._client.chat.completions.create(
            model="gpt-4o",
            messages=messages, 
            tools=self._tools,
            #response_format=format
        )

        output = completion_2.choices[0].message.content

        if output:
            return output
        else:
            return "I'm sorry. I am not able to answer that question right now"

        output = completion_2.choices[0].message

        # If the model refuses to respond, return the refusal message
        if output.refusal:
            return {"explanation": "Response refused", "answer": output.refusal}
        else:
            # Return structured output as a dictionary (JSON-like format)
            return {
                "explanation": output.parsed.how_to_solve_explanation,
                "answer": output.parsed.answer
            }

