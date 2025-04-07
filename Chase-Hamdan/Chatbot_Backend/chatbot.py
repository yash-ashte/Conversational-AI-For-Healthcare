from openai import OpenAI
import os
import json
from model import Data_Model
from formats import Response_Output
import base64

cd = os.path.dirname(os.path.abspath(__file__))
IRIS_DATA_FILE = os.path.join(cd, "data", "iris_dataset.csv")
IRIS_KNN_MODEL_FILE = os.path.join(cd, "models", "iris_xgboost_model.pkl")
TOOLS_FILE = os.path.join(cd, "tools", "tools.json")
API_KEY_PATH = os.path.join(cd, "keys", "gpt.key")
LOG_FILE_PATH = os.path.join(cd, "logs", "chatbot_log.json")

class Chatbot:
    def __init__(self):
        """Initialize the chatbot object"""

        # Load API key from hidden file
        with open(API_KEY_PATH, "r") as key:
            self._client = OpenAI(api_key=key.read().strip())

        # Load custom data model
        self._iris_model = Data_Model(IRIS_DATA_FILE, IRIS_KNN_MODEL_FILE)

        # Load data model tool json
        with open(TOOLS_FILE, "r") as tools_file:
            self._tools = json.load(tools_file)

        # Initialize chatbot headers
        self._system = """You are an assistant capable of intelligently answering questions related to the Iris Dataset and a trained XGBoost classifier model. You should always answer questions by thinking step-by-step. Please explain each step before providing the final answer. When asked about descriptive statistics or a trained model, always provide reasoning first, then follow up with the answer. If needed, use available tools for plotting or computations."""
        self._CoT_Header = "Lets think this step by step.\n1. Identify Question intent\n2. Reason and Identify steps needed to answer\n3. Identify and request any tools needed to answer from tool list."


    def _execute(self, name, args_dict:dict):
        """Execute a tool request and return output"""

        if args_dict:
            args = {key: value for key, value in args_dict.items()}
            output = getattr(self._iris_model, name, None)(**args)
        else:
            output = getattr(self._iris_model, name, None)()
            
        if type(output) == str:
            return output, None
        else:
            return output[0], output[1]
        
    def run(self, question):
        """Querie the chatbot and return structured output"""

        # Load initial messages
        messages = [{"role": "system", "content": self._system}, {"role": "user", "content": question + "\n" + self._CoT_Header},]
        
        # Call the chatbot api
        completion = self._client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self._tools,
        )

        # Append response to the message list
        reasoning = completion.choices[0].message
        messages.append(reasoning)

        # Handle each tool request and add to message list
        plots = []
        if completion.choices[0].message.tool_calls:
            for tool_call in completion.choices[0].message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                result = self._execute(name, args)

                if result[1]:
                    plots.extend(result[1])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result[0])
                })

        # Call the chatbot api after handling tool calls
        completion_2 = self._client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages, 
            tools=self._tools,
            response_format=Response_Output
        )

        # Return chatbot output
        output = completion_2.choices[0].message

        # Write chatbot logic to log
        with open(LOG_FILE_PATH, '+a') as log:
            log.write(f'{{\"Question\": \"{question}\", \"Reasoning\": {reasoning.model_dump_json()}, \"Response\": {output.model_dump_json()}}}\n')

        # Process return json
        if output.refusal:
            return {"explanation": "Response refused", "answer": output.refusal}
        if output.content == None:
            return {"comprehensive_output_answer": "Sorry the model has errored! please try a different question.", "plots": []}
        else:
            response = json.loads(output.model_dump()['content'])
            response["plots"] = []
            for plot in plots:
                with open(plot, 'rb') as plot_file:
                    plot_base64 = base64.b64encode(plot_file.read()).decode("utf-8")
                response["plots"].append(f"data:image/png;base64,{plot_base64}")

            return response

