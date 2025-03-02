from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from chatbot import Chatbot
import os
import json
import re

app = Flask(__name__)
CORS(app)
chatbot = Chatbot()

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input from the request
    user_input = request.json.get('input')

    # Process the input and generate response along with plot path
    response_text, plot_path = process_input(user_input)

    # Check if there's a SHAP plot to send
    if plot_path:
        return jsonify({"response": response_text, "plot": plot_path})
    else:
        return jsonify({"response": response_text})

def process_input(user_input):
    # Here you can add your AI or logic to process the input
    response = chatbot.run(user_input)

    #with open("test.json", "w") as file:
    #    json.dump(response, file)

    #response = "<pre>" + response["explanation"] + "\n" + response["answer"] + "</pre>"

    response = re.sub(r'\n', '<br/>', response)
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)

    # Check if the input triggers SHAP plot request (customize as needed)
    if "shap" in response.lower() and "force" in response.lower():
        # Define your plot file path (absolute path)
        shap_plot_path = "force_plot.html"
        return response, shap_plot_path
    else:
        return response, None
    
@app.route('/get_plot/<filename>', methods=['GET'])
def get_plot(filename):
    return send_file(os.path.join('plots', filename), mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
