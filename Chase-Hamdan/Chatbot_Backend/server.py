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
    question = request.json.get('input')

    # Process the input and generate response along with plot path
    output = process_input(question)

    # Return output json string to frontend for display
    return jsonify(output)

def process_input(question):

    # Process query and get chatbot response json
    cb_ouptut = chatbot.run(question)
    
    # Parse chatbot response json into frontend json
    output = {'response': cb_ouptut['comprehensive_output_answer'], 'plots': cb_ouptut['plots']}

    # Convert special characters to html readable format
    output['response'] = re.sub(r'\n', '<br/>', output['response'])
    output['response'] = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', output['response'])

    # Return output json
    return output

if __name__ == '__main__':
    app.run(debug=True)
