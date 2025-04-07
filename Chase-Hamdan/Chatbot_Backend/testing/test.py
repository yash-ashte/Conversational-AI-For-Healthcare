from chatbot import Chatbot
import json

bot = Chatbot()


while True:
    question = input("User: ")
    if question.lower() in ['quit', 'exit']:
        break
    
    output_json = bot.run(question)

   #print(f"Test: {output_json}")

    print(f"ChatGPT:")

    print(output_json['comprehensive_output_answer'])
    with open('test.json', 'w') as file:
        json.dump(output_json, file)

    continue