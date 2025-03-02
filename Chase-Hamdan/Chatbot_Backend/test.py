from chatbot import Chatbot
from model import Data_Model

model = Data_Model("data/iris_dataset.csv", "models/iris_knn_model.pkl")

print(model.shap_summary_plot())



"""
bot = Chatbot()

with open("system.txt", "r") as file:
    system = file.read()

#bot.load_system(system)

while True:
    question = input("User: ")
    if question.lower() in ['quit', 'exit']:
        break
    
    output = bot.run(question)

    print(f"ChatGPT: {output}")
"""