import os, sys, json
cd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cd, '..')))
from model import Data_Model

tools = []

function_list = [func for func in dir(Data_Model) if callable(getattr(Data_Model, func)) and not func.startswith('_')]

for func in function_list:
    tool_dict = {"type":"function"}
    function_dict = {}
    function_dict["name"] = func
    function_dict["description"] = getattr(Data_Model, func).__doc__
    function_dict["parameters"] = {"type": "object", "properties": {}, "required": [], 
                                   #"additionalProperties": False
                                   }
    #function_dict["strict"] = True
    tool_dict["function"] = function_dict
    tools.append(tool_dict)

with open("tools.json", "w") as tool_output:
    json.dump(tools, tool_output)
