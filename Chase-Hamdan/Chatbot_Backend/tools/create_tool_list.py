import os, sys, json, inspect, re, ast

cd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cd, '..')))
from model import Data_Model

def python_type_to_json_type(python_type):
    """Map Python types to JSON Schema types"""
    if python_type == 'int':
        return "integer"
    elif python_type == 'float':
        return "number"
    elif python_type == 'bool':
        return "boolean"
    elif python_type == 'str':
        return "string"
    else:
        return "string"
    
def parse_docstring(docstring):
    """Parse Data Model function docstring to get descriptions and parameters"""

    # Regex search patterns to find description, args, and return data
    description_pattern = r'^(.*?)(?=Args:)'
    args_pattern = r'Args:(.*?)(?=Returns:)'
    returns_pattern = r'Returns:(.*)' 

    # Apply search patterns
    description = re.search(description_pattern, docstring, re.DOTALL)
    args = re.search(args_pattern, docstring, re.DOTALL)
    returns = re.search(returns_pattern, docstring, re.DOTALL)

    # Extract the matched groups
    parsed_description = description.group(1).strip() if description else ""
    parsed_args = args.group(1).strip() if args else ""
    parsed_returns = returns.group(1).strip() if returns else ""

    # Parse argument sectiont to get individual options
    args_dict = {}
    arg_pattern = r'(\w+)\s*\((\w+)\):\s*(.*?)(?=\n|$)'
    arg_names = []
    for match in re.finditer(arg_pattern, parsed_args):
        arg_name, arg_type, arg_desc = match.groups()
        arg_names.append(arg_name)
        if "Enum[" in arg_desc:
            desc_split = arg_desc.split("Enum")
            arg_desc = desc_split[0]
            enum_list = ast.literal_eval(desc_split[1])
            args_dict[arg_name] = {
                'type': python_type_to_json_type(arg_type),
                'enum': enum_list,
                'description': arg_desc.strip()
            }

        else:
            args_dict[arg_name] = {
                'type': python_type_to_json_type(arg_type),
                'description': arg_desc.strip()
            }

    # Return parsed json
    return {
        'description': parsed_description + ' Returns ' + parsed_returns,
        'parameters': {
            'type': 'object',
            'properties': args_dict,
            'required': arg_names,
        },
    }

# Initialize tool json and get list of all public functions
tools = []
function_list = [
    func for func in dir(Data_Model)
    if callable(getattr(Data_Model, func)) and not func.startswith('_')
]

# Generate tool json for each public function in Data Model
for func_name in function_list:
    func = getattr(Data_Model, func_name)
    sig = inspect.signature(func)
    doc = func.__doc__ or ""
    parsed_doc = parse_docstring(doc)

    parsed_doc['name'] = func_name

    # For Formatted outputs
    parsed_doc['strict'] = True
    parsed_doc['parameters']['additionalProperties'] = False

    tools.append({"type": "function", "function": parsed_doc})

# Write the tool json to the output fil
with open("tools.json", "w") as f:
    json.dump(tools, f, indent=2)
