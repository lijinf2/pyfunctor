import json

def load_json_dicts(json_input_path):
    dicts = [];
    fin = open(json_input_path, "r")
    for row in fin.readlines():
        dicts.append(json.loads(row))
    return dicts

def load_json_dicts_with_filter(json_input_path, bool_func):
    dicts = []
    fin = open(json_input_path, "r")
    for json_row in fin.readlines():
        dict_row = json.loads(json_row)
        if bool_func(dict_row):
            dicts.append(dict_row)
    return dicts
