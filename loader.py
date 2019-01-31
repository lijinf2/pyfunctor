import csv
import json

def load_csv_dicts(csv_input_path):
    dicts = [];
    fin = open(csv_input_path, "r")
    reader = csv.DictReader(fin)
    for row in reader:
        dicts.append(row)
    return dicts

def array_to_csvline(array):
    result = ""
    is_first = True
    for term in array:
        if is_first:
            is_first = False
        else:
            result += ","
        result += "\"";
        result += str(term)
        result += "\""
    result += "\n"
    return result

def dict_to_csvline(dict_map, header, term_value_map = lambda term, value: str(value)):
    result = ""
    is_first = True
    for term in header:
        if is_first:
            is_first = False
        else:
            result += ","
        result += "\"";
        result += term_value_map(term, dict_map[term])
        result += "\""
    result += "\n"
    return result

def dump_csv_dicts(output_path, header, csv_dicts, term_value_map = lambda term, value: str(value)):
    fout = open(output_path, "w")
    fout.write(array_to_csvline(header))
    for row in csv_dicts:
        string = dict_to_csvline(row, header, term_value_map)
        fout.write(string)
    fout.close()

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
