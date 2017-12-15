# DynamoDB file parser
from os import listdir
from os.path import isfile, join, isdir
import dynamoparser
import pandas as pd
import flatten_json
import time

base_path = "./input"
database_list = [f for f in listdir(base_path) if isdir(join(base_path, f))]


def file_parse(path_to_file):
    mongo_file = open(path_to_file, 'r')
    mongo_lines = mongo_file.read().replace("\"s\"", "\"S\"").replace("\"l\"", "\"L\"").replace("\"n\"", "\"N\"").replace("\"m\"", "\"M\"").replace(" ", "").split("\n")
    converted_lines = dynamoparser.loads(mongo_lines)
    jsons_arrays = []
    for line in converted_lines:
        if len(line) < 2:
            continue
        jsons_arrays.append(flatten_json.flatten_json(dynamoparser.loads(line)))
    return jsons_arrays


for database_folder in database_list:
    folder_path = base_path + "/" + database_folder
    database_date_list = [f for f in listdir(folder_path) if isdir(join(folder_path, f))]
    print(len(database_date_list))
    # ~ 49 folders
    json_array = []
    for database_date_folder in database_date_list:
        date_folder_path = folder_path + "/" + database_date_folder
        files_in_folder_list = [f for f in listdir(date_folder_path) if isfile(join(date_folder_path, f))]
        files_in_folder_list = list(files_in_folder_list)
        
        
        for file_name in files_in_folder_list:
            if (file_name != "manifest") & (file_name != "_SUCCESS"):
                file_path = date_folder_path + "/" + file_name
                print(file_path)
                json_array += file_parse(file_path)

    
    pd.io.json.json_normalize(json_array).to_csv(str("output_" + database_folder + ".csv"), index=False)


