import json
import numpy as np
import logging
def count_frequency(track):
    result = {}
    for e in track:
        result[e] = result.setdefault(e, 0) + 1
    result = dict(sorted(result.items(), key=lambda x:x[1], reverse=True))
    return result

def read_list_for_txt(path):
    f = open(path, "r")
    data = []
    for line in f.readlines():
        line = line.strip()
        data_line = line.split(" ")
        line = []
        for j in range(len(data_line)):
            if(data_line[j] != ' '):
                line.append(np.double(data_line[j]))
        data.append(line)
    
    f.close()
    return data

def save_json(data, path):
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_json(path):
    with open(path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    return data

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def read_dict(path):
    f = open(path, "r")
    data = {}
    for line in f.readlines():
        line = line.strip()
        data_line = line.split(" ")
        line = []
        
        for j in range(1, len(data_line)):
            if(data_line[j] != ' '):
                if(data_line[j] != "NA"):
                    line.append(np.float32(data_line[j]))
                else:
                    line.append(0)
        if(len(line) == 1):
            data[(int(data_line[0]))] = line[0]
        else:
            data[(int(data_line[0]))] = np.array(line)
    
    f.close()
    return data


def get_logger(name, logging_path):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(logging_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger