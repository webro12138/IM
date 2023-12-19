import numpy as np
def array_to_writable_str(data):
    data_array = np.array(data)
    
    shape = data_array.shape
    lines = ""
    assert len(shape) <= 2, "The shape of the wroten data must be no more than 2"
    if(len(shape) == 1):
       lines = vec_to_string(data)
    if(len(shape) == 2):
        for i in range(shape[0] - 1):
            lines = lines + vec_to_string(data_array[i]) + "\n"
        lines = lines + vec_to_string(data_array[shape[0] -1])
    return lines
def vec_to_string(vec):
    vec_string = ""
    for i in range(len(vec) - 1):
        vec_string = vec_string + str(vec[i]) + " "
    vec_string = vec_string + str(vec[len(vec) - 1])
    return vec_string

def dict_to_writable_str(D:dict):
    vec_string = ""
    keys = list(D.keys())
    for i in range(len(keys)): 
        t = type(D[keys[i]])
        if(t is int or t is float ):
            vec_string = vec_string + str(keys[i]) + " " + str(D[keys[i]])
        if(t is np.ndarray or t is list):
            vec_string = vec_string + str(keys[i]) + " " + vec_to_string(D[keys[i]])
        if(i < len(keys) - 1):
             vec_string =  vec_string + "\n"
    return vec_string

def write_to_file(data, path, way="w"):
    f = open(path, way)
    t = type(data)
    if(t is np.ndarray or t is list):
        f.write(array_to_writable_str(data))
    if(t is dict):
        f.write(dict_to_writable_str(data))
    f.close()

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
            data[(data_line[0])] = line[0]
        else:
            data[(data_line[0])] = np.array(line)
    
    f.close()
    return data




def read_array(path):
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
    
    return np.array(data)
    
    