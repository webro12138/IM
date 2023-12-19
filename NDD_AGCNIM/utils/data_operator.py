from utils.dataset import Dataset
import os
from utils.get_labels import  get_labels, get_labels_by_pagerank, get_labels_by_degree
import numpy as np
def dataset_to_feed_data(path, range):
    features = []
    graphs = []
    labels = []
    print("load data start")
    count = 0
   
    for i in range:
        count = count + 1
        d = Dataset()
        d.load_graph(path + "dataset-" +str(i) + "/edges.txt", True, 0.1)
        
        features_path = path + "dataset-" + str(i) + "/features.txt"
        if(os.path.exists(features_path)):
            d.load_features(type="load", path = features_path)
        else: 
            d.load_features(type="cal")
            d.save_features(path + "dataset-" + str(i) + "/features.txt")
        
        graphs.append(d.get_graph())
        features.append(d.get_features_list())
        
        labels_path = path + "dataset-" + str(i) + "/labels.txt"
        if(os.path.exists(labels_path)):
            d.load_labels(get_labels, type="load", path = labels_path)
        else:
            d.load_labels(get_labels, "cal")
            d.save_lables(path + "dataset-" + str(i) + "/labels.txt")
        labels.append(d.get_labels_list())
        print("数据集>>>>>>>>dataset-" + str(i) + " 加载成功")     
        
    return features, graphs, labels


def normalize(features):
    if(type(features[0][0]) == list):
        for i in range(len(features)):
            for j in range(len(features[i])):
                features[i][j] = (features[i][j] - np.min(features[i][j])) / (np.max(features[i][j] - np.min(features[i][j])))


        for i in range(len(features)):
            for j in range(len(features[i])):
                features[i][j] = (features[i][j] - np.mean(features[i][j])) / np.std(features[i][j])
        
    else:
        for j in range(len(features)):
            features[j] = (features[j] - np.min(features[j])) / (np.max(features[j] - np.min(features[j])))
                
        for j in range(len(features)):
                features[j] = (features[j] - np.mean(features[j])) / np.std(features[j])
    return features