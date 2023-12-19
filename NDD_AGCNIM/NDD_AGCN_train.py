import numpy as np
import tensorflow as tf
from model.classification_model.layer.SGC import SGC
from model.classification_model.TFGCN import get_TFGCN
from utils.dataset import Dataset
from utils.data_operator import dataset_to_feed_data, normalize
import matplotlib.pyplot as plt
import networkx as nx


path = r"/home/webro/code/python/agcnim/data/synthesized_dataset/train_data3/" 

# lrange = []
# for i in range(0, 100):
#     lrange += [i, i+100, i+200]

lrange = list(range(0, 200))
#lrange = list(range(0, 222))
features, graphs, labels = dataset_to_feed_data(path, lrange)
features = normalize(features)


features = [np.array(elem) for elem in features]
labels = [np.array(elem) for elem in labels]
model = get_TFGCN()
model.compile(optimizer="adam",
              loss = tf.keras.losses.mse,
              metrics=['mse']
              )


history = model.fit(features, labels,  graphs, epochs=10)
model.save_weights("model.h5")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel("数据集")
plt.ylabel("重要节点预测准确率")
for i in range(len(history["predict_accuarcy"])):
    plt.plot(list(range(history["size"])), history["predict_accuarcy"][i], label="epoch" + str(i))
    plt.xlabel("数据集")
    plt.ylabel("节点预测准确率")

plt.legend()
plt.figure()

plt.xlabel("数据集")
plt.ylabel("重要节点预测准确率")    
for i in range(len(history["influntial_node_accuarcy"])):
    plt.plot(list(range(history["size"])), history["influntial_node_accuarcy"][i], label="epoch" + str(i))


plt.legend()
plt.savefig("/home/webro/code/python/agcnim/result/ACGN", format='svg')
# plt.show()