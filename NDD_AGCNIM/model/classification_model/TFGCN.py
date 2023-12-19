from math import floor
from turtle import shape
import tensorflow as tf
import numpy as np
from utils.processbar import evaluate_process_bar, fit_process_bar
from model.classification_model.layer.SGC import SGC
import networkx as nx
class TFGCN(tf.keras.Model):
    
    def __init__(self):
        super(TFGCN, self).__init__()
        self.layer_list = []
    
    def addLayer(self, layer):             
        self.layer_list.append(layer)
        
    def addLayers(self, layers):        

        self.layer_list = layers
     
    def call(self, input, graph):
        output = input
        if(len(self.layer_list) == 0):
            return output
        for layer in self.layer_list:
            if("sgc" in layer.name):
                output = layer(output, graph)
            elif("batch_normalization" in layer.name):
                output = layer(output)
            elif("dense" in layer.name):
                output = layer(output)  
        
        return output   
    
    
    def compile(self, optimizer, loss, metrics=['accuracy']):
        if(isinstance(optimizer, str)):
            self.optimizer = tf.keras.optimizers.get(optimizer)
        
        else:
            self.optimizer = optimizer   
        if(isinstance(loss, str)):
            self.loss = tf.keras.losses.get(loss)
            
        else:
            self.loss = loss
        
        self.metric_list= []
        for metric in metrics:
            if(isinstance(metric, str)):
                self.metric_list.append(tf.keras.metrics.get(metric))
            else:
                self.metric_list.append(metric)
       
       

    def fit(self, features, labels, graphs, epochs=2):
        assert self.optimizer != None and self.loss != None, "model's complie method should be complete before this mothod" 
        history = {"size":len(features), "predict_accuarcy":[], "influntial_node_accuarcy":[]}
        predict_accuarcy_list = []
        influntial_node_accuarcy_list = []
        feature_length = len(features)
        for epoch in range(epochs):
            
            print("\nStart of epoch %d\n"% (epoch, ))
        
            for i in range(feature_length):
                with tf.GradientTape() as tape:
                    
                    y_pred = self.call(features[i], graphs[i])
                    y_pred = tf.reshape(y_pred, shape=(y_pred.shape[0],))
                    
                    y = np.array(labels[i])
                    loss_fn = self.loss(y, y_pred)
                    
                gradients = tape.gradient(loss_fn, self.trainable_variables)
                
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
                
                
                fit_process_bar(floor((i + 1) / 3), floor(feature_length / 3) + 1, np.sum(loss_fn))
         
        return history
       
    def predict(self, x, g):
        output = self.call(x, g)
        return output
    
    
    def evaluate(self, x, g_list, y):
        mse_list = []
        
        for i in range(len(x)):
            output = self.call(x[i], g_list[i])
            output = tf.reshape(output, shape=(output.shape[0],))
            
            mse = np.mean(tf.keras.metrics.mse(y[i], output))
            #evaluate_process_bar(i+ 1, len(x),  mse)
            print(mse)
            mse_list.append(mse)
            
         
        return mse_list
    
    
    def find_seeds(self, x, g, k):
        output = self.call(x, g)
        allNodes = list(g.nodes)
        result = {}
        for i in range(len(allNodes)):
            result[allNodes[i]] = output[i].numpy()
        result = sorted(result.items(), key=lambda x:x[1], reverse=True)
        seeds = []
        for i in range(k):
            seeds.append(result[i][0])
        
        return seeds
            
        
    def accuarcy_mrtric(self, y_true, y_pred):
        correct = 0
        incorrect = 0
        influential_node_acc = 0
        influential_node_num = np.count_nonzero(y_true)
        for j in range(len(y_pred)):
            if(y_pred[j] < 0.5 and y_true[j] == 0):
                correct = correct + 1
            elif(y_pred[j] > 0.5 and y_true[j] == 1):
                correct = correct + 1
                influential_node_acc = influential_node_acc + 1
            else:
                incorrect = incorrect + 1
        
        return correct / (correct + incorrect), influential_node_acc / influential_node_num
    
def get_TFGCN():
    
    model = TFGCN()
    model.addLayers([
    
        SGC(64, tf.keras.initializers.glorot_uniform, 'relu', bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(),

        SGC(64, tf.keras.initializers.glorot_uniform, 'relu', bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(),
 
        SGC(32, tf.keras.initializers.glorot_uniform, 'relu', bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(),

        SGC(16, tf.keras.initializers.glorot_uniform, 'relu', bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(),
        
        SGC(16, tf.keras.initializers.glorot_uniform, 'relu', bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(),
    
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return model