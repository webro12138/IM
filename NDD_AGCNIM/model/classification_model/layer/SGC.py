
import tensorflow as tf
import networkx as nx
import numpy as np
def get_init_Variable(shape, initializer, dtype, name):
    if(isinstance(initializer, str)):
        initializer = tf.keras.initializers.get(initializer)
    else:
        initializer = initializer()
    
    return tf.Variable(initializer(shape, dtype=dtype), name=name)
    

class SGC(tf.keras.layers.Layer):
    
    def __init__(self,filters, kernel_initializer, activation, bias_initializer, K = 2):
        super(SGC, self).__init__()
        self.kernel_initializer = kernel_initializer        
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.filters = filters
        self.K = K
        
        
    def build(self, input_shape):
        
        self.kernel = get_init_Variable([input_shape[1] * self.K, self.filters], initializer=self.kernel_initializer, dtype=tf.float32, name="kernel")
        self.bias = get_init_Variable([self.filters], initializer=self.bias_initializer, dtype=tf.float32, name="bias")
        #self.W_d = get_init_Variable([input_shape[1] * self.K, self.filters], initializer=self.kernel_initializer, dtype=tf.float32, name="W_d")
        
    def call(self, x, graph):
        return self.specgraph_LL(graph, x)
    
    def specgraph_LL(self, graph, x):
        
        laplacian = nx.normalized_laplacian_matrix(graph).A.astype(np.float32)
     
        n_features = x.shape[1]
        x0 = x
        x = tf.expand_dims(x0, 0)  
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin
           
            return tf.concat([x, x_], 0)
            
        if self.K > 1:
            x1 = tf.matmul(laplacian, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.matmul(laplacian, x1) - x0  # M x Fin
            x = concat(x, x2)
            x0, x1 = x1, x2
            
        M = laplacian.shape[0]#tf.squeeze((tf.transpose(i, perm=[1, 0]), 0))
        shape = tf.stack([self.K, M, n_features])
        shape2 = tf.stack([M, self.K * n_features])
        x = tf.reshape(x, shape)
            
        x = tf.transpose(x, perm=[1, 2, 0])  # x -> M x Fin x K
        x = tf.reshape(x, shape2)  # x-> M x (Fin*K)
        x = tf.matmul(x, self.kernel) + self.bias  # x -> M x Fout + Fout
        
        return x
    def func(self, x):
        
        ##x_w = tf.matmul(x, M)
        M = tf.matmul(self.W_d ,self.W_d)
    
        D = [[] for _ in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x)):
                """if j == i:
                    D[i].append(0.0)
                    continue
                u = x_w[i]
                v = x_w[j]
                
                dist = np.linalg.norm(u - v)
                value = 1 * np.exp(-1 * dist)
                
                D[i].append(value)
                """
                x_val = x[i] - x[j]
                dist = np.array(x_val).T @ np.array(M) @ np.array(x_val)
                value = 1 * np.exp(-1 * dist)
                D[i].append(value)
        
        W = np.asarray(D).astype(np.float32)
        adj_m = W
                    
        d = W.sum(axis=0)  # degree for each atom
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = np.diag(d.squeeze())  # D^{-1/2}
        I = np.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
                    
        return L.astype(np.float32), adj_m.astype(np.float32)   