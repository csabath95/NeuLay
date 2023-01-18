#import packages
import networkx as nx
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorchtools import EarlyStopping
from scipy import spatial
from scipy.spatial import cKDTree
import scipy.sparse as sp
import time


import json
from networkx.readwrite import json_graph


# measure time
class tictoc():
    def __init__(self):
        self.prev = 0
        self.now = 0
    def tic(self):
        self.prev = time.time()
    def toc(self):
        self.now = time.time()
        #print( "dt(s) = %.3g" %(self.now - self.prev))
        t = self.now - self.prev
        #self.prev = time.time()
        return t
        
tt = tictoc()


## sparse matrix formula
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# introduce a graph

def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

G = read_json_file('./real_data/internet2006_relabeled.json')

A = nx.to_numpy_matrix(G)
N = len(A)
##A = sp.coo_matrix(AA) #sparsification of A

##adjacency_list = (torch.LongTensor(sp.triu(A, k=1).row), torch.LongTensor(sp.triu(A, k=1).col))
adjacency_list = torch.where(torch.triu(torch.tensor(A)))

#model
def c_kdtree(x, r):
    tree = cKDTree(x.detach().numpy())
    return tree.query_pairs(r,output_type = 'ndarray')
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, gain= N**(1/dim))
        
def Distances_kdtree(X, pairs):
        X1 = X[pairs[:,0]]
        X2 = X[pairs[:,1]]
        dX = torch.sum((X1-X2)**2, axis = -1) 
        
        return dX
    
#FDL model
class LayoutLinear(nn.Module):
    def __init__(self, weight):
        super(LayoutLinear, self).__init__()
        self.weight = weight
        
    def forward(self, inp):
        x = torch.spmm(inp, self.weight)
        
        return  x
    

#input, output dimensions
dim = 3 
x = torch.eye(N) #.to(device)

##x = sp.eye(N)
##x = x.tocoo()
##x = sparse_mx_to_torch_sparse_tensor(x)


#loss function
radius = .4
magnitude = 100*N**(1/3)*radius
k = 1

def custom_loss(output, Dist):
    X = output
    X1 = X[adjacency_list[0]] 
    X2 = X[adjacency_list[1]] 
        
    V_el = (k/2)*torch.sum( torch.sum((X1-X2)**2, axis = -1))
    V_nn = magnitude * torch.sum(torch.exp(-Dist/4/(radius**2)))
   
    return V_el + V_nn
    
    
#energy

def energy(output):    
    X = output

    X1 = X[adjacency_list[0]] 
    X2 = X[adjacency_list[1]] 
        
    V_el = (k/2)*torch.sum( torch.sum((X1-X2)**2, axis = -1))
    r = X[...,np.newaxis,:] - X[...,np.newaxis,:,:]
    r2_len = torch.sum(r**2, axis = -1)
    V_nn = magnitude * torch.sum(torch.exp(-r2_len /4/(radius**2) ) ) 
    return V_el + V_nn
    
#stopping
def difference(r):
    return (max(r)-min(r))/max(r)
    
#optimizer    
    
energy_hist_lin = []
time_hist_lin = []
hist = []
output_ = []

for i in range(1):
    x = torch.eye(N)
    
    net = nn.Linear(N, dim, bias=False)
    net.apply(init_weights)


    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
    criterion = custom_loss
    
    loss_history_lin= [] 
    time_hist = []
    
    patience = 5
    r = list(np.zeros(patience))
    
    tt.tic()
    for epoch in range(500000):  
        inp = x
    
        optimizer.zero_grad()

        outputsLin = net(inp)
        
        if epoch%5 ==0:
            pairs = c_kdtree(outputsLin, 4)
        
        Dist = Distances_kdtree(outputsLin, pairs)
        

        loss = criterion(outputsLin, Dist)

    
        loss.backward(retain_graph=True)
        optimizer.step()
        
        loss_history_lin.append(loss.item())
        
    
        r.append(loss.item())
        r.pop(0)
        
        time_hist.append(tt.toc())
        
        
        if (difference(r)) < 1e-8*np.sqrt(N):
                print(difference(r))
                time_hist_lin += [tt.toc()]
                
                break
              
        
        
        time_hist_lin += [tt.toc()]
        
        
    print('Finished training: ', i)
    
    hist += [loss_history_lin]
    energy_hist_lin += [energy(outputsLin).detach().numpy()]
    
    
    
d = pd.DataFrame(energy_hist_lin)
d.to_csv('./internet/internet_energy_fdl.csv', header=True,index=False)

d = pd.DataFrame(time_hist_lin)
d.to_csv('./internet/internet_time_fdl.csv', header=True,index=False)

d = pd.DataFrame(hist)
d.to_csv('./internet/internet_loss_fdl.csv', header=True,index=False)

d = pd.DataFrame(outputsLin.detach().numpy())
d.to_csv('./internet/internet_output_fdl.csv', header=True,index=False)


