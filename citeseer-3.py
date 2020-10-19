import torch, pickle, sys, subprocess, random, copy, time, evaluate
import numpy as np
import importlib.util
from torch_geometric.data import Data, Batch
from data.handlers import subgraph_loader
from torch import nn
from layers.gnn import GraphSAGE
from external_libs.subgraph_utils import get_subgraph
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch.nn import Sequential
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

""" srun -n 1 --gres=gpu:1 --cpus-per-task=15 --partition=ml-all-gpu --output=citeseer-3.txt python -u citeseer-3.py & """
""" srun -n 1 --gres=gpu:1 --cpus-per-task=15 --nice=1000 --partition=ml01-gpu,ml02-gpu,ml04-gpu,ml05-gpu,ml06-gpu,ml07-gpu,ml08-gpu --time=12:00:00 --pty bash """

""" Arguments: """

dataset = "citeseer"
k = 3
hidden_size = 128
tol = 1e-2
patience = 5
num_cpus = 10
lr = 0.001
num_tours = 80
batch_size = 50
super_node_size = 1000
cuda = True

train = { "DAG": torch.load("data/"+dataset+"/train.pt")[k]["DAG"] , "Hyperedge": torch.load("data/"+dataset+"/train.pt")[k]["Hyperedge"] }
test =  { "DAG": torch.load("data/"+dataset+"/test.pt")[k]["DAG"] ,  "Hyperedge": torch.load("data/"+dataset+"/test.pt")[k]["Hyperedge"] }

graph = torch.load("data/"+dataset+"/data.pt")["graph"]
dataset = torch.load("data/"+dataset+"/data.pt")["dataset"]

input_size = graph.x.size(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gnn_layer = GraphSAGE( input_size, hidden_size, double_layer=False )
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, 1))
    def forward(self, data):
        subgraph_data = subgraph_loader( data, k, super_node_size, num_tours, num_cpus )
        subgraphs = [get_subgraph(data[subgraph_data.batch[i].item()], subgraph_data.subgraphs[i].squeeze()) for i in range(len(subgraph_data.subgraphs))]
        subgraphs_lst = []
        for i in range(0, len(subgraphs), 500):
            subgraphs_b =  Batch().from_data_list(subgraphs[i:i+min([500,len(subgraphs)-i])])
            subgraphs_b = self.gnn_layer(subgraphs_b.x.cuda(), subgraphs_b.edge_index.cuda(), subgraphs_b.batch.cuda()) \
            if next(self.parameters()).get_device() != -1 else self.gnn_layer(subgraphs_b.x, subgraphs_b.edge_index, subgraphs_b.batch)
            subgraphs_lst.append(subgraphs_b)
        subgraphs = torch.cat(subgraphs_lst,dim=0)
        subgraphs = self.output_layer(subgraphs)
        weights = subgraph_data.weights.cuda() if next(self.parameters()).get_device() != -1 else subgraph_data.weights
        batch = subgraph_data.batch.cuda() if next(self.parameters()).get_device() != -1 else subgraph_data.batch
        subgraphs = subgraphs*weights
        norm = global_add_pool(weights, batch)
        energy = global_add_pool(subgraphs, batch)
        return energy/norm
    def embedding(self, subgraphs):
        with torch.no_grad():
            subgraphs_lst = []
            for i in range(0, len(subgraphs), 500):
                subgraphs_b =  Batch().from_data_list(subgraphs[i:i+min([500,len(subgraphs)-i])])
                subgraphs_b = self.gnn_layer(subgraphs_b.x.cuda(), subgraphs_b.edge_index.cuda(), subgraphs_b.batch.cuda()) \
                if next(self.parameters()).get_device() != -1 else self.gnn_layer(subgraphs_b.x, subgraphs_b.edge_index, subgraphs_b.batch)
                subgraphs_lst.append(subgraphs_b)
            subgraphs = torch.cat(subgraphs_lst,dim=0)
            return subgraphs

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

""" Creating samplers to parallelize over each minibatch """
for i in range(1,batch_size+1):
    process = subprocess.Popen("cp external_libs/rgpm/tradicional external_libs/rgpm/tradicional-"+str(i), stdout=subprocess.PIPE, stderr=None, shell=True, stdin=subprocess.PIPE)
    process.wait()

model = Net()
model.apply(init_weights)

best_model = copy.deepcopy(model)
best_loss = float("inf")
cn_wait = 0
step = 0

print("===\t RANDOM MODEL RESULTS \t===")
print("dag:\t", evaluate.logit(train_x=best_model.embedding([ get_subgraph(graph, train["DAG"]["x"][i].squeeze()) for i in range(len(train["DAG"]["x"])) ]).detach().cpu(), train_y=train["DAG"]["y"], \
test_x=best_model.embedding([ get_subgraph(graph, test["DAG"]["x"][i].squeeze()) for i in range(len(test["DAG"]["x"])) ]).detach().cpu(), test_y=test["DAG"]["y"] ), \
"hyperedge:\t", evaluate.logit(train_x=best_model.embedding([ get_subgraph(graph, train["Hyperedge"]["x"][i].squeeze()) for i in range(len(train["Hyperedge"]["x"])) ]).detach().cpu(), train_y=train["Hyperedge"]["y"], \
test_x=best_model.embedding([ get_subgraph(graph, test["Hyperedge"]["x"][i].squeeze()) for i in range(len(test["Hyperedge"]["x"])) ]).detach().cpu(), test_y=test["Hyperedge"]["y"] ) )
print("===\t                \t===")

if cuda: model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.BCEWithLogitsLoss()

random.shuffle(dataset)
stime = time.time()
for i in range(0, len(dataset), batch_size):
    model.train()
    optimizer.zero_grad()
    y = torch.tensor([graph.y for graph in dataset[i:i+min([batch_size,len(dataset)-i])]]).float().unsqueeze(-1)
    if cuda: y = y.cuda()
    energy = model(dataset[i:i+min([batch_size,len(dataset)-i])])
    loss = loss_fn(-energy, y)
    loss.backward()
    optimizer.step()
    if loss.item() <= best_loss - tol:
        best_loss, cn_wait = loss.item(), 1
        best_model = copy.deepcopy(model)
    else:
        cn_wait += 1
    print("loss:\t", loss.item(), "time:\t", time.time()-stime, "step:\t", step, "dag:\t", evaluate.logit(train_x=model.embedding([ get_subgraph(graph, train["DAG"]["x"][i].squeeze()) \
    for i in range(len(train["DAG"]["x"])) ]).detach().cpu(), train_y=train["DAG"]["y"], \
      test_x=model.embedding([ get_subgraph(graph, test["DAG"]["x"][i].squeeze()) for i in range(len(test["DAG"]["x"])) ]).detach().cpu(), test_y=test["DAG"]["y"] ))
    if cn_wait == patience and best_loss < 0.2: break
    step += 1

print("===\t TRAINED MODEL RESULTS \t===")
print("dag:\t", evaluate.logit(train_x=best_model.embedding([ get_subgraph(graph, train["DAG"]["x"][i].squeeze()) for i in range(len(train["DAG"]["x"])) ]).detach().cpu(), train_y=train["DAG"]["y"], \
test_x=best_model.embedding([ get_subgraph(graph, test["DAG"]["x"][i].squeeze()) for i in range(len(test["DAG"]["x"])) ]).detach().cpu(), test_y=test["DAG"]["y"] ), \
"hyperedge:\t", evaluate.logit(train_x=best_model.embedding([ get_subgraph(graph, train["Hyperedge"]["x"][i].squeeze()) for i in range(len(train["Hyperedge"]["x"])) ]).detach().cpu(), train_y=train["Hyperedge"]["y"], \
test_x=best_model.embedding([ get_subgraph(graph, test["Hyperedge"]["x"][i].squeeze()) for i in range(len(test["Hyperedge"]["x"])) ]).detach().cpu(), test_y=test["Hyperedge"]["y"] ) )
print("===\t                \t===")
