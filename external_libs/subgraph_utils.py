import numpy as np
import sys, subprocess, torch, signal, os, time, multiset
from multiprocessing import Process
from torch_geometric.data import Data
from itertools import permutations, combinations

def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True, stdin=subprocess.PIPE)
    output = process.communicate()
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except:
        pass
    return output[0].decode("utf-8")

def get_subgraph(G, nodes):
    x = G.x[nodes,:]
    edge_idxs = []
    for e in combinations( nodes.numpy().tolist() , 2):
        if e in G.edge_dict:
            edge_idxs.append(G.edge_dict[e])
            edge_idxs.append(G.edge_dict[(e[1],e[0])])
    edge_index = G.edge_index[:,edge_idxs]
    for i in range(edge_index.size(1)):
        edge_index[0,i], edge_index[1,i] = ((nodes == edge_index[0,i]).nonzero()).item(), ((nodes == edge_index[1,i]).nonzero()).item()
    subgraph = Data(x=x.float(), edge_index=edge_index.long())
    return subgraph

def sample_with_tours(root_path, graph, k, num_tours, super_node_size, folder_suffix):
    stime = time.time()
    out = "INIT_RW_STEPS=100;\nMAX_RW_STEPS=100000000;\nMAX_NUM_TOURS="+str(num_tours)+";\nMAX_TOUR_STEPS=1;\nMAX_INIT_ATTEMPT=1000;"
    with open(root_path+"/external_libs/rgpm/config"+folder_suffix+".txt", 'w') as fout:
        fout.write(out)
    f = open(root_path+"/external_libs/rgpm/G"+folder_suffix+".lg","w")
    for i in range(graph.x.size(0)):
        f.write("v "+str(i)+" 0\n")
    for i in range(graph.edge_index.size(1)):
        if graph.edge_index[0,i].item() < graph.edge_index[1,i].item(): f.write("e "+str(graph.edge_index[0,i].item())+" "+str(graph.edge_index[1,i].item())+" 0\n")
    f.close()
    stime = time.time()
    out = subprocess_cmd("cd " + root_path + "/external_libs/rgpm/ ; stdbuf -o0 ./tradicional" +folder_suffix+  " -i G" +folder_suffix+  ".lg -o dmp -p " + str(k) + " -s " + str(super_node_size) + " -c config" +folder_suffix+  ".txt -l debug")
    if "EXIT" in out:
        print("ERROR: Tour didn't finish")
        sys.exit()
    out = out.split('\n')
    super_node_degree = None
    super_node = []
    samples = []
    samples_pattern = []
    super_node_pattern = []
    weights = []
    exact = False
    for line in out:
        if "Has all:" in line:
            if int(line[line.find("Has all: ")+len("Has all: ")]) == 1:
                exact = True
        try:
            if "SuperEmbedding:" in line: super_node_degree = float(line[line.find(" Total Degree:")+len(" Total Degree:"):line.find(" }SN")])
            if "SN[" in line:
                line_phash = line[line.find("PHASH: ")+len("PHASH: "):]
                super_node_pattern.append(int(line_phash[:line_phash.find(" InDegree:")]))
                sample = list(map(int,line[line.find("vertices: [ ")+len("vertices: [ "):line.find(" ]")].split(' ')))
                super_node.append(sorted(sample))
            if "Tour:" in line:
                line_phash = line[line.find("PHASH: ")+len("PHASH: "):]
                samples_pattern.append(int(line_phash[:line_phash.find(" InDegree:")]))
                sample = list(map(int, line[line.find("vertices: [ ")+len("vertices: [ "):line.find(" ]")].split(" ")))
                degree = float(line[line.find(" Degree: ")+len(" Degree: "):len(line)])
                samples.append(sorted(sample))
                weights.append(1/(num_tours*degree))
        except:
            weights, samples = [], []
            break
    if super_node_degree == None:
        return None, None, None, False
    weights = np.asarray(weights)
    weights = weights*super_node_degree
    samples = samples + super_node
    patterns = np.asarray(samples_pattern + super_node_pattern)
    weights = np.concatenate([weights, np.ones(len(super_node))])
    try:
        samples = np.asarray(samples).astype(int)
    except:
        return [],[], False
    return samples, weights.reshape(-1, 1), patterns.reshape(-1, 1), exact
