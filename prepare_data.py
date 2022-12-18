import numpy as np
# import networkx as nx
# from torch_geometric.utils.convert import to_networkx, from_networkx
import json, joblib
from tqdm import tqdm
import torch
from utils import *
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset, Dataset
# sentence_encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
# sentence_encoder_model.cuda(0)

def eds_encode(sentence, eds, feature_dict):
    nodes2idx = {}
    counter = 0

    nodes = []
    for n in eds.nodes:
        nodes2idx[n.id] = counter
        counter += 1
        nodes.append(node_feature_encode(sentence, n, feature_dict))

    edges = []
    for n in eds.nodes:
        for key in n.edges:
            edge = [nodes2idx[n.id], nodes2idx[n.edges[key]]]
            edges.append(edge)
    
    return torch.stack(nodes).squeeze(), torch.tensor(edges), nodes2idx

def node_feature_encode(sentence, eds_node, features):
    text = sentence[eds_node.lnk.data[0]: eds_node.lnk.data[1]]
    eds_predicate_feature = [1 if x in eds_node.predicate else 0 for x in features['predicate']]
    eds_type_feature = [1 if x == eds_node.type else 0 for x in features['type']]
    eds_properties_feature = []
    for key in features['property'].keys():
        if key in eds_node.properties.keys():
            key_feature = [1 if val == eds_node.properties[key] else 0 for val in features['property'][key].keys()]
        else:
            key_feature = [0 for val in features['property'][key].keys()]

        eds_properties_feature += key_feature

    if not(1 in eds_predicate_feature):
        eds_predicate_feature.append(1)
    else:
        eds_predicate_feature.append(0)
    # to handle unseen inputs

    node_feature = torch.tensor(eds_predicate_feature + eds_type_feature + eds_properties_feature)
    #node_feature = torch.cat((torch.from_numpy(sentence_encoder_model.encode(text)), node_feature), 0)
    return node_feature

class edsDataset():    
    def __init__(self, data):          
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    fn_frame2id = {}
    fn_role2id = {}

    with open('fn_frames.json') as f:
        fn_frames = json.load(f)
    counter = 0
    for k in fn_frames.keys():
        fn_frame2id[k.lower()] = counter
        counter += 1

    with open('fn_roles.json') as f:
        fn_roles = json.load(f)
    counter = 0
    for k in fn_roles.keys():
        fn_role2id[k.lower()] = counter
        counter += 1
    fn_role2id['****'] = counter #for no assigned role

    with open('fn_frame2id.json','w') as f:
        f.write(json.dumps(fn_frame2id, indent=2))
        f.close()

    with open('fn_role2id.json','w') as f:
        f.write(json.dumps(fn_role2id, indent=2))
        f.close()

    node_classification_data = []
    edge_classification_data = []

    with open('filtered_data.json', 'r') as f:
        data = json.load(f)

    with open('features_config.json', 'r') as f:
        feature_dict = json.load(f)

    for k in tqdm(data.keys(), desc='data'):
        s, _, _ = get_from_file('deepbank_raw/' + k)
        current_eds = loads(data[k])[0]
        for node in current_eds.nodes:
            if '-fn.' in node.predicate: 
                target_fn_frame = node.predicate.split('-fn.')[-1].lower()
                target_fn_frame_id = fn_frame2id[target_fn_frame]
                if target_fn_frame != 'in' and target_fn_frame != 'nf':
                    verb_data = {'sentence':s, 'eds':current_eds, 'verb_id':node.id, 'target_fn_frame_id':target_fn_frame_id}
                    node_classification_data.append(verb_data)

                    for key in node.edges:
                        if not '-FN.' in key:
                            target_fn_role = '****'
                        else:
                            target_fn_role = key.split('-FN.')[-1].lower()
                        target_fn_role_id = fn_role2id[target_fn_role]
                        edge_data = {'sentence':s, 'eds':current_eds, 'start':node.id, 'end':node.edges[key], 'target_fn_role_id':target_fn_role_id}
                        edge_classification_data.append(edge_data)

    with open('verb_data_balanced.pkl','wb') as f:
        joblib.dump(node_classification_data, f)

    with open('edge_data_balanced.pkl','wb') as f:
        joblib.dump(edge_classification_data, f)

    exit()