import numpy as np
import networkx as nx
#from torch_geometric.utils.convert import to_networkx, from_networkx
import torch
from utils import *
from sentence_transformers import SentenceTransformer
sentence_encoder_model = SentenceTransformer('all-MiniLM-L6-v2')

def eds2nx(eds:EDS):
    G = nx.DiGraph()
    for node in eds.nodes:
        G.add_node(node.id, label=node.predicate)
        for edge in node.edges.keys():
            G.add_edge(node.id, node.edges[edge])
    return G

def manual_one_hot(types, t):
    return [1 if x == t else 0 for x in types]

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

    node_feature = eds_predicate_feature + eds_type_feature + eds_properties_feature

    return node_feature

