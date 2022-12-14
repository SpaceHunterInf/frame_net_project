from annotation import *
from delphin.codecs.eds import loads
from delphin.eds import *
from nltk.tree import Tree
from nltk.tokenize.treebank import *
import os, re, sys

def get_parse_from_file(filename, idx):
    with open(filename, 'r', encoding='UTF-8') as f:
        parses = f.read().split('\n\n')
    return parses[idx]

def tokenize(semlink:SemLinkAnnotation):
    tree_raw = get_parse_from_file(semlink.file_path, int(semlink.sentence_no))
    tree = Tree.fromstring(tree_raw)
    return tree.leaves()

def detokenize(tokens):
    tokens = [x for x in tokens if not('*' in x)]
    dt = TreebankWordDetokenizer()
    return dt.detokenize(tokens)

def get_verb_nodes(eds:EDS):
    return [x for x in eds.nodes if '_v_' in x.predicate]

def get_node(eds:EDS, node_id):
    for x in eds.nodes:
        if x.id == node_id:
            return x
    raise Warning('No node has id {}'.format(node_id))

def get_file_name(semi:SemLinkAnnotation):
    file_index = semi.source_file.split('.')[0].split('_')[-1] #get file number
    sentence_index = str(int(semi.sentence_no) + 1)
    sentence_index = '0'*(3-len(sentence_index)) + sentence_index

    return '2' + file_index + sentence_index

def get_from_file(file):
    with open(file, 'r', encoding='UTF-8') as f:
        data = f.read()
    
    blocks = data.split('\n\n')
    eds_blocks = blocks[7] #manual index
    eds = loads(eds_blocks)[0]

    sentence_blocks = blocks[1]
    
    sentence_start = -1
    sentence_end = -1
    for idx in range(len(sentence_blocks)):
        if sentence_blocks[idx] == '`':
            sentence_start = idx
            break
    
    for idx in range(len(sentence_blocks)):
        if sentence_blocks[idx] == "'":
            sentence_end = idx

    #print(sentence_start, sentence_end)
    return sentence_blocks[sentence_start+1:sentence_end], eds_blocks, eds

def get_children_strings(sentence, eds, node):
    arg_str_dict = {}
    for key in node.edges.keys():
        arg_str_dict[key] = ""

    for key in node.edges.keys():
        batch = [get_node(eds, node.edges[key])]
        index_tuples = [get_node(eds, node.edges[key]).lnk.data]
        children_string = ""

        while batch != []:
            new_batch = []
            for c in batch:
                # print(c.predicate)
                # print(c.edges)
                for c_key in c.edges.keys():
                    # print(c_key)
                    # print(c.edges[key])
                    new_batch.append(get_node(eds, c.edges[c_key]))
                    #collect all children nodes
            #print(new_batch)
            for c in new_batch:
                new_node = c
                index_tuples.append(new_node.lnk.data)
            batch = new_batch
    
        index_tuples = sorted([(int(i), int(j)) for (i,j) in index_tuples], key=lambda x : x[0])
        for (i,j) in index_tuples:
            children_string += sentence[i:j]

        arg_str_dict[key] = children_string
    
    return arg_str_dict

class sentence_eds():
    def __init__(self, filename):
        self.sentence, self.eds_text, self.eds = get_from_file(filename)
        self.predicate_reg = r':.*?<'
        self.edge_reg = r'\[.*?\]'
        self.filename = filename

    def update_text(self, id, new_predicate, new_edges):
        self.eds_lines = self.eds_text.split('\n ')
        for line_idx in range(len(self.eds_lines)):
            line = self.eds_lines[line_idx]
            if line.split(':')[0] == id and line != self.eds_lines[0]:
                update_predicate = re.sub(self.predicate_reg, ':' + new_predicate + '<', line)
                
                edges_strings = []
                for key in new_edges.keys():
                    edges_strings.append(key + ' ' + new_edges[key])
                update_edges = re.sub(self.edge_reg, '[' + ', '.join(edges_strings) + ']', update_predicate)

                self.eds_lines[line_idx] = update_edges

                self.eds_text = '\n '.join(self.eds_lines)
                self.eds = loads(self.eds_text)[0]
                break
    
    def save_to_path(self, path=''):
        with open(os.path.join(path, self.filename + '_projected'), 'w') as f:
            f.write(self.eds_text)

