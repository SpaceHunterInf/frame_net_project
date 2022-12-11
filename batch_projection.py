from nltk.tokenize.treebank import *
from nltk.metrics.distance import edit_distance
from annotation import *
from copy import deepcopy
from tqdm import tqdm
from utils import *
import os, re, sys
import joblib
import logging
logging.basicConfig(filename='projection_log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")

logger = logging.getLogger('urbanGUI')

#PTB tokenizer and detonkenizer
#tokenizer = TreebankWordTokenizer()


def update_graph_old(sentence, eds:EDS, semlink:SemLinkAnnotation):
    tokens = tokenize(semlink)
    token_idx = int(semlink.token_no)
    target_verb = tokens[token_idx]
    begin_idx = len(detokenize(tokens[:token_idx]))
    end_idx = len(detokenize(tokens[:token_idx+1]))
    candidate_verb_nodes = get_verb_nodes(eds)
    
    source_verb_node = None
    #update predicate structure
    for verb_node in candidate_verb_nodes:
        s,e = verb_node.lnk.data
        # print(s,e)
        idx_pairs = sorted([(s,e), (begin_idx, end_idx)], key= lambda x : x[0])

        if idx_pairs[0][1] > idx_pairs[1][0]:
            #overlap confirmed
            source_verb_node = verb_node
    
    # print(begin_idx, end_idx)
    # print(source_verb_node)
    if source_verb_node == None:
        print(target_verb)
        print(sentence)
        raise Exception('empty verb???')
    updated_predicate = source_verb_node.predicate + '-fn.' + semlink.fn_frame

    #update edge roles
    target_dependencies = [x for x in semlink.dependencies if ';' in x]
    candidate_edges = deepcopy(source_verb_node.edges)

    source_verb_edges_dict = deepcopy(source_verb_node.edges)
    for dependency in target_dependencies:
        token_span = dependency.split('-')[0]
        start_token_idx = int(token_span.split(':')[0])
        end_token_idx = start_token_idx + int(token_span.split(':')[1]) + 1
        begin_idx = len(detokenize(tokens[:start_token_idx]))
        end_idx = len(detokenize(tokens[:end_token_idx]))
        fn_role = dependency.split(';')[-1]

        target_child_node = None

        for key in candidate_edges.keys():
            tmp_child_node = get_node(eds, candidate_edges[key])
            s,e = tmp_child_node.lnk.data
            idx_pairs = sorted([(s,e), (begin_idx, end_idx)], key= lambda x : x[0])

            if idx_pairs[0][1] > idx_pairs[1][0]:
                #overlap confirmed
                target_child_node = tmp_child_node
                new_key = key + '-fn.' + fn_role
                
                # print(dependency)
                # print(s,e)
                # print(begin_idx, end_idx)
                # print(new_key)\
                    
                #update edge_dict key
                source_verb_edges_dict[new_key] = source_verb_edges_dict.pop(key)
                # old_edge = (source_verb_node.id, key, tmp_child_node.id)
                # new_edge = (source_verb_node.id, new_key, tmp_child_node.id)
                # eds.edges.remove(old_edge)
                # eds.edges.append(new_edge)
                #TODO update edges

                break
    
    return source_verb_node.id, updated_predicate, source_verb_edges_dict

def update_graph(sentence, eds:EDS, semlink:SemLinkAnnotation):
    tokens = tokenize(semlink)
    token_idx = int(semlink.token_no)
    target_verb = tokens[token_idx]
    begin_idx = len(detokenize(tokens[:token_idx]))
    end_idx = len(detokenize(tokens[:token_idx+1]))
    candidate_verb_nodes = get_verb_nodes(eds)
    
    source_verb_node = None
    #update predicate structure
    for verb_node in candidate_verb_nodes:
        s,e = verb_node.lnk.data
        # print(s,e)
        idx_pairs = sorted([(s,e), (begin_idx, end_idx)], key= lambda x : x[0])

        if idx_pairs[0][1] > idx_pairs[1][0]:
            #overlap confirmed
            source_verb_node = verb_node
    
    # print(begin_idx, end_idx)
    # print(source_verb_node)
    if source_verb_node == None:
        print(target_verb)
        print(sentence)
        raise Exception('empty verb???')
    updated_predicate = source_verb_node.predicate + '-fn.' + semlink.fn_frame

    #update edge roles
    target_dependencies = [x for x in semlink.dependencies if ';' in x]
    arg_assosicated_strings = get_children_strings(sentence, eds, source_verb_node)

    source_verb_edges_dict = deepcopy(source_verb_node.edges)
    for dependency in target_dependencies:
        token_intervals = dependency.split('-')[0].replace(';','*').split('*')
        concate_string = ""
        for interval in token_intervals:
            start_token_idx = int(interval.split(':')[0])
            end_token_idx = start_token_idx + int(interval.split(':')[1]) + 1
            concate_string += detokenize(tokens[start_token_idx : end_token_idx])
        
        fn_role = dependency.split(';')[-1]

        target_child_node = None

        max_len = 0
        for key in arg_assosicated_strings:
            if len(arg_assosicated_strings[key]) > max_len:
                max_len = len(arg_assosicated_strings[key])
        
        for key in arg_assosicated_strings:
            arg_assosicated_strings[key] += '='*(max_len - len(arg_assosicated_strings[key]))

        edit_distances = sorted([(key, edit_distance(concate_string, arg_assosicated_strings[key])) for key in arg_assosicated_strings.keys()], key= lambda x : x[1])
        try:
            old_key = edit_distances[0][0]

            new_key = old_key + '-fn.' + fn_role
            arg_assosicated_strings.pop(old_key)
            source_verb_edges_dict[new_key] = source_verb_edges_dict.pop(old_key)
            print(new_key)
        except:
            #exit() annotation mismatch
            logging.warning('annotation mismatch', edit_distance, source_verb_node, source_verb_node.edges, sentence, semlink.file_path)

        
    
    return source_verb_node.id, updated_predicate, source_verb_edges_dict

if __name__ == "__main__":
    # semlinks_data = [line.rstrip() for line in open('1.2.2c.okay.txt')]
    # deepbank_files = os.listdir('deepbank_raw')
    # semlinks_dict = {}
    # counter = 0
    # for d in tqdm(semlinks_data, desc='Filtering Redundant Semlink Annotations'):
    #     semlink = SemLinkAnnotation(d)
    #     filename = get_file_name(semlink)
    #     if filename in deepbank_files and os.path.exists(semlink.file_path):
    #         if not(filename in semlinks_dict.keys()):
    #             semlinks_dict[filename] = []
    #         semlinks_dict[filename].append(semlink)
    #         counter += 1
    
    # print(counter)
    # with open('semlink_dict.pkl', 'wb') as f:
    #     joblib.dump(semlinks_dict, f)

    with open('semlink_dict.pkl', 'rb') as f:
        semlinks_dict = joblib.load(f)

    for f in tqdm(semlinks_dict.keys(), desc='Processing Files'):
        current_deepbank = sentence_eds(os.path.join('deepbank_raw',f))
        for semlink in semlinks_dict[f]:
            new_id, new_pred, new_edges = update_graph(current_deepbank.sentence, current_deepbank.eds, semlink)
            current_deepbank.update_text(new_id, new_pred, new_edges)
        
        with open(os.path.join('deepbank_projected', f+'_projected'), 'w') as f:
            f.write(current_deepbank.eds_text)
            f.close()
