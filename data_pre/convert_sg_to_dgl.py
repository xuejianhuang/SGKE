import json
import dgl
from transformers import BertTokenizer, BertModel
import torch as th
from dgl import save_graphs
from tqdm import tqdm

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('../bert-base-multilingual-uncased')
model = BertModel.from_pretrained('../bert-base-multilingual-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with th.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Take the mean of the embeddings

# Function to process edges and nodes
def process_edges_and_nodes(edges):
    src_nodes, dst_nodes, rels, isolated_nodes = [], [], [], []
    for edge in edges:
        edge = edge.strip('( ,')
        parts = [part.strip() for part in edge.split(', ')]
        if len(parts) == 3:  # Ensure we have exactly three parts: src, rel, dst
            src, rel, dst = parts
            src_nodes.append(src.strip())
            dst_nodes.append(dst.strip())
            rels.append(rel.strip())
        else:  # 孤立节点
            isolated_nodes.extend(parts)

    return src_nodes, dst_nodes, rels, isolated_nodes


# Function to convert scene graph to DGL format
def convert_to_dgl_graph(sg_data, knowledge_distillation=False,knowledge={}):
    text_all_graphs = []
    img_all_graphs = []

    for key, value in tqdm(sg_data.items(), desc="Processing graphs"):
        # Process text2SG
        text_edges = value['text2SG'].split(')')[:-1]
        t_src_nodes, t_dst_nodes, t_rels, t_isolated_nodes = process_edges_and_nodes(text_edges)

        merged_list = list(set(t_src_nodes + t_dst_nodes + t_isolated_nodes))

        if knowledge_distillation:
            for n in merged_list:
                entity_uri = knowledge[key]['text']["entity_uri"].get(n)
                if entity_uri:
                    concepts = knowledge[key]['text']['concepts'].get(entity_uri,[])
                    for c in concepts:
                        t_src_nodes.append(n)
                        t_dst_nodes.append(c)
                        t_rels.append("is")
        merged_list = list(set(t_src_nodes + t_dst_nodes + t_isolated_nodes))

        # Create DGL graph
        src_nodes_id = [merged_list.index(src) for src in t_src_nodes]
        dst_nodes_id = [merged_list.index(dst) for dst in t_dst_nodes]
        edges = th.tensor(src_nodes_id, dtype=th.int32), th.tensor(dst_nodes_id, dtype=th.int32)

        t_g = dgl.graph(edges, idtype=th.int32, num_nodes=len(merged_list))
        t_g.ndata['x'] = get_bert_embeddings(merged_list)
        if t_rels:
            t_g.edata['x'] = get_bert_embeddings(t_rels)
        text_all_graphs.append(t_g)

        # Process img2SG
        img_edges = value['img2SG'].split(')')[:-1]
        i_src_nodes, i_dst_nodes, i_rels, i_isolated_nodes = process_edges_and_nodes(img_edges)

        merged_list = list(set(i_src_nodes + i_dst_nodes + i_isolated_nodes))

        if knowledge_distillation:
            for n in merged_list:
                entity_uri = knowledge[key]['img']["entity_uri"].get(n)
                if entity_uri:
                    concepts = knowledge[key]['img']['concepts'].get(entity_uri,[])
                    for c in concepts:
                        i_src_nodes.append(n)
                        i_dst_nodes.append(c)
                        i_rels.append("is")
        merged_list = list(set(i_src_nodes + i_dst_nodes + i_isolated_nodes))
        src_nodes_id = [merged_list.index(src) for src in i_src_nodes]
        dst_nodes_id = [merged_list.index(dst) for dst in i_dst_nodes]
        edges = th.tensor(src_nodes_id, dtype=th.int32), th.tensor(dst_nodes_id, dtype=th.int32)

        i_g = dgl.graph(edges, idtype=th.int32, num_nodes=len(merged_list))
        i_g.ndata['x'] = get_bert_embeddings(merged_list)
        if i_rels:
            i_g.edata['x'] = get_bert_embeddings(i_rels)
        img_all_graphs.append(i_g)
    return text_all_graphs,img_all_graphs


if __name__ == '__main__':

    base_path='../../data/Weibo'

    with open(base_path+'/graph/SG.json') as f:
        sg_data = json.load(f)

    knowledge_distillation = True
    knowledge = {}

    if knowledge_distillation:
        with open(base_path+'/knowledge_distillation.json', encoding="utf-8") as f:
            knowledge = json.load(f)
        text_graphs_save_path = base_path+"/graph/knowledge-enhanced_text_dgl_graph.bin"
        img_graphs_save_path = base_path+"/graph/knowledge-enhanced_img_dgl_graph.bin"
    else:
        text_graphs_save_path = base_path+"/graph/text_dgl_graph.bin"
        img_graphs_save_path = base_path+"/graph/img_dgl_graph.bin"

    # Convert to DGL graphs
    text_graphs, img_graphs = convert_to_dgl_graph(sg_data, knowledge_distillation, knowledge)
    print(len(text_graphs))
    print(len(img_graphs))
    save_graphs(text_graphs_save_path, text_graphs)
    save_graphs(img_graphs_save_path, img_graphs)
