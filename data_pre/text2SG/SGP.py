from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
import torch
import json


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def text2SG(input_file, caption_key, text_key, output_file):
    data = load_json(input_file)

    # Extract captions and img-to-texts
    captions = [item[caption_key] for item in data.values()]

    img_to_texts = [item[text_key] for item in data.values()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = SceneGraphParser('pre_model/flan-t5-base-VG-factual-sg', device=device)

    # Parse text and image-to-text in parallel
    text2SG = parser.parse(captions, beam_size=5, return_text=True,batch_size=16)
    img2SG = parser.parse(img_to_texts, beam_size=5, return_text=True,batch_size=16)

    # Create a dictionary to store results
    sg = {
        key: {
            'text2SG': text2SG[idx],
            'img2SG': img2SG[idx]
        }
        for idx, key in enumerate(data.keys())
    }

    # Save the parsed results to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sg, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':

    input_file='../../../data/Weibo/dataset_items_merged.json'
    caption_key="caption_en"
    text_key="img-to-text"
    output_file='../../../data/Weibo/graph/SG.json'
    text2SG(input_file,caption_key,text_key,output_file)
