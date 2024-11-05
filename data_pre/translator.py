import json
import os.path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def trans_en_zh(dir_root,inpu_file,out_file):
    # 初始化翻译模型和 tokenizer
    translation_tokenizer = AutoTokenizer.from_pretrained("../Helsinki-NLP/opus-mt-en-zh")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("../Helsinki-NLP/opus-mt-en-zh").to("cuda")
    # 读取 JSON 文件内容
    with open(os.path.join(dir_root,inpu_file), 'r',encoding='utf-8') as file:
        data = json.load(file)

    # 翻译每个图像描述，使用 tqdm 添加进度条
    for key, value in tqdm(data.items(), desc="Translating captions"):
        caption = value.get('img-to-text', "")

        if caption:
            # 翻译生成的文本为中文
            translation_inputs = translation_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(
                "cuda")
            translated_out = translation_model.generate(**translation_inputs)
            translated_caption = translation_tokenizer.decode(translated_out[0], skip_special_tokens=True)

            # 将翻译后的文本添加到 JSON 数据中
            value['img-to-text_zh'] = translated_caption

    # 将更新后的数据写回 JSON 文件
    with open(os.path.join(dir_root,out_file), 'w',encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def trans_zh_en(dir_root,inpu_file,out_file):
    # 初始化翻译模型和 tokenizer
    translation_tokenizer = AutoTokenizer.from_pretrained("../Helsinki-NLP/opus-mt-zh-en")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("../Helsinki-NLP/opus-mt-zh-en").to("cuda")
    # 读取 JSON 文件内容
    with open(os.path.join(dir_root,inpu_file), 'r',encoding='utf-8') as file:
        data = json.load(file)

    # 翻译每个图像描述，使用 tqdm 添加进度条
    for key, value in tqdm(data.items(), desc="Translating captions"):
        caption = value.get('caption', "")
        if caption:
            # 翻译生成的文本为中文
            translation_inputs = translation_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(
                "cuda")
            translated_out = translation_model.generate(**translation_inputs)
            translated_caption = translation_tokenizer.decode(translated_out[0], skip_special_tokens=True)

            # 将翻译后的文本添加到 JSON 数据中
            value['caption_en'] = translated_caption

    # 将更新后的数据写回 JSON 文件
    with open(os.path.join(dir_root,out_file), 'w',encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    dir_root = '../../data/Weibo/'
    input_file = 'dataset_items_merged.json'
    out_file = 'output_translated.json'
    trans_zh_en(dir_root, input_file, out_file)