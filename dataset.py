from torch.utils.data import Dataset,DataLoader
from dgl import load_graphs
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from collate_fn import *
from utils.util import *
from models import *
import config


class SGKE_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir,knowledge_enhanced=True):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        self.ELA_transform = get_transform_Compose_ELA()
        if knowledge_enhanced:
            self.img_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_img_dgl_graph.bin'))[0]
            self.text_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_text_dgl_graph.bin'))[0]
        else:
            self.img_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/text_dgl_graph.bin'))[0]

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))

        ELA_image_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        ELA_img=self.ELA_transform(load_img_pil(ELA_image_path))

        direct_path_item = os.path.join(self.data_root_dir, item['direct_path']) #视觉证据
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])  #文本证据
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        t_evidence = load_captions(inv_ann_dict)
        t_evidence += load_captions_weibo(direct_dict)
        t_evidence = t_evidence[:config.max_captions_num]
        i_evidence = load_imgs_direct_search(self.Img_transform, direct_path_item, direct_dict,config.max_images_num)
        qImg, qCap = self.load_data(key)
        sample = {'label': label, 't_evidence': t_evidence, 'i_evidence': i_evidence, 'qImg': qImg, 'qCap': qCap,'ELA_img':ELA_img,'img_dgl_graph':self.img_dgl_graph[int(key)],'text_dgl_graph':self.text_dgl_graph[int(key)]}

        return sample, len(t_evidence), i_evidence.shape[0], key

    def __len__(self):
        return len(self.context_data_items_dict)

class SGKE_ELA_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir,knowledge_enhanced=True):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        if knowledge_enhanced:
            self.img_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_img_dgl_graph.bin'))[0]
            self.text_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_text_dgl_graph.bin'))[0]
        else:
            self.img_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/text_dgl_graph.bin'))[0]

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))


        direct_path_item = os.path.join(self.data_root_dir, item['direct_path']) #视觉证据
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])  #文本证据
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        t_evidence = load_captions(inv_ann_dict)
        t_evidence += load_captions_weibo(direct_dict)
        t_evidence = t_evidence[:config.max_captions_num]
        i_evidence = load_imgs_direct_search(self.Img_transform, direct_path_item, direct_dict,config.max_images_num)
        qImg, qCap = self.load_data(key)
        sample = {'label': label, 't_evidence': t_evidence, 'i_evidence': i_evidence, 'qImg': qImg, 'qCap': qCap,'img_dgl_graph':self.img_dgl_graph[int(key)],'text_dgl_graph':self.text_dgl_graph[int(key)]}

        return sample, len(t_evidence), i_evidence.shape[0], key

    def __len__(self):
        return len(self.context_data_items_dict)

class SGKEv_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir,knowledge_enhanced=True):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        self.ELA_transform = get_transform_Compose_ELA()
        if knowledge_enhanced:
            self.img_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_img_dgl_graph.bin'))[0]
            self.text_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_text_dgl_graph.bin'))[0]
        else:
            self.img_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/text_dgl_graph.bin'))[0]

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))

        ELA_image_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        ELA_img=self.ELA_transform(load_img_pil(ELA_image_path))

        direct_path_item = os.path.join(self.data_root_dir, item['direct_path']) #视觉证据
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        i_evidence = load_imgs_direct_search(self.Img_transform, direct_path_item, direct_dict,config.max_images_num)
        qImg, qCap = self.load_data(key)
        sample = {'label': label, 'i_evidence': i_evidence, 'qImg': qImg, 'qCap': qCap,'ELA_img':ELA_img,'img_dgl_graph':self.img_dgl_graph[int(key)],'text_dgl_graph':self.text_dgl_graph[int(key)]}

        return sample, i_evidence.shape[0], key

    def __len__(self):
        return len(self.context_data_items_dict)

class SGKEt_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir,knowledge_enhanced=True):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        self.ELA_transform = get_transform_Compose_ELA()
        if knowledge_enhanced:
            self.img_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_img_dgl_graph.bin'))[0]
            self.text_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_text_dgl_graph.bin'))[0]
        else:
            self.img_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/text_dgl_graph.bin'))[0]

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))

        ELA_image_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        ELA_img=self.ELA_transform(load_img_pil(ELA_image_path))

        direct_path_item = os.path.join(self.data_root_dir, item['direct_path']) #视觉证据
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])  #文本证据
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        t_evidence = load_captions(inv_ann_dict)
        t_evidence += load_captions_weibo(direct_dict)
        t_evidence = t_evidence[:config.max_captions_num]

        qImg, qCap = self.load_data(key)
        sample = {'label': label, 't_evidence': t_evidence, 'qImg': qImg, 'qCap': qCap,'ELA_img':ELA_img,'img_dgl_graph':self.img_dgl_graph[int(key)],'text_dgl_graph':self.text_dgl_graph[int(key)]}

        return sample, len(t_evidence), key

    def __len__(self):
        return len(self.context_data_items_dict)

class SGK_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir,knowledge_enhanced=True):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        self.ELA_transform = get_transform_Compose_ELA()
        if knowledge_enhanced:
            self.img_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_img_dgl_graph.bin'))[0]
            self.text_dgl_graph=load_graphs(os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_text_dgl_graph.bin'))[0]
        else:
            self.img_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/text_dgl_graph.bin'))[0]

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))

        ELA_image_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        ELA_img=self.ELA_transform(load_img_pil(ELA_image_path))


        qImg, qCap = self.load_data(key)
        sample = {'label': label,  'qImg': qImg, 'qCap': qCap,'ELA_img':ELA_img,'img_dgl_graph':self.img_dgl_graph[int(key)],'text_dgl_graph':self.text_dgl_graph[int(key)]}

        return sample, key

    def __len__(self):
        return len(self.context_data_items_dict)

class SGKE_SG_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        self.ELA_transform = get_transform_Compose_ELA()

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))

        ELA_image_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        ELA_img=self.ELA_transform(load_img_pil(ELA_image_path))

        direct_path_item = os.path.join(self.data_root_dir, item['direct_path']) #视觉证据
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])  #文本证据
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        t_evidence = load_captions(inv_ann_dict)
        t_evidence += load_captions_weibo(direct_dict)
        t_evidence = t_evidence[:config.max_captions_num]
        i_evidence = load_imgs_direct_search(self.Img_transform, direct_path_item, direct_dict,config.max_images_num)
        qImg, qCap = self.load_data(key)
        sample = {'label': label, 't_evidence': t_evidence, 'i_evidence': i_evidence, 'qImg': qImg, 'qCap': qCap,'ELA_img':ELA_img}

        return sample, len(t_evidence), i_evidence.shape[0], key

    def __len__(self):
        return len(self.context_data_items_dict)

class SGKE_SG_E_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.Img_transform = get_transform_Compose()
        self.ELA_transform = get_transform_Compose_ELA()

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.Img_transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))

        ELA_image_path = os.path.join(self.data_root_dir, get_ela_image_path(item['image_path']))
        ELA_img=self.ELA_transform(load_img_pil(ELA_image_path))

        qImg, qCap = self.load_data(key)
        sample = {'label': label, 'qImg': qImg, 'qCap': qCap,'ELA_img':ELA_img}

        return sample, key

    def __len__(self):
        return len(self.context_data_items_dict)

class TI_Dataset(Dataset):
    def __init__(self, data_items,data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        qCap = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)

        return label, qCap, transform_img,key

    def __len__(self):
        return len(self.context_data_items_dict)

class SG_Dataset(DGLDataset):
    def __init__(self, data_items,data_root_dir,knowledge_enhanced=True):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        if knowledge_enhanced:
            self.img_dgl_graph = load_graphs(
                os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(
                os.path.join(self.data_root_dir, 'graph/knowledge-enhanced_text_dgl_graph.bin'))[0]
        else:
            self.img_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/img_dgl_graph.bin'))[0]
            self.text_dgl_graph = load_graphs(os.path.join(self.data_root_dir, 'graph/text_dgl_graph.bin'))[0]

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        img_dgl=self.img_dgl_graph[int(key)]
        text_dgl=self.text_dgl_graph[int(key)]
        return label.to(config.device), img_dgl.to(config.device),text_dgl.to(config.device),key

    def __len__(self):
        return len(self.context_data_items_dict)

class T_Dataset(Dataset):
    def __init__(self, data_items):
        self.context_data_items_dict = data_items
        self.idx_to_keys = list(self.context_data_items_dict.keys())

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        qCap = self.context_data_items_dict[key]['caption']
        return label, qCap, key

    def __len__(self):
        return len(self.context_data_items_dict)

class Img_Dataset(Dataset):
    def __init__(self, data_items,data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return label.to(config.device), transform_img.to(config.device),key

    def __len__(self):
        return len(self.context_data_items_dict)

class ELA_Dataset(Dataset):
    def __init__(self, data_items,data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose_ELA()

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        ELA_image_path =os.path.join(self.data_root_dir,  get_ela_image_path(item['image_path']))
        pil_img = load_img_pil(ELA_image_path)
        transform_img = self.transform(pil_img)
        return label.to(config.device), transform_img.to(config.device),key

    def __len__(self):
        return len(self.context_data_items_dict)


def getModelAndData(model_name='SGKE', dataset='weibo',batch=config.batch_size):

    if dataset =='weibo':
        data_root_dir = config.weibo_dataset_dir
    elif dataset=='twitter':
        data_root_dir=config.twitter_dataset_dir

    elif dataset=='pheme':
        data_root_dir=config.pheme_dataset_dir
    elif dataset=='weibo2':
        data_root_dir=config.weibo2_dataset_dir

    dataset_path=os.path.join(data_root_dir, 'dataset_items_merged.json')

    train_items, val_items, test_items = split_dataset(dataset_path,config.train_ratio, config.val_ratio)

    # SGKE/SGE/SGKE_ELA/SGKEv/SGKEt/SGK/SGKE_SG/SGKE_SG_E/TI/SG/Text/Image/ELA/GKE
    if model_name in ['SGKE', 'SGE']:
        if model_name == 'SGKE':
            train_dataset = SGKE_Dataset(train_items, data_root_dir,knowledge_enhanced=True)
            val_dataset = SGKE_Dataset(val_items, data_root_dir,knowledge_enhanced=True)
            test_dataset = SGKE_Dataset(test_items, data_root_dir,knowledge_enhanced=True)
        else:
            train_dataset = SGKE_Dataset(train_items, data_root_dir, knowledge_enhanced=False)
            val_dataset = SGKE_Dataset(val_items, data_root_dir, knowledge_enhanced=False)
            test_dataset = SGKE_Dataset(test_items, data_root_dir, knowledge_enhanced=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGKE)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGKE)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGKE)
        model = SGKE(config.node_feats, config.edge_feats, config.out_feats, config.num_heads, config.n_layers).to(config.device)

    elif model_name=='SGKE_ELA':
        train_dataset = SGKE_ELA_Dataset(train_items, data_root_dir)
        val_dataset = SGKE_ELA_Dataset(val_items, data_root_dir)
        test_dataset=SGKE_ELA_Dataset(test_items, data_root_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_SGKE_ELA)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, collate_fn=collate_SGKE_ELA)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=collate_SGKE_ELA)
        model = SGKE_ELA(config.node_feats, config.edge_feats, config.out_feats, config.num_heads, config.n_layers).to(config.device)

    elif model_name == 'SGKEv':
        train_dataset = SGKEv_Dataset(train_items, data_root_dir)
        val_dataset = SGKEv_Dataset(val_items, data_root_dir)
        test_dataset = SGKEv_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGKEv)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGKEv)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGKEv)
        model = SGKEv(config.node_feats, config.edge_feats, config.out_feats, config.num_heads, config.n_layers).to(config.device)

    elif model_name == 'SGKEt':
        train_dataset = SGKEt_Dataset(train_items, data_root_dir)
        val_dataset = SGKEt_Dataset(val_items, data_root_dir)
        test_dataset = SGKEt_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGKEt)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGKEt)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGKEt)
        model = SGKEt(config.node_feats, config.edge_feats, config.out_feats, config.num_heads, config.n_layers).to(config.device)

    elif model_name=='SGK':
        train_dataset = SGK_Dataset(train_items, data_root_dir)
        val_dataset = SGK_Dataset(val_items, data_root_dir)
        test_dataset = SGK_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGK)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGK)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGK)
        model=SGK(config.node_feats, config.edge_feats, config.out_feats, config.num_heads, config.n_layers).to(config.device)

    elif model_name ==  'SGKE_SG':
        train_dataset = SGKE_SG_Dataset(train_items, data_root_dir)
        val_dataset = SGKE_SG_Dataset(val_items, data_root_dir)
        test_dataset = SGKE_SG_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGKE_SG)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGKE_SG)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGKE_SG)
        model = SGKE_SG().to(config.device)

    elif model_name == 'SGKE_SG_E':
        train_dataset = SGKE_SG_E_Dataset(train_items, data_root_dir)
        val_dataset = SGKE_SG_E_Dataset(val_items, data_root_dir)
        test_dataset = SGKE_SG_E_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGKE_SG_E)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGKE_SG_E)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGKE_SG_E)
        model = SGKE_SG_E().to(config.device)

    elif model_name=='TI':
        train_dataset = TI_Dataset(train_items, data_root_dir)
        val_dataset = TI_Dataset(val_items, data_root_dir)
        test_dataset=TI_Dataset(test_items, data_root_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_TI)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, collate_fn=collate_TI)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=collate_TI)
        model = TI().to(config.device)

    elif model_name=='SG':
        train_dataset = SG_Dataset(train_items, data_root_dir, knowledge_enhanced=config.knowledge_enhanced)
        val_dataset = SG_Dataset(val_items, data_root_dir, knowledge_enhanced=config.knowledge_enhanced)
        test_dataset = SG_Dataset(test_items, data_root_dir, knowledge_enhanced=config.knowledge_enhanced)
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_dataloader = GraphDataLoader(val_dataset, batch_size=batch, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch, shuffle=False)
        model = SG(config.node_feats, config.edge_feats, config.out_feats, config.num_heads, config.n_layers).to(config.device)

    elif model_name=='Text':
        train_dataset = T_Dataset(train_items)
        val_dataset = T_Dataset(val_items)
        test_dataset=T_Dataset(test_items)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,collate_fn=collate_text)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,collate_fn=collate_text)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,collate_fn=collate_text)
        model=Text().to(config.device)

    elif model_name=='Image':
        train_dataset = Img_Dataset(train_items, data_root_dir)
        val_dataset = Img_Dataset(val_items, data_root_dir)
        test_dataset=Img_Dataset(test_items, data_root_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
        model = Image().to(config.device)

    elif model_name=='ELA':
        train_dataset = ELA_Dataset(train_items, data_root_dir)
        val_dataset = ELA_Dataset(val_items, data_root_dir)
        test_dataset=ELA_Dataset(test_items, data_root_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
        model = ELA().to(config.device)
    elif model_name=='GKE':
        train_dataset = SGKE_Dataset(train_items, data_root_dir, knowledge_enhanced=True)
        val_dataset = SGKE_Dataset(val_items, data_root_dir, knowledge_enhanced=True)
        test_dataset = SGKE_Dataset(test_items, data_root_dir, knowledge_enhanced=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                      collate_fn=collate_SGKE)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                    collate_fn=collate_SGKE)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                     collate_fn=collate_SGKE)
        model = GKE(config.node_feats, config.out_feats, config.num_heads, config.n_layers).to(
            config.device)
    else:
        print("The model does not exist")
        return None

    return model, train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    dataset_path = os.path.join(config.twitter_dataset_dir, 'dataset_items_merged.json')
    train,val,test=split_dataset(dataset_path)
    print(len(train),len(val),len(test))
