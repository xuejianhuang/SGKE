import torch
import config
import dgl

def collate_SGKE(batch):
    samples = [item[0] for item in batch]
    max_t_evidence_len = max([item[1] for item in batch])
    max_i_evidence_len = max([item[2] for item in batch])
    samples_key=[item[3] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    ELA_img_batch=[]
    i_evidence_batch = []
    t_evidence_batch = []
    labels = []
    img_dgl_graph=[]
    text_dgl_graph=[]
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        t_evidence = sample['t_evidence']
        cap_len = len(t_evidence)
        for i in range(max_t_evidence_len - cap_len):
            t_evidence.append("")
        t_evidence_tokenizer = config._tokenizer(t_evidence, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                                 truncation=True).to(config.device)
        t_evidence_batch.append(t_evidence_tokenizer)

        if len(sample['i_evidence'].shape) > 2:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], sample['i_evidence'].shape[1], sample['i_evidence'].shape[2], sample['i_evidence'].shape[3])
        else:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['i_evidence'], torch.zeros(padding_size)), dim=0)
        i_evidence_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])
        img_dgl_graph.append(sample['img_dgl_graph'])
        text_dgl_graph.append(sample['text_dgl_graph'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    i_evidence_batch = torch.stack(i_evidence_batch, dim=0).to(config.device)
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    ELA_img_batch=torch.stack(ELA_img_batch,dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, t_evidence_batch, i_evidence_batch, qCap_batch, qImg_batch, ELA_img_batch, dgl.batch(img_dgl_graph).to(config.device), dgl.batch(text_dgl_graph).to(config.device), samples_key

def collate_SGKE_ELA(batch):
    samples = [item[0] for item in batch]
    max_t_evidence_len = max([item[1] for item in batch])
    max_i_evidence_len = max([item[2] for item in batch])
    samples_key=[item[3] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    i_evidence_batch = []
    t_evidence_batch = []
    labels = []
    img_dgl_graph=[]
    text_dgl_graph=[]
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        t_evidence = sample['t_evidence']
        cap_len = len(t_evidence)
        for i in range(max_t_evidence_len - cap_len):
            t_evidence.append("")
        t_evidence_tokenizer = config._tokenizer(t_evidence, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                                 truncation=True).to(config.device)
        t_evidence_batch.append(t_evidence_tokenizer)

        if len(sample['i_evidence'].shape) > 2:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], sample['i_evidence'].shape[1], sample['i_evidence'].shape[2], sample['i_evidence'].shape[3])
        else:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['i_evidence'], torch.zeros(padding_size)), dim=0)
        i_evidence_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        img_dgl_graph.append(sample['img_dgl_graph'])
        text_dgl_graph.append(sample['text_dgl_graph'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    i_evidence_batch = torch.stack(i_evidence_batch, dim=0).to(config.device)
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, t_evidence_batch, i_evidence_batch, qCap_batch, qImg_batch, dgl.batch(img_dgl_graph).to(config.device), dgl.batch(text_dgl_graph).to(config.device), samples_key

def collate_SGKEv(batch):
    samples = [item[0] for item in batch]
    max_i_evidence_len = max([item[1] for item in batch])
    samples_key=[item[2] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    ELA_img_batch=[]
    i_evidence_batch = []
    labels = []
    img_dgl_graph=[]
    text_dgl_graph=[]
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])

        if len(sample['i_evidence'].shape) > 2:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], sample['i_evidence'].shape[1], sample['i_evidence'].shape[2], sample['i_evidence'].shape[3])
        else:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['i_evidence'], torch.zeros(padding_size)), dim=0)
        i_evidence_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])
        img_dgl_graph.append(sample['img_dgl_graph'])
        text_dgl_graph.append(sample['text_dgl_graph'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    i_evidence_batch = torch.stack(i_evidence_batch, dim=0).to(config.device)
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    ELA_img_batch=torch.stack(ELA_img_batch,dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, i_evidence_batch, qCap_batch, qImg_batch, ELA_img_batch, dgl.batch(img_dgl_graph).to(config.device), dgl.batch(text_dgl_graph).to(config.device), samples_key

def collate_SGKEt(batch):
    samples = [item[0] for item in batch]
    max_t_evidence_len = max([item[1] for item in batch])
    samples_key=[item[2] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    ELA_img_batch=[]
    t_evidence_batch = []
    labels = []
    img_dgl_graph=[]
    text_dgl_graph=[]
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        t_evidence = sample['t_evidence']
        cap_len = len(t_evidence)
        for i in range(max_t_evidence_len - cap_len):
            t_evidence.append("")
        t_evidence_tokenizer = config._tokenizer(t_evidence, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                                 truncation=True).to(config.device)
        t_evidence_batch.append(t_evidence_tokenizer)


        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])
        img_dgl_graph.append(sample['img_dgl_graph'])
        text_dgl_graph.append(sample['text_dgl_graph'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    ELA_img_batch=torch.stack(ELA_img_batch,dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, t_evidence_batch, qCap_batch, qImg_batch, ELA_img_batch, dgl.batch(img_dgl_graph).to(config.device), dgl.batch(text_dgl_graph).to(config.device), samples_key

def collate_SGK(batch):
    samples = [item[0] for item in batch]
    samples_key=[item[1] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    ELA_img_batch=[]
    labels = []
    img_dgl_graph=[]
    text_dgl_graph=[]
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])
        img_dgl_graph.append(sample['img_dgl_graph'])
        text_dgl_graph.append(sample['text_dgl_graph'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    ELA_img_batch=torch.stack(ELA_img_batch,dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, qCap_batch, qImg_batch, ELA_img_batch, dgl.batch(img_dgl_graph).to(config.device), dgl.batch(text_dgl_graph).to(config.device), samples_key

def collate_SGKE_SG(batch):
    samples = [item[0] for item in batch]
    max_t_evidence_len = max([item[1] for item in batch])
    max_i_evidence_len = max([item[2] for item in batch])
    samples_key=[item[3] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    ELA_img_batch=[]
    i_evidence_batch = []
    t_evidence_batch = []
    labels = []
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        t_evidence = sample['t_evidence']
        cap_len = len(t_evidence)
        for i in range(max_t_evidence_len - cap_len):
            t_evidence.append("")
        t_evidence_tokenizer = config._tokenizer(t_evidence, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                                 truncation=True).to(config.device)
        t_evidence_batch.append(t_evidence_tokenizer)

        if len(sample['i_evidence'].shape) > 2:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], sample['i_evidence'].shape[1], sample['i_evidence'].shape[2], sample['i_evidence'].shape[3])
        else:
            padding_size = (max_i_evidence_len - sample['i_evidence'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['i_evidence'], torch.zeros(padding_size)), dim=0)
        i_evidence_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    i_evidence_batch = torch.stack(i_evidence_batch, dim=0).to(config.device)
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    ELA_img_batch=torch.stack(ELA_img_batch,dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, t_evidence_batch, i_evidence_batch, qCap_batch, qImg_batch, ELA_img_batch, samples_key

def collate_SGKE_SG_E(batch):
    samples = [item[0] for item in batch]
    samples_key=[item[1] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    ELA_img_batch=[]
    labels = []
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])

        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        ELA_img_batch.append(sample['ELA_img'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)

    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    ELA_img_batch=torch.stack(ELA_img_batch,dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    return labels, qCap_batch, qImg_batch, ELA_img_batch, samples_key

def collate_TI(batch):
    labels=[]
    qCap_batch=[]
    qImg_batch=[]
    samples_key = []
    for item in batch:
        labels.append(item[0])
        qCap_batch.append(item[1])
        qImg_batch.append(item[2])
        samples_key.append(item[3])
    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    qImg_batch=torch.stack(qImg_batch,dim=0).to(config.device)
    return labels,qCap_batch,qImg_batch,samples_key

def collate_text(batch):
    labels=[]
    qCap_batch=[]
    samples_key = []
    for item in batch:
        labels.append(item[0])
        qCap_batch.append(item[1])
        samples_key.append(item[2])
    #text_max_length = get_maxlength(qCap_batch)
    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(
        config.device)  # qCap_batch=[batch_size,tokenizer]
    labels = torch.stack(labels, dim=0).to(config.device)  # labels=[batch_size]
    return labels,qCap_batch,samples_key
