import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
import config
from dgl.nn import GATConv

from layers import SwinTransformer,SGATConv

class SGKE(nn.Module):
    def __init__(self,in_feats,edge_feats,out_feats,num_heads=2,n_layers=2,residual=False):
        super(SGKE, self).__init__()
        # Initialize SGAT
        self.SGAT_layers = nn.ModuleList([
            SGATConv(in_feats if i == 0 else out_feats,
                     edge_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Linear layer for DGL features
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Cross-Attention layers
        self.cross_attention_text = nn.MultiheadAttention(config.text_dim,
                                                          config.att_num_heads,
                                                          batch_first=True,
                                                          dropout=config.att_dropout)
        self.cross_attention_image = nn.MultiheadAttention(config.img_dim,
                                                           config.att_num_heads,
                                                           batch_first=True,
                                                           dropout=config.att_dropout)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 6, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        t_evidence, v_evidence, qCap, qImg, ELA_img, img_dgl, text_dgl=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]

        # Textual evidence
        t_evidence_features = torch.stack([self.bert(**cap)['last_hidden_state'][:, 0, :] for cap in t_evidence],
                                          dim=0)  # shape=[batch_size,max_captions_num,768]
        textual_evidence, _ = self.cross_attention_text(lstm_out, t_evidence_features, t_evidence_features)
        textual_evidence = textual_evidence.mean(dim=1)  # 对齐的文本证据 [batch_size,768]

        # Visual evidence
        i_evidence_features = torch.stack([self.st_img(img).mean(dim=1) for img in v_evidence], dim=0)  # shape=[batch_size,max_images_num,768]
        visual_evidence, _ = self.cross_attention_image(qImg_feature, i_evidence_features, i_evidence_features)
        visual_evidence = visual_evidence.mean(dim=1)  #  Aligned visual evidence, shape=[batch_size,768]

        # DGL layers
        img_dgl_node_feats = img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']

        for layer in self.SGAT_layers:
            img_dgl_node_feats = layer(img_dgl, img_dgl_node_feats, img_dgl.edata['x']).mean(dim=1)
            text_dgl_node_feats = layer(text_dgl, text_dgl_node_feats, text_dgl.edata['x']).mean(dim=1)

        img_dgl.ndata['h'] = img_dgl_node_feats
        text_dgl.ndata['h'] = text_dgl_node_feats

        # Read out node features
        img_dgl_feature = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        # Semantic similarity features
        SG_features = F.leaky_relu(self.dgl_linear(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1)))

        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean, ELA_img_feature_mean,textual_evidence, visual_evidence, SG_features), dim=-1))

        return logits

# w/o ELA
class SGKE_ELA(nn.Module):
    def __init__(self,in_feats,edge_feats,out_feats,num_heads=2,n_layers=2,residual=False):
        super(SGKE_ELA, self).__init__()
        # Initialize SGAT
        self.SGAT_layers = nn.ModuleList([
            SGATConv(in_feats if i == 0 else out_feats,
                     edge_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Linear layer for DGL features
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Cross-Attention layers
        self.cross_attention_text = nn.MultiheadAttention(config.text_dim,
                                                          config.att_num_heads,
                                                          batch_first=True,
                                                          dropout=config.att_dropout)
        self.cross_attention_image = nn.MultiheadAttention(config.img_dim,
                                                           config.att_num_heads,
                                                           batch_first=True,
                                                           dropout=config.att_dropout)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        t_evidence, v_evidence, qCap, qImg, img_dgl, text_dgl=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]

        # Textual evidence
        t_evidence_features = torch.stack([self.bert(**cap)['last_hidden_state'][:, 0, :] for cap in t_evidence],
                                          dim=0)  # shape=[batch_size,max_captions_num,768]
        textual_evidence, _ = self.cross_attention_text(lstm_out, t_evidence_features, t_evidence_features)
        textual_evidence = textual_evidence.mean(dim=1)  # 对齐的文本证据 [batch_size,768]

        # Visual evidence
        i_evidence_features = torch.stack([self.st_img(img).mean(dim=1) for img in v_evidence], dim=0)  # shape=[batch_size,max_images_num,768]
        visual_evidence, _ = self.cross_attention_image(qImg_feature, i_evidence_features, i_evidence_features)
        visual_evidence = visual_evidence.mean(dim=1)  #  Aligned visual evidence, shape=[batch_size,768]

        # DGL layers
        img_dgl_node_feats = img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']

        for layer in self.SGAT_layers:
            img_dgl_node_feats = layer(img_dgl, img_dgl_node_feats, img_dgl.edata['x']).mean(dim=1)
            text_dgl_node_feats = layer(text_dgl, text_dgl_node_feats, text_dgl.edata['x']).mean(dim=1)

        img_dgl.ndata['h'] = img_dgl_node_feats
        text_dgl.ndata['h'] = text_dgl_node_feats

        # Read out node features
        img_dgl_feature = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        # Semantic similarity features
        SG_features = F.leaky_relu(self.dgl_linear(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1)))

        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean,textual_evidence, visual_evidence, SG_features), dim=-1))

        return logits

# w/o Textual evidence
class SGKEv(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, num_heads=2, n_layers=2, residual=False):
        super(SGKEv, self).__init__()
        # Initialize SGAT
        self.SGAT_layers = nn.ModuleList([
            SGATConv(in_feats if i == 0 else out_feats,
                     edge_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert = config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Linear layer for DGL features
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Cross-Attention layers

        self.cross_attention_image = nn.MultiheadAttention(config.img_dim,
                                                           config.att_num_heads,
                                                           batch_first=True,
                                                           dropout=config.att_dropout)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        v_evidence, qCap, qImg, ELA_img, img_dgl, text_dgl = data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean = qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]

        # Visual evidence
        i_evidence_features = torch.stack([self.st_img(img).mean(dim=1) for img in v_evidence],
                                          dim=0)  # shape=[batch_size,max_images_num,768]
        visual_evidence, _ = self.cross_attention_image(qImg_feature, i_evidence_features, i_evidence_features)
        visual_evidence = visual_evidence.mean(dim=1)  # Aligned visual evidence, shape=[batch_size,768]

        # DGL layers
        img_dgl_node_feats = img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']

        for layer in self.SGAT_layers:
            img_dgl_node_feats = layer(img_dgl, img_dgl_node_feats, img_dgl.edata['x']).mean(dim=1)
            text_dgl_node_feats = layer(text_dgl, text_dgl_node_feats, text_dgl.edata['x']).mean(dim=1)

        img_dgl.ndata['h'] = img_dgl_node_feats
        text_dgl.ndata['h'] = text_dgl_node_feats

        # Read out node features
        img_dgl_feature = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        # Semantic similarity features
        SG_features = F.leaky_relu(self.dgl_linear(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1)))

        logits = self.classifier(torch.cat(
            (qcap_feature, qImg_feature_mean, ELA_img_feature_mean, visual_evidence, SG_features),
            dim=-1))

        return logits

# w/o Visual evidence
class SGKEt(nn.Module):
    def __init__(self,in_feats,edge_feats,out_feats,num_heads=2,n_layers=2,residual=False):
        super(SGKEt, self).__init__()
        # Initialize SGAT
        self.SGAT_layers = nn.ModuleList([
            SGATConv(in_feats if i == 0 else out_feats,
                     edge_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Linear layer for DGL features
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Cross-Attention layers
        self.cross_attention_text = nn.MultiheadAttention(config.text_dim,
                                                          config.att_num_heads,
                                                          batch_first=True,
                                                          dropout=config.att_dropout)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        t_evidence, qCap, qImg, ELA_img, img_dgl, text_dgl=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]

        # Textual evidence
        t_evidence_features = torch.stack([self.bert(**cap)['last_hidden_state'][:, 0, :] for cap in t_evidence],
                                          dim=0)  # shape=[batch_size,max_captions_num,768]
        textual_evidence, _ = self.cross_attention_text(lstm_out, t_evidence_features, t_evidence_features)
        textual_evidence = textual_evidence.mean(dim=1)  # 对齐的文本证据 [batch_size,768]


        # DGL layers
        img_dgl_node_feats = img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']

        for layer in self.SGAT_layers:
            img_dgl_node_feats = layer(img_dgl, img_dgl_node_feats, img_dgl.edata['x']).mean(dim=1)
            text_dgl_node_feats = layer(text_dgl, text_dgl_node_feats, text_dgl.edata['x']).mean(dim=1)

        img_dgl.ndata['h'] = img_dgl_node_feats
        text_dgl.ndata['h'] = text_dgl_node_feats

        # Read out node features
        img_dgl_feature = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        # Semantic similarity features
        SG_features = F.leaky_relu(self.dgl_linear(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1)))

        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean, ELA_img_feature_mean,textual_evidence, SG_features), dim=-1))

        return logits

# w/o Textual evidence and Textual evidence
class SGK(nn.Module):
    def __init__(self,in_feats,edge_feats,out_feats,num_heads=2,n_layers=2,residual=False):
        super(SGK, self).__init__()
        # Initialize SGAT
        self.SGAT_layers = nn.ModuleList([
            SGATConv(in_feats if i == 0 else out_feats,
                     edge_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Linear layer for DGL features
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        qCap, qImg, ELA_img, img_dgl, text_dgl=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]

        # DGL layers
        img_dgl_node_feats = img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']

        for layer in self.SGAT_layers:
            img_dgl_node_feats = layer(img_dgl, img_dgl_node_feats, img_dgl.edata['x']).mean(dim=1)
            text_dgl_node_feats = layer(text_dgl, text_dgl_node_feats, text_dgl.edata['x']).mean(dim=1)

        img_dgl.ndata['h'] = img_dgl_node_feats
        text_dgl.ndata['h'] = text_dgl_node_feats

        # Read out node features
        img_dgl_feature = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        # Semantic similarity features
        SG_features = F.leaky_relu(self.dgl_linear(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1)))

        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean, ELA_img_feature_mean, SG_features), dim=-1))

        return logits

# w/o Scene graph-based text-image semantic matching
class SGKE_SG(nn.Module):
    def __init__(self):
        super(SGKE_SG, self).__init__()

        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)



        # Cross-Attention layers
        self.cross_attention_text = nn.MultiheadAttention(config.text_dim,
                                                          config.att_num_heads,
                                                          batch_first=True,
                                                          dropout=config.att_dropout)
        self.cross_attention_image = nn.MultiheadAttention(config.img_dim,
                                                           config.att_num_heads,
                                                           batch_first=True,
                                                           dropout=config.att_dropout)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        t_evidence, v_evidence, qCap, qImg, ELA_img=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]

        # Textual evidence
        t_evidence_features = torch.stack([self.bert(**cap)['last_hidden_state'][:, 0, :] for cap in t_evidence],
                                          dim=0)  # shape=[batch_size,max_captions_num,768]
        textual_evidence, _ = self.cross_attention_text(lstm_out, t_evidence_features, t_evidence_features)
        textual_evidence = textual_evidence.mean(dim=1)  # 对齐的文本证据 [batch_size,768]

        # Visual evidence
        i_evidence_features = torch.stack([self.st_img(img).mean(dim=1) for img in v_evidence], dim=0)  # shape=[batch_size,max_images_num,768]
        visual_evidence, _ = self.cross_attention_image(qImg_feature, i_evidence_features, i_evidence_features)
        visual_evidence = visual_evidence.mean(dim=1)  #  Aligned visual evidence, shape=[batch_size,768]



        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean, ELA_img_feature_mean,textual_evidence, visual_evidence), dim=-1))

        return logits

# w/o Dual evidence and Scene graph-based text-image semantic matching
class SGKE_SG_E(nn.Module):
    def __init__(self):
        super(SGKE_SG_E, self).__init__()

        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        qCap, qImg, ELA_img=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        #qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]


        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean, ELA_img_feature_mean), dim=-1))

        return logits

# o/h Text and Image
class TI(nn.Module):
    def __init__(self):
        super(TI, self).__init__()
        self.st_img = SwinTransformer()
        self.bert = config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        qCap, qImg = data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean = qImg_feature.mean(dim=1)  # [batch_size,768]

        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean), dim=-1))

        return logits

# o/h Scene graph-based text-image semantic matching
class SG(nn.Module):
    def __init__(self,in_feats,edge_feats,out_feats,num_heads=2,n_layers=2,residual=False):
        super(SG, self).__init__()
        self.in_feats=in_feats
        self.edge_feats=edge_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.n_layers=n_layers
        self.residual=residual

        self.layers=nn.ModuleList()
        self.layers.append(SGATConv(in_feats=self.in_feats, edge_feats=self.edge_feats,
                                    out_feats=self.out_feats, num_heads=self.num_heads, residual=self.residual, allow_zero_in_degree=True))
        for i in range(1,self.n_layers):
            self.layers.append(SGATConv(in_feats=self.out_feats, edge_feats=self.edge_feats,
                                        out_feats=self.out_feats, num_heads=self.num_heads, residual=self.residual, allow_zero_in_degree=True))

        self.classifier=nn.Sequential(
            nn.Linear(self.out_feats*2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )
    def forward(self, data):
        img_dgl,text_dgl = data
        img_dgl_node_feats=img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']
        for l in range(self.n_layers):
            img_dgl_node_feats = self.layers[l](img_dgl,img_dgl_node_feats, img_dgl.edata['x']).mean(dim=1)
            text_dgl_node_feats = self.layers[l](text_dgl,text_dgl_node_feats, text_dgl.edata['x']).mean(dim=1)
        img_dgl.ndata['h']=img_dgl_node_feats
        text_dgl.ndata['h']=text_dgl_node_feats

        img_dgl_feature=dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        logits = self.classifier(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1))
        return logits

#o/h Text
class Text(nn.Module):
    def __init__(self):
        super(Text, self).__init__()
        self.bert = config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        qCap, = data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        #qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        qcap_feature = lstm_out.mean(dim=1)

        logits = self.classifier(qcap_feature)

        return logits

# o/h Image
class Image(nn.Module):
    def __init__(self):
        super(Image, self).__init__()
        self.st=SwinTransformer()
        self.classifier=nn.Sequential(
            nn.Linear(config.img_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )
    def forward(self, data):
        img, = data
        img_feature=self.st(img).mean(dim=1)
        logits = self.classifier(img_feature)
        return logits

# o/h ELA
class ELA(nn.Module):
    def __init__(self):
        super(ELA, self).__init__()
        self.st_ELA=SwinTransformer()
        self.classifier=nn.Sequential(
            nn.Linear(config.img_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )
    def forward(self, data):
        ELA_img, = data
        ELA_img_feature=self.st_ELA(ELA_img).mean(dim=1)
        logits = self.classifier(ELA_img_feature)
        return logits

# SGAT->GAT
class GKE(nn.Module):
    def __init__(self,in_feats,out_feats,num_heads=2,n_layers=2,residual=False):
        super(GKE, self).__init__()
        # Initialize SGAT
        self.GAT_layers = nn.ModuleList([
            GATConv(in_feats if i == 0 else out_feats,
                     out_feats,
                     num_heads,
                     residual=residual,
                     allow_zero_in_degree=True)
            for i in range(n_layers)
        ])
        self.st_ELA = SwinTransformer()
        self.st_img = SwinTransformer()
        self.bert=config._bert_model

        # LSTM for textual data
        self.lstm = nn.LSTM(config.text_dim, config.text_dim // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Linear layer for DGL features
        self.dgl_linear = nn.Linear(out_feats * 2, out_feats)

        # Cross-Attention layers
        self.cross_attention_text = nn.MultiheadAttention(config.text_dim,
                                                          config.att_num_heads,
                                                          batch_first=True,
                                                          dropout=config.att_dropout)
        self.cross_attention_image = nn.MultiheadAttention(config.img_dim,
                                                           config.att_num_heads,
                                                           batch_first=True,
                                                           dropout=config.att_dropout)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 6, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        # Freeze BERT parameters if required
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, data):
        t_evidence, v_evidence, qCap, qImg, ELA_img, img_dgl, text_dgl=data

        qcap_hidden = self.bert(**qCap)['last_hidden_state']  # Shape: [batch_size,seq_len,768]
        lstm_out, _ = self.lstm(qcap_hidden)
        qcap_feature = lstm_out[:, -1, :]  # Get last LSTM output
        # qcap_feature = lstm_out.mean(dim=1)

        # Process images and ELA
        qImg_feature = self.st_img(qImg)  # shape [batch_size,49,768]
        qImg_feature_mean=qImg_feature.mean(dim=1)  # [batch_size,768]
        ELA_img_feature_mean = self.st_ELA(ELA_img).mean(dim=1)  # [batch_size,768]

        # Textual evidence
        t_evidence_features = torch.stack([self.bert(**cap)['last_hidden_state'][:, 0, :] for cap in t_evidence],
                                          dim=0)  # shape=[batch_size,max_captions_num,768]
        textual_evidence, _ = self.cross_attention_text(lstm_out, t_evidence_features, t_evidence_features)
        textual_evidence = textual_evidence.mean(dim=1)  # 对齐的文本证据 [batch_size,768]

        # Visual evidence
        i_evidence_features = torch.stack([self.st_img(img).mean(dim=1) for img in v_evidence], dim=0)  # shape=[batch_size,max_images_num,768]
        visual_evidence, _ = self.cross_attention_image(qImg_feature, i_evidence_features, i_evidence_features)
        visual_evidence = visual_evidence.mean(dim=1)  #  Aligned visual evidence, shape=[batch_size,768]

        # DGL layers
        img_dgl_node_feats = img_dgl.ndata['x']
        text_dgl_node_feats = text_dgl.ndata['x']

        for layer in self.GAT_layers:
            img_dgl_node_feats = layer(img_dgl, img_dgl_node_feats).mean(dim=1)
            text_dgl_node_feats = layer(text_dgl, text_dgl_node_feats).mean(dim=1)

        img_dgl.ndata['h'] = img_dgl_node_feats
        text_dgl.ndata['h'] = text_dgl_node_feats

        # Read out node features
        img_dgl_feature = dgl.readout_nodes(img_dgl, 'h', op='mean')
        text_dgl_feature = dgl.readout_nodes(text_dgl, 'h', op='mean')

        # Semantic similarity features
        SG_features = F.leaky_relu(self.dgl_linear(torch.cat((img_dgl_feature, text_dgl_feature), dim=-1)))

        logits = self.classifier(torch.cat((qcap_feature, qImg_feature_mean, ELA_img_feature_mean,textual_evidence, visual_evidence, SG_features), dim=-1))

        return logits
