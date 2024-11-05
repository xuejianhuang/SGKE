import torch as th
import torch.nn as nn
import config
from transformers import SwinModel

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax


class SGATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            edge_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=True,
            activation=None,
            allow_zero_in_degree=False,
            bias=True,
    ):
        super(SGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # Initialize linear layers based on input features type
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # Initialize attention parameters
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Initialize bias
        self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,))) if bias else None

        # Initialize residual connection
        self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False) if residual else None

        # Initialize edge feature layers
        self._edge_feats = edge_feats
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)
        self.attn_edge = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError("There are 0-in-degree nodes in the graph, output for those nodes will be invalid.")

            # Process source and destination features
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            # Linearly transform the edge features
            n_edges = edge_feat.shape[:-1]
            feat_edge = self.fc_edge(edge_feat).view(*n_edges, self._num_heads, self._out_feats)

            # Add edge features to graph
            graph.edata["ft_edge"] = feat_edge

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # Calculate scalar for each edge
            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            graph.edata["ee"] = ee

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # Compute edge attention
            graph.apply_edges(fn.u_add_v("el", "er", "e_tmp"))

            # Combine attention weights of source and destination node
            graph.edata["e"] = graph.edata["e_tmp"] + graph.edata["ee"]

            # Create new edge features combining source node features and edge features
            graph.apply_edges(fn.u_add_e("ft", "ft_edge", "ft_combined"))

            e = self.leaky_relu(graph.edata.pop("e"))
            # Compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # Multiply combined features by attention coefficients
            graph.edata["m_combined"] = graph.edata["ft_combined"] * graph.edata["a"]

            # Copy edge features and sum them up
            graph.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))

            rst = graph.dstdata["ft"]
            # Apply residual connection
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst += resval
            # Add bias
            if self.bias is not None:
                rst += self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # Apply activation function
            if self.activation:
                rst = self.activation(rst)

            return (rst, graph.edata["a"]) if get_attention else rst


class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        # Load pre-trained Swin Transformer model
        self.swin = SwinModel.from_pretrained(config.swin_transformer)

    def forward(self, x):
        # Extract image features using Swin Transformer
        features = self.swin(pixel_values=x).last_hidden_state
        return features  # The output feature dimension of Swin Transformer is (49,768)