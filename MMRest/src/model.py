import pdb
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, Clip, get_resNet, get_resNet2, get_resNet3, get_resNet4

from transformers import BertModel, BertConfig
from modules.transformer import TransformerEncoder
import torchvision
from torchvision import models




def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class MMRest(nn.Module):
    def __init__(self, hp):
        """Construct MMRest model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        # 'if add va MMILB module---add_va'
        self.add_va = hp.add_va
        # args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim  768 20 5
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        # 2. Crossmodal Attentions

        self.visual_enc = RNNEncoder(  # (rnn): LSTM(20, 16) Linear(in_features=16, out_features=16, bias=True)
            in_size=hp.d_vin,  # 20
            hidden_size=hp.d_vh,  # 16
            out_size=hp.d_vout,  # 16
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(  # (rnn): LSTM(20, 16)Linear(in_features=16, out_features=16, bias=True)
            in_size=hp.d_ain,  # 5
            hidden_size=hp.d_ah,  # 16
            out_size=hp.d_aout,  # 16
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        self.fusion_prj = SubNet(
            in_size=512,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.embed_dim = hp.embed_dim
        self.num_heads = hp.num_heads
        self.layers = hp.layers
        self.attn_dropout = hp.attn_dropout
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout
        self.embed_dropout = hp.embed_dropout
        self.attn_mask = hp.attn_mask

        self.ta_clip = Clip(768, 64)
        self.tv_clip = Clip(768, 64)
        self.av_clip = Clip(64, 64)
        self.mass_t_clip = Clip(768, 64)
        self.mass_v_clip = Clip(64, 64)
        self.mass_a_clip = Clip(64, 64)
        self.mass_tav_clip = Clip(64, 64)

        self.conv_tv = nn.Conv1d(
            in_channels=832, out_channels=64, kernel_size=1)
        self.conv_av = nn.Conv1d(
            in_channels=128, out_channels=64, kernel_size=1)
        self.conv_vt = nn.Conv1d(
            in_channels=832, out_channels=64, kernel_size=1)
        self.conv_tav = nn.Conv1d(
            in_channels=896, out_channels=64, kernel_size=1)
        self.conv_res = nn.Conv1d(
            in_channels=200, out_channels=512, kernel_size=1)
        self.getresNet = get_resNet()
        self.getresNet2 = get_resNet2()
        self.getresNet3 = get_resNet3()
        self.getresNet4 = get_resNet4()
        self.tfn_tv = Clip(256036, 256036)  
        self.tfn_ta = Clip(256036, 256036)
        self.tfn_av = Clip(256036, 256036)


        self.conv_tfn_t = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4,  kernel_size=(3, 3)),

            nn.ReLU(),
            nn.Conv2d(
                in_channels=4, out_channels=6,  kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=(3, 3)), nn.ReLU(),
            nn.Flatten()

        )
        self.conv_tfn_v = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4,  kernel_size=(3, 3)), nn.ReLU(),

            nn.Conv2d(
                in_channels=4, out_channels=6,   kernel_size=(3, 3)), nn.ReLU(),

            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=(3, 3)), nn.ReLU(),
            nn.Flatten()
        )
        self.conv_tfn_a = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4,  kernel_size=(3, 3)), nn.ReLU(),

            nn.Conv2d(
                in_channels=4, out_channels=6,   kernel_size=(3, 3)),  nn.ReLU(),

            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=(3, 3)), nn.ReLU(),
            nn.Flatten()
        )

        self.clus_text = nn.Linear(768, 64)


    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, c_sim_loss=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """  # (32,50,768)
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type,
                                 bert_sent_mask)
        # pdb.set_trace()
        text = enc_word[:, 0, :]
        clus_text = self.clus_text(text)
        
        acoustic, aco_rnn_output = self.acoustic_enc(
            acoustic, a_len)
        clus_acoustic = acoustic

        visual, vis_rnn_output = self.visual_enc(
            visual, v_len)
        clus_visual = visual

  
        tav = torch.cat([text, acoustic, visual], dim=1).unsqueeze(2)
        qwer = self.getresNet4(tav).squeeze(2)

        res = self.getresNet(text.unsqueeze(2)).squeeze(2)
        res2 = self.getresNet2(visual.unsqueeze(2)).squeeze(2)  
        res3 = self.getresNet3(acoustic.unsqueeze(2)).squeeze(2)  


        res_all = torch.cat([res, res2, res3], dim=1)  # [64, 1536]
        xxx = torch.cat([res_all, qwer], dim=1).unsqueeze(2)
        res_mm = self.conv_res(xxx).squeeze(2)

        # pdb.set_trace()



        c_fusion = torch.cat([clus_text, clus_acoustic, clus_visual], dim=0)  # (3B, D)
        # pdb.set_trace()


        pred_from_geometry = None
        if c_sim_loss is not None:
            try:
                # pdb.set_trace()
                centers, M_0, Delta_M_list, K, bins, beta = c_sim_loss.get_current_clusters_info(c_fusion.device)
                if M_0 is not None and K > 0:

                    c_fusion = torch.nan_to_num(c_fusion)
                    centers = torch.nan_to_num(centers)
                    M_0 = torch.nan_to_num(M_0)
                    Delta_M_list = torch.nan_to_num(Delta_M_list)
                    

                    M_0 = 0.5 * (M_0 + M_0.transpose(0, 1))
                    
                    B, D = c_fusion.shape
                    

                    M_list = M_0.unsqueeze(0) + Delta_M_list + beta * torch.eye(D, device=M_0.device, dtype=M_0.dtype).unsqueeze(0)


                    diff = c_fusion.unsqueeze(1) - centers.unsqueeze(0)  # (3B, K, D)
                    

                    diff_M = torch.einsum('bkd,kdf->bkf', diff, M_list)
                    dists = (diff_M * diff).sum(dim=-1)  # (3B, K)

                    if K >= 2:
                        vals_two, idxs_two = torch.topk(dists, k=2, largest=False)
                        idx_near = idxs_two[:, 0]
                        idx_second = idxs_two[:, 1]
                    else:
                        idx_near = torch.zeros(c_fusion.shape[0], dtype=torch.long, device=c_fusion.device)
                        idx_second = torch.zeros(c_fusion.shape[0], dtype=torch.long, device=c_fusion.device)

                    c_near = centers[idx_near]
                    c_second = centers[idx_second]


                    u = c_fusion - c_near
                    v = c_second - c_near
                    u_M0 = torch.matmul(u, M_0)
                    num = (u_M0 * v).sum(dim=-1)
                    v_M0 = torch.matmul(v, M_0)
                    v_norm_sq = (v_M0 * v).sum(dim=-1)
                    v_norm_sq = torch.nan_to_num(v_norm_sq).clamp_min(1e-12)
                    den = torch.sqrt(v_norm_sq)
                    r = torch.nan_to_num(num) / den
                    r = torch.nan_to_num(r)


                    bin_to_mid = {
                        0: -2.5, 1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.5, 6: 2.5
                    }
                    if isinstance(bins, (list, tuple)):
                        bins_tensor = torch.tensor(bins, device=c_fusion.device, dtype=torch.long)
                    else:
                        bins_tensor = torch.as_tensor(bins, device=c_fusion.device, dtype=torch.long)
                    
                    near_bins = bins_tensor[idx_near]
                    far_bins = bins_tensor[idx_second]
                    mids = torch.tensor([bin_to_mid[int(b)] for b in range(7)], device=c_fusion.device, dtype=c_fusion.dtype)
                    base_near = mids[near_bins]
                    base_far = mids[far_bins]
                    denom_bins = (torch.abs(base_far) - torch.abs(base_near))
                    denom_bins = torch.abs(denom_bins).clamp_min(1e-2)
                    bias = r / denom_bins
                    bias_r = r.unsqueeze(-1)
                    part_1_r, part_2_r, part_3_r = torch.chunk(bias_r, 3, dim=0)
                    avg_r = (part_1_r + part_2_r + part_3_r) / 3
                    
                    pred_from_geometry = (base_near + bias).unsqueeze(-1)
                    part1, part2, part3 = torch.chunk(pred_from_geometry, 3, dim=0)
                    avg_result = (part1 + part2 + part3) / 3
            except Exception as e:
                print(f"警告: 几何预测失败，使用回退：{e}")


        if pred_from_geometry is not None:
            fusion, preds = self.fusion_prj(res_mm)
            preds = 0.05 * avg_r + preds
        else:
            fusion, preds = self.fusion_prj(res_mm)  # 32, 512, 1



        return preds, clus_text, clus_visual, clus_acoustic
