from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BartModel


class LabelSmoothCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        # logs = self.log_softmax(logits)
        logs = torch.log(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
        
# 权重初始化，默认xavier
def init_network(model, method='xavier', seed=99):
    for name, w in model.named_parameters():
        # if exclude not in name:
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass


class PGNModel(nn.Module):
    def __init__(self, config: dict):
        super(PGNModel, self).__init__()
        self.bart = BartModel.from_pretrained(config['model_path'])
        self.attention_network = Attention(config)
        init_network(self.attention_network)        
        self.dropout = nn.Dropout(config['dropout'])
        self.p_gen_linear = nn.Linear(config['hidden_dim'] * 3, 1)
        init_network(self.p_gen_linear)
        #p_vocab
        self.out1 = nn.Linear(config['hidden_dim'] * 2, config['hidden_dim'])
        init_network(self.out1)
        self.out2 = nn.Linear(config['hidden_dim'], config['vocab_size'])
        init_network(self.out2)


    def forward(self, encoder_input, decoder_input, encoder_input_mask, decoder_input_mask, decoder_output, coverage):
        d_outputs = self.bart(input_ids=encoder_input, attention_mask=encoder_input_mask,
                    decoder_input_ids=decoder_input, decoder_attention_mask=decoder_input_mask)
        d_output = self.dropout(d_outputs['last_hidden_state'])
        e_output = self.dropout(d_outputs['encoder_last_hidden_state'])
        
        ht_hat, attn_dist, coverage_loss, _ = self.attention_network(
                                        decoder_outputs=d_output, 
                                        decoder_input_mask=decoder_input_mask,
                                        encoder_outputs=e_output, 
                                        enc_padding_mask=encoder_input_mask, 
                                        coverage=coverage)
        p_gen_input = torch.cat((ht_hat, d_output, self.bart.shared(decoder_input)), -1) 
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = torch.sigmoid(p_gen)
        output = torch.cat((d_output, ht_hat), -1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=-1)
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        encoder_input_ = encoder_input.unsqueeze(1).expand(encoder_input.size(0), d_output.size(1), encoder_input.size(1))

        final_dist = vocab_dist_.scatter_add(2, encoder_input_, attn_dist_)
        decoder_output = decoder_output.contiguous().view(-1)
        if coverage_loss == -1:
            loss = LabelSmoothCEV1(ignore_index=-1)(final_dist.view(-1, final_dist.size(-1)), decoder_output)
        else:
            loss = LabelSmoothCEV1(ignore_index=-1)(final_dist.view(-1, final_dist.size(-1)), decoder_output) + coverage_loss

        return loss


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # attention
        self.config = config
        self.W_h = nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False)
        if config['is_coverage']:
            self.W_c = nn.Linear(1, config['hidden_dim'], bias=False)
        self.decode_proj = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.v = nn.Linear(config['hidden_dim'], 1, bias=False)

    def forward(self, decoder_outputs, decoder_input_mask, encoder_outputs, enc_padding_mask, coverage):
        encoder_feature = self.W_h(encoder_outputs)
        dec_fea = self.decode_proj(decoder_outputs)  # W_s             # B * seq_d * hid_dim
        dec_fea_expanded = dec_fea.unsqueeze(2).expand(dec_fea.size(0), dec_fea.size(1), encoder_outputs.size(1), dec_fea.size(2)) # B x seq_d X seq_e x hid_dim
        encoder_feature_expanded = encoder_feature.unsqueeze(1).expand(encoder_feature.size(0), dec_fea.size(1), encoder_feature.size(1), encoder_feature.size(2))

        att_features = encoder_feature_expanded + dec_fea_expanded  # batch, seq_d, seq_e, hidden

        if self.config['is_coverage']:
            ht_hat_list, attn_dist_list, step_loss_list = [], [], []
            # 累加初始到当前时间步的注意力权重
            for i in range(att_features.size(1)):
                coverage_feature = self.W_c(coverage.unsqueeze(-1))   # B * seq_e * hidden
                att_features_step = att_features[:, i] + coverage_feature # b * seq_e * hidden
                e_step = torch.tanh(att_features_step)   
                scores_step = self.v(e_step).squeeze(-1)   # b * seq_e
                attn_dist_ = F.softmax(scores_step, dim=-1)*enc_padding_mask # batch, seq_e
                normalization_factor = attn_dist_.sum(-1, keepdim=True) 
                attn_dist_step = attn_dist_ / normalization_factor
                attn_dist_step = attn_dist_step.unsqueeze(1) 
                attn_dist_list.append(attn_dist_step)
                ht_hat_step = torch.bmm(attn_dist_step, encoder_outputs)  #  b * 1 * hidden
                ht_hat_list.append(ht_hat_step)
                step_coverage_loss = torch.sum(torch.min(attn_dist_step.squeeze(1), coverage), 1) # attn_dist当前时间步的注意力权重；coverage先前时间步的注意力权重的累积
                coverage = torch.add(coverage, attn_dist_step.squeeze(1)) # b * seq_e
                step_loss = self.config['cov_loss_wt'] * step_coverage_loss
                step_loss_list.append(step_loss)

            ht_hat = torch.cat(ht_hat_list, dim=1)
            attn_dist = torch.cat(attn_dist_list, dim=1)
            step_loss_list = torch.stack(step_loss_list)
            converge_loss = torch.sum(step_loss_list*decoder_input_mask.T) / torch.sum(decoder_input_mask)

            return ht_hat, attn_dist, converge_loss, coverage
        else:
            e = torch.tanh(att_features)   
            scores = self.v(e).squeeze(-1)             # batch, seq_d, seq_e

            attn_dist_ = F.softmax(scores, dim=-1)*enc_padding_mask.unsqueeze(1).expand(enc_padding_mask.size(0), scores.size(1), enc_padding_mask.size(1)) # batch, seq_d, seq_e
            normalization_factor = attn_dist_.sum(-1, keepdim=True) 
            attn_dist = attn_dist_ / normalization_factor

            ht_hat = torch.bmm(attn_dist, encoder_outputs)  #  # b * seq_d * hidden

            return ht_hat, attn_dist, -1, None
        