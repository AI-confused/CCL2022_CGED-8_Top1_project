import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BERT_Model(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrained_model'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.label_ignore_id = config['label_ignore_id']

        if tie_cls_weight:
            self.tie_cls_weight()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, trg_ids=None):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        logits = self.classifier(bert_output.last_hidden_state)
        
        loss = None
        if trg_ids is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
            loss = loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))

        return loss, logits
    
    def tie_cls_weight(self):
            self.classifier.weight = self.bert.embeddings.word_embeddings.weight


class BERT_LSTM_Model(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_LSTM_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrained_model'])

        self.lstm = nn.LSTM(self.bert.config.hidden_size, 
                            self.bert.config.hidden_size//2, 
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                            dropout=config['dropout']
                            )
        
        self.FC = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.label_ignore_id = config['label_ignore_id']

        if tie_cls_weight:
            self.tie_cls_weight()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, trg_ids=None, err_flag=None):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out,_ = self.lstm(bert_output.last_hidden_state)
        out_fc = self.FC(out)
        logits = self.classifier(out_fc)

        loss = None
        if trg_ids is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
            loss = loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))

        return loss, logits
    
    def tie_cls_weight(self):
            self.classifier.weight = self.bert.embeddings.word_embeddings.weight


class BERT_LSTM_flag_Model(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_LSTM_flag_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrained_model'])

        self.lstm = nn.LSTM(self.bert.config.hidden_size, 
                            self.bert.config.hidden_size//2, 
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                            dropout=config['dropout']
                            )
        
        self.lstm_flag = nn.LSTM(self.bert.config.hidden_size, 
                            200//2, 
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                            dropout=config['dropout']
                            )
        
        self.FC = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        
        self.FC_flag = nn.Linear(200, 2)

        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.label_ignore_id = config['label_ignore_id']

        if tie_cls_weight:
            self.tie_cls_weight()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, trg_ids=None, err_flag=None):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out,_ = self.lstm(bert_output.last_hidden_state)
        out_fc = self.FC(out)
        logits = self.classifier(out_fc)

        out2,_ = self.lstm_flag(bert_output.last_hidden_state)
        out_fc2 = self.FC_flag(out2)

        loss = None
        if trg_ids is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
            loss = loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))
            loss2 = loss_function(out_fc2.view(-1, 2), err_flag.view(-1))

        return loss+loss2, logits, out_fc2
    
    def tie_cls_weight(self):
            self.classifier.weight = self.bert.embeddings.word_embeddings.weight

