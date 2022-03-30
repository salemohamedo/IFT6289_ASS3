# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import (BertForSequenceClassification, BertModel)


class BertFineTuneModel(nn.Module):
    def __init__(self, num_labels=2):
        super(BertFineTuneModel, self).__init__()
        self.plm = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.loss = nn.CrossEntropyLoss()

        for param in self.plm.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        logits = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)['logits']
        return logits


class BertPrefixTuneModel(nn.Module):
    def __init__(self, num_labels=2, prefix_len=20):
        super(BertPrefixTuneModel, self).__init__()
        self.plm = BertModel.from_pretrained('bert-base-uncased')
        self.cls_head = nn.Linear(self.plm.config.hidden_size, num_labels)
        self.loss = nn.CrossEntropyLoss()

        self.prefix_len = prefix_len
        self.prefix_tokens = torch.arange(self.prefix_len).long()
        self.prefix_embedding = nn.Embedding(prefix_len,
                                             2 * self.plm.config.num_hidden_layers * self.plm.config.hidden_size)

        for param in self.plm.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.plm.device)
        past_key_values = self.prefix_embedding(prefix_tokens)
        past_key_values = past_key_values.view(batch_size,
                                               self.prefix_len,
                                               self.plm.config.num_hidden_layers * 2,
                                               self.plm.config.num_attention_heads,
                                               self.plm.config.hidden_size // self.plm.config.num_attention_heads)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        prefix_attn_mask = torch.ones(batch_size, self.prefix_len).to(self.plm.device)
        attention_mask = torch.cat((prefix_attn_mask, attention_mask), dim=1)
        reprs = self.plm(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                         return_dict=True)['pooler_output']
        logits = self.cls_head(reprs)
        return logits
