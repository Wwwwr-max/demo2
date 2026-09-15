import torch.nn as nn
from transformers import BertModel

class Mymodel(nn.Module):
    def __init__(self,model_path,num_labels,dropout):
        super(Mymodel,self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.liner = nn.Linear(self.bert.config.hidden_size,num_labels)

    def forward(self,input_ids,attention_mask,token_type_ids = None,labels = None):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        seq_out = bert_out.last_hidden_state
        seq_out = self.dropout(seq_out)
        logits = self.liner(seq_out)
        return logits

