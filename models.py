import torch
import torch.nn as nn
from transformers import BertModel

class BertRegressor(nn.Module):

    def __init__(self, n_fc_layers=2, dropout=0.0):
        super(BertRegressor, self).__init__()
        # For the essay scoring predictor, the number of FC layers is set to 2.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        layers = []
        for _ in range(n_fc_layers):
            layers.extend([
                nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
                nn.GELU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid(),
        ])
        self.score_fn = nn.Sequential(*layers)

    
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.bert(input_ids, attention_mask).last_hidden_state # (batch_size, sequence_length, hidden_size)
        first_token_tensor = last_hidden_state[:, 0] # (batch_size, hidden_state)
        y_pred = self.score_fn(first_token_tensor).view(-1) # (batch_size)
        return y_pred
