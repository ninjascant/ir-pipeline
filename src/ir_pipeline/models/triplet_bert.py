import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


def calculate_loss(query_embed, pos_ctx_embed, neg_ctx_embed):
    loss_fn = nn.TripletMarginLoss(margin=3)
    loss = loss_fn(query_embed, pos_ctx_embed, neg_ctx_embed)
    return loss


class TripletBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

    def get_embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        out = torch.mean(last_hidden_state, 1)
        return out

    def forward(
            self,
            pos_ctx_input_ids,
            neg_ctx_input_ids,
            pos_ctx_attention_mask,
            neg_ctx_attention_mask,
            query_input_ids,
            query_attention_mask,
    ) -> torch.Tensor:
        query_embed = self.get_embed(query_input_ids, query_attention_mask)
        pos_ctx_embed = self.get_embed(pos_ctx_input_ids, pos_ctx_attention_mask)
        neg_ctx_embed = self.get_embed(neg_ctx_input_ids, neg_ctx_attention_mask)

        loss = calculate_loss(query_embed, pos_ctx_embed, neg_ctx_embed)
        return loss,
