import torch
import torch.nn as nn
from transformers import AutoModel


def calculate_loss(query_embed, pos_ctx_embed, neg_ctx_embed):
    loss_fn = nn.TripletMarginLoss(margin=3)
    loss = loss_fn(query_embed, pos_ctx_embed, neg_ctx_embed)
    return loss


class TripletBert(nn.Module):
    def __init__(self, model_name_or_path: str):
        super(TripletBert, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.linear = nn.Linear(312, 312)
        self.ln = nn.LayerNorm(312)

    def get_embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        out = torch.mean(last_hidden_state, 1)
        return out

    def forward(
            self,
            **kwargs
    ) -> torch.Tensor:
        print(**kwargs.keys())
        query_embed = self.get_embed(query_input_ids, query_attention_mask)
        pos_ctx_embed = self.get_embed(pos_context_input_ids, pos_context_attention_mask)
        neg_ctx_embed = self.get_embed(neg_context_input_ids, neg_context_attention_mask)

        loss = calculate_loss(query_embed, pos_ctx_embed, neg_ctx_embed)
        return loss,