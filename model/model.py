import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
)


class LabelAttentionClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.label_queries = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.label_weights = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )

    def forward(self, encoding):
        # encoding: Tensor of size num_chunks x hidden_size

        attention_weights = F.softmax(
            encoding @ self.label_queries, dim=0
        )  # (num_chunks x num_labels)

        attention_value = encoding.T @ attention_weights  # hidden_size x num labels

        score = torch.sum(attention_value * self.label_weights, dim=0)  # num_labels
        probability = torch.sigmoid(score)

        return probability


class TemporalLabelAttentionClassifier(nn.Module):
    def __init__(self, hidden_size, seq_len, num_labels, device):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.device = device

        self.label_queries = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.label_weights = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )

    def forward(self, encoding, note_end_chunk_ids):
        # encoding: Tensor of size (Nc x T) x H
        # mask: Tensor of size Nn x (Nc x T) x H
        # temporal_encoding = Nn x (N x T) x hidden_size
        T = self.seq_len
        Nc = int(encoding.shape[0] / T)
        Nn = len(note_end_chunk_ids)
        H = self.hidden_size
        Nl = self.num_labels

        mask = torch.zeros(size=(Nn, Nc * T)).to(device=self.device)
        for i in range(Nn):
            mask[i, : (note_end_chunk_ids[i] + 1) * T] = float("-inf")
        ###### FIX IT IS NOT note_end_chunk_ids we need to mulitply by T !!!!!!!!!!!!!!!!!!!!!!!!!

        # shape Nn x H x Nc*T
        # (Nn x NcT x 1) x (1 x NcT x H) -
        # temporal_encoding = torch.mul(mask.unsqueeze(2), encoding.unsqueeze(0)) # Nn x NcT x H
        breakpoint()
        # encoding ((Nc x T) x H) x label queries (H x Nl) = ((NcxT) x Nl)
        attention_scores = encoding @ self.label_queries

        # mask attention scores: Nn x NcxT x Nl
        attention_scores = attention_scores.unsqueeze(0) + mask.unsqueeze(2)

        # shape Nn x Nc*T x Nl
        attention_weights = F.softmax(attention_scores, dim=0)

        # shape Nn x Nl x H
        attention_values = attention_weights.view(Nn, Nl, Nc * T) @ encoding.view(
            Nc * T, H
        )
        # attention_value = torch.bmm(
        #     encoding.view(Nn,H,Nc*T),
        #     attention_weights.view(Nn,Nc*T,Nl)
        # )

        # shape (Nn, Nl)
        # label weights (H, Nl)
        score = torch.sum(
            attention_values * self.label_weights.unsqueeze(0).view(1, Nl, H), dim=1
        )  # num_labels
        # shape (Nn, Nl)
        probability = torch.sigmoid(score)
        return probability


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        for key in config:
            setattr(self, key, config[key])

        self.seq_len = 512
        self.hidden_size = 768
        self._initialize_embeddings()

        # base transformer
        self.transformer = AutoModel.from_pretrained(self.model_name)

        # hierarchical transformer
        self.transformer2_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=self.num_attention_heads
        )
        self.transformer2 = nn.TransformerEncoder(
            self.transformer2_layer, num_layers=self.num_layers
        )

        # LWAN
        self.label_attn = LabelAttentionClassifier(self.hidden_size, self.num_labels)
        self.temp_label_attn = TemporalLabelAttentionClassifier(
            self.hidden_size, self.seq_len, self.num_labels
        )

    def _initialize_embeddings(self):
        self.pelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversepelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.delookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversedelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.celookup = nn.parameter.Parameter(
            torch.normal(0, 0.1, size=(15, 1, self.hidden_size), dtype=torch.float),
            requires_grad=True,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        seq_ids,
        category_ids,
        note_end_chunk_ids,
        token_type_ids=None,
        **kwargs
    ):
        max_seq_id = seq_ids[-1].item()
        reverse_seq_ids = max_seq_id - seq_ids

        chunk_count = input_ids.size()[0]
        reverse_pos_ids = (chunk_count - torch.arange(chunk_count) - 1).to(self.device)

        sequence_output = self.transformer(input_ids, attention_mask).last_hidden_state
        if self.use_positional_embeddings:
            sequence_output += self.pelookup[: sequence_output.size()[0], :, :]
        if self.use_reverse_positional_embeddings:
            sequence_output += torch.index_select(
                self.reversepelookup, dim=0, index=reverse_pos_ids
            )

        if self.use_document_embeddings:
            sequence_output += torch.index_select(self.delookup, dim=0, index=seq_ids)
        if self.use_reverse_document_embeddings:
            sequence_output += torch.index_select(
                self.reversedelookup, dim=0, index=reverse_seq_ids
            )

        if self.use_category_embeddings:
            sequence_output += torch.index_select(
                self.celookup, dim=0, index=category_ids
            )
        if self.use_all_tokens:
            # before: sequence_output shape [batchsize, seqlen, hiddensize] = [# chunks, 512, hidden size]
            # after: sequence_output shape [#chunks*512, 1, hidden size]
            sequence_output = sequence_output.view(-1, 1, self.hidden_size)
        else:
            sequence_output = sequence_output[:, [0], :]
        if self.num_layers > 0:
            sequence_output = self.transformer2(sequence_output)[:, 0, :]
        else:
            sequence_output = sequence_output[
                :, 0, :
            ]  # remove the singleton to get something of shape [#chunks*512, hidden_size]

        # sequence_output = self.label_attn(sequence_output) # apply label attention at token-level
        sequence_output = self.temp_label_attn(
            sequence_output, note_end_chunk_ids
        )  # apply label attention at token-level

        return sequence_output
