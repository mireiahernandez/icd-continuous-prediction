import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
import ipdb
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

    def forward(self, encoding, cutoffs=None):
        # encoding: Tensor of size num_chunks x hidden_size

        attention_weights = F.softmax(
            encoding @ self.label_queries, dim=0
        )  # (num_chunks x num_labels)

        attention_value = encoding.T @ attention_weights  # hidden_size x num labels

        score = torch.sum(attention_value * self.label_weights, dim=0)  # num_labels
        # probability = torch.sigmoid(score)

        # return probability
        return score.unsqueeze(0)  # CHANGED THIS FOR DEBUGGING


class HierARDocumentTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers=1, nhead=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=nhead
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, document_encodings):
        # flag is causal = True so that it cannot attend to future document embeddings
        mask = nn.Transformer.generate_square_subsequent_mask(
            sz=document_encodings.shape[0]
        )
        document_encodings = self.transformer_encoder(
            document_encodings, mask=mask
        ).squeeze(
            1
        )  # shape Nc x 1 x D

        return document_encodings


class NextDocumentCategoryPredictor(nn.Module):
    def __init__(self, hidden_size, num_categories):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_categories = num_categories
        self.linear = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.linear2 = nn.Linear(
            self.hidden_size // 2, num_categories + 1
        )  # 11 is the number of categories

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, document_encodings):
        # predict next document category
        categories = self.relu(self.linear(document_encodings))
        categories = self.softmax(self.linear2(categories))
        return categories


class NextDocumentEmbeddingPredictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.relu2 = nn.ReLU()

    def forward(self, document_encodings):
        # predict next document embedding
        next_document_encodings = self.relu1(self.linear(document_encodings))
        next_document_encodings = self.linear2(next_document_encodings)
        return next_document_encodings


class TemporalMultiHeadLabelAttentionClassifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        seq_len,
        num_labels,
        num_heads,
        device,
        all_tokens=True,
        reduce_computation=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device
        self.all_tokens = all_tokens
        self.reduce_computation = reduce_computation

        self.multiheadattn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )

        self.label_queries = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.num_labels, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.label_weights = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )

    def forward(self, encoding, all_tokens=True, cutoffs=None):
        # encoding: Tensor of size (Nc x T) x H
        # mask: Tensor of size Nn x (Nc x T) x H
        # temporal_encoding = Nn x (N x T) x hidden_size
        T = self.seq_len
        if not self.all_tokens:
            T = 1  # only use the [CLS]-token representation
        Nc = int(encoding.shape[0] / T)
        H = self.hidden_size
        Nl = self.num_labels

        # label query: shape L, H
        # encoding: hape NcxT, H
        # query shape:  Nn, L, H
        # key shape: Nn, Nc*T, H
        # values shape: Nn, Nc*T, H
        # key padding mask: Nn, Nc*T (true if ignore)
        # output: N, L, H
        mask = torch.ones(size=(Nc, Nc * T), dtype=torch.bool).to(device=self.device)
        for i in range(Nc):
            mask[i, : (i + 1) * T] = False

        # only mask out at 2d, 5d, 13d and no DS to reduce computation
        # get list of cutoff indices from cutoffs dictionary

        if self.reduce_computation:
            cutoff_indices = [cutoffs[key][0] for key in cutoffs]
            mask = mask[cutoff_indices, :]

        attn_output = self.multiheadattn.forward(
            query=self.label_queries.repeat(mask.shape[0], 1, 1),
            key=encoding.repeat(mask.shape[0], 1, 1),
            value=encoding.repeat(mask.shape[0], 1, 1),
            key_padding_mask=mask,
            need_weights=False,
        )[0]

        score = torch.sum(
            attn_output
            * self.label_weights.unsqueeze(0).view(
                1, self.num_labels, self.hidden_size
            ),
            dim=2,
        )
        return score


class Model(nn.Module):
    """Model for ICD-9 code temporal predictions.

    Model based on HTDC (Ng et al, 2022), with the original
    contribution of adding the temporal aspect."""

    def __init__(self, config, device):
        super().__init__()
        for key in config:
            setattr(self, key, config[key])

        self.seq_len = 512
        self.hidden_size = 768
        self.device = device
        self._initialize_embeddings()

        # base transformer
        self.transformer = AutoModel.from_pretrained(self.base_checkpoint)

        # LWAN
        if self.use_multihead_attention:
            self.label_attn = TemporalMultiHeadLabelAttentionClassifier(
                self.hidden_size,
                self.seq_len,
                self.num_labels,
                self.num_heads_labattn,
                device=device,
                all_tokens=self.use_all_tokens,
                reduce_computation=self.reduce_computation,
            )
            # self.label_attn = TemporalLabelAttentionClassifier(
            #     self.hidden_size,
            #     self.seq_len,
            #     self.num_labels,
            #     self.num_heads_labattn,
            #     device=device,
            #     all_tokens=self.use_all_tokens,
            # )
        else:
            self.label_attn = LabelAttentionClassifier(
                self.hidden_size, self.num_labels
            )
        # hierarchical AR transformer
        if not self.is_baseline:
            self.document_regressor = HierARDocumentTransformer(
                self.hidden_size, self.num_layers, self.num_attention_heads
            )
        if self.aux_task in ("next_document_embedding", "last_document_embedding"):
            self.document_predictor = NextDocumentEmbeddingPredictor(self.hidden_size)
        elif self.aux_task == "next_document_category":
            self.category_predictor = NextDocumentCategoryPredictor(
                self.hidden_size, self.num_categories
            )
        elif self.aux_task != "none":
            raise ValueError(
                "auxiliary_task must be next_document_embedding or next_document_category or none"
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
        cutoffs,
        note_end_chunk_ids=None,
        token_type_ids=None,
        is_evaluation=False,
        return_attn_weights=False,
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

        sequence_output = sequence_output[
            :, 0, :
        ]  # remove the singleton to get something of shape [#chunks, hidden_size] or [#chunks*512, hidden_size]

        # if not baseline, add document autoregressor
        if not self.is_baseline:
            # document regressor returns document embeddings and predicted categories
            sequence_output = self.document_regressor(
                sequence_output.view(-1, 1, self.hidden_size)
            )
        # make aux predictions
        if self.aux_task in ("next_document_embedding", "last_document_embedding"):
            if self.apply_transformation:
                aux_predictions = self.document_predictor(sequence_output)
            else:
                aux_predictions = sequence_output
        elif self.aux_task == "next_document_category":
            aux_predictions = self.category_predictor(sequence_output)
        elif self.aux_task == "none":
            aux_predictions = None
        # apply label attention at document-level
        if is_evaluation == False:
            scores = self.label_attn(
                sequence_output, cutoffs=cutoffs
            )  # apply label attention at token-level
            return scores, sequence_output, aux_predictions

        else:
            return sequence_output
