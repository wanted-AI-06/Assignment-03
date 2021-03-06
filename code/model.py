import torch
from torch import nn
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
)
import torch.nn.functional as F
import numpy as np


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RobertaForStsRegression(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForStsRegression, self).__init__(config)
        self.roberta = RobertaModel(config=config)  # Load pretrained model

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.sentence_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.sentence_fc_layer2 = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.sentence_fc_layer3 = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.embedding_vectors=FCLayer(config.hidden_size, 1, 0.1)
        self.dense = FCLayer(config.hidden_size * 6, config.hidden_size , 0.1)
        self.dense2 = FCLayer(config.hidden_size , config.hidden_size, 0.1)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            0.1,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        s1_mask=None,
        s2_mask=None,
        # ctx_mask=None,
    ):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=None, output_hidden_states=True
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        s1_h = self.entity_average(sequence_output, s1_mask)
        s2_h = self.entity_average(sequence_output, s2_mask)
        s1_h = self.sentence_fc_layer(s1_h)
        s2_h = self.sentence_fc_layer(s2_h)
        
        #roberta ????????? 2?????? layer??? hidden_states??? ????????? ????????? ??????
        sequence_output_sec_layer=outputs['hidden_states'][2]
        ss1_h = self.entity_average(sequence_output_sec_layer, s1_mask)
        ss2_h = self.entity_average(sequence_output_sec_layer, s2_mask)
        ss1_h = self.sentence_fc_layer2(ss1_h)
        ss2_h = self.sentence_fc_layer2(ss2_h)

        #roberta ????????? 3?????? layer??? hidden_states??? ????????? ????????? ??????
        sequence_output_sec_layer2=outputs['hidden_states'][3]
        sss1_h = self.entity_average(sequence_output_sec_layer2, s1_mask)
        sss2_h = self.entity_average(sequence_output_sec_layer2, s2_mask)
        sss1_h = self.sentence_fc_layer3(sss1_h)
        sss2_h = self.sentence_fc_layer3(sss2_h)

        #emb_vec=self.embedding_vectors(outputs['hidden_states'][0])
        #dump_shape=(outputs['hidden_states'][0].shape[0],outputs['hidden_states'][0].shape[2]-outputs['hidden_states'][0].shape[1])
        #dump=torch.zeros(dump_shape).to('cuda')
        #emb_vec=torch.cat([emb_vec.squeeze(2),dump],dim=1)

        # Concat -> fc_layer
        # concat_h = torch.cat([pooled_output, s1_h, s2_h], dim=-1)
        concat_h = torch.cat([s1_h, s2_h,ss1_h, ss2_h,sss1_h, sss2_h], dim=-1)# last layer, 2layer, 3layer??? ??? ?????? ????????? concat
        concat_h = self.dense(concat_h) #??? ?????? ????????? ?????? ??????
        concat_h = self.dense2(concat_h) #??? ?????? ????????? ?????? ??????2
        logits = self.label_classifier(concat_h)
        # print(outputs)

        # add hidden states and attention if they are here
        outputs = (logits,) + ()
        # print(outputs)
        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs
