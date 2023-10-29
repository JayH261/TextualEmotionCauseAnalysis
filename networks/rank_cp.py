import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from networks.our_model import *

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.decoder = Decoder() 
        self.seq = Seq2Seq(self.decoder) # currently fail here
        self.fc5 = nn.Linear(768,1)
        
    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len, epoch,testing=False,target=None):
        # bert encoder to give clause representation
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),attention_mask=bert_masks_b.to(DEVICE)) 
        bert_output = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        # print("BERT OUTPUT SHAPE:", bert_output.shape)
        # print("FINISHED BERT ENCODER, ABOUT TO GO INTO SEQ")
        pred_e = self.seq(bert_output, doc_len, epoch,testing,target) # we feed data here and decoder inside failed to decode our data :)
        # print("FINISHED SEQUENCE AND HAVE PREDICTION RESULT !!!")
        return pred_e

    def loss_pre(self, pred_e, y_emotions, source_length):
        #print('loss function shape is ',pred_e.shape,y_emotions.shape)   #seq_len * batch  * class  .  batch * seq_len
        y_emotions = torch.LongTensor(y_emotions).to(DEVICE)
        packed_y = torch.nn.utils.rnn.pack_padded_sequence(pred_e, list(source_length),enforce_sorted=False).data
        target_ = torch.nn.utils.rnn.pack_padded_sequence(y_emotions.permute(1,0), list(source_length),enforce_sorted=False).data
        loss_e  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y), target_)
        return loss_e

    def batched_index_select(self, bert_output, bert_clause_b):
        #print("BERT OUTPUT:",bert_output)
        hidden_state = bert_output[0]
        doc_sents_h = torch.zeros(bert_clause_b.size(0), bert_clause_b.size(1) + 1, hidden_state.size(2)).to(DEVICE)
        #print("DOC SENTS H shape BEFORE:", doc_sents_h.shape)
        #print(hidden_state.shape,bert_clause_b.shape)
        for i in range(doc_sents_h.shape[0]):
            for j in range(doc_sents_h.shape[1]):
                if j == doc_sents_h.shape[1] -1:
                    hidden = hidden_state[i,bert_clause_b[i,j-1]:,:]
                    weight = F.softmax(self.fc5(hidden),0)
                    hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                elif bert_clause_b[i,j]!=0:
                    if j==0:
                        hidden = hidden_state[i,0:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    else:
                        hidden = hidden_state[i,bert_clause_b[i,j-1]:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                else:
                    hidden = hidden_state[i,bert_clause_b[i,j-1]:,:]
                    weight = F.softmax(self.fc5(hidden),0)
                    hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                    break    
        #print("DOC SENTS H shape AFTER:", doc_sents_h.shape)
        return doc_sents_h

