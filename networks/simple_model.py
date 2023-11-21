import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from config import DEVICE
from transformers import BertModel

# target decoder for sequence2sequence to employ
class Decoder(torch.nn.Module):
    def __init__(self, encoder_hidden_size=100, decoder_hidden_size=100, num_classes=10):
        super(Decoder, self).__init__()

        self.label_embedding = torch.nn.Embedding(num_classes, 50) 
        self.rnn = torch.nn.GRU(150, decoder_hidden_size)
        self.hidden2label = torch.nn.Linear(decoder_hidden_size, num_classes)
        self.linear1 = torch.nn.Linear(2 * encoder_hidden_size, decoder_hidden_size)

    def forward(self, inputs, last_hidden, encoder_outputs, current_encoder_outputs, time_step, max_len, mask,
                inputs_mask=None):
        embedded = self.label_embedding(inputs).unsqueeze(0)
        input = self.linear1(current_encoder_outputs).permute(1,0,2)
        input1 = torch.cat((embedded, input), 2)
        # GRU layer
        output, hidden = self.rnn(input1, last_hidden) # contextualized clause repr
        # mapping the hidden state of the decoder to an output label
        output = self.hidden2label(output).squeeze(0)
        output = F.log_softmax(output, dim=1)
        return output, hidden

class Seq2Seq(torch.nn.Module):
    def __init__(self, decoder):
        super(Seq2Seq, self).__init__()
        self.gru = nn.GRU(768,100, bidirectional=True,batch_first=True)
        self.decoder = decoder
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, source, source_len, epoch, testing=False,target=None):
        # source : 32 * 75 * 30 , target : 32 * 75
        # now :  batch * seq * hidden_size
        output1 ,hidden = self.gru(source) # contextualized clause representation (encoder gru)
        batch_size = source.size(0)
        max_len = max(source_len)  # in other sq2seq, max_len should be target.size()
        outputs = Variable(torch.zeros(max_len, batch_size,10)).to(DEVICE)

        #  encoder_outputs.shape   75 * 32 * 200
        hidden = hidden[:1]
        output = Variable(torch.zeros((batch_size))).long().to(DEVICE)
        mask = torch.zeros(batch_size, 10).long().to(DEVICE)
        
        for t in range(max_len):
            current_encoder_outputs = output1[:,t, :].unsqueeze(1)
            output, hidden = self.decoder(output, hidden, output1.permute(1,0,2), current_encoder_outputs, t,max_len, mask, source_len)
            outputs[t] = output
            is_teacher = random.random() < 1 - epoch * 0.05
            top1 = output.data.max(1)[1]
            if testing:
                output = Variable(top1).to(DEVICE)
            elif is_teacher:
                target = torch.LongTensor(target)
                output = Variable(target.permute(1,0)[t]).to(DEVICE)
            else:
                output = Variable(top1).to(DEVICE)
        return outputs

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.decoder = Decoder() 
        self.seq = Seq2Seq(self.decoder)
        self.fc5 = nn.Linear(768,1)
        
    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len, epoch,testing=False,target=None):
        # bert encoder to give clause representation
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),attention_mask=bert_masks_b.to(DEVICE)) 
        bert_output = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        pred_e = self.seq(bert_output, doc_len, epoch,testing,target)
        return pred_e

    def loss_pre(self, pred_e, y_emotions, source_length):
        #print('loss function shape is ',pred_e.shape,y_emotions.shape)   #seq_len * batch  * class  .  batch * seq_len
        y_emotions = torch.LongTensor(y_emotions).to(DEVICE)
        packed_y = torch.nn.utils.rnn.pack_padded_sequence(pred_e, list(source_length),enforce_sorted=False).data
        target_ = torch.nn.utils.rnn.pack_padded_sequence(y_emotions.permute(1,0), list(source_length),enforce_sorted=False).data
        loss_e  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y), target_)
        return loss_e

    def batched_index_select(self, bert_output, bert_clause_b):
        '''
        Provided by UTOS authors
        
        Purpose of this function is to process the BERT hidden states based on the clause indices provided in bert_clause_b 
        and compute the weighted sum of the selected hidden states
        '''
        hidden_state = bert_output[0]
        doc_sents_h = torch.zeros(bert_clause_b.size(0), bert_clause_b.size(1) + 1, hidden_state.size(2)).to(DEVICE)
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
                        # If it's the first position, select the hidden states from the beginning to the current index
                        hidden = hidden_state[i,0:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    else:
                        # Select the hidden states between the previous and current index
                        hidden = hidden_state[i,bert_clause_b[i,j-1]:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                else:
                    # If the current index is 0, select the hidden states based on the previous index and apply a weighted sum
                    hidden = hidden_state[i, bert_clause_b[i, j - 1]:, :]
                    hidden = hidden_state[i,bert_clause_b[i,j-1]:,:]
                    weight = F.softmax(self.fc5(hidden),0)
                    hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                    break    
        return doc_sents_h
