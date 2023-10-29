import torch
import time
import json
import numpy as np
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from config import DEVICE

# target decoder for sequence2sequence to employ
class Decoder(torch.nn.Module):
    def __init__(self, encoder_hidden_size=100, decoder_hidden_size=100, attention_hidden_size=100, num_classes=10,
                 dropout=0.2):
        super(Decoder, self).__init__()

        self.label_embedding = torch.nn.Embedding(num_classes, 50)
        self.dropout = torch.nn.Dropout(dropout)
        # print("ABOUT TO HIT ATTENTION LAYER IN OUR MODEL DECODER")
        self.attention = self.attention = nn.MultiheadAttention(embed_dim=50, num_heads=5, dropout=0.2) # take in q, k, v
        self.q = nn.Sequential(nn.Linear(encoder_hidden_size, 50, bias=False), nn.SELU(), nn.Dropout(p=0.2))
        self.k = nn.Sequential(nn.Linear(2 * encoder_hidden_size, 50, bias=False), nn.SELU(), nn.Dropout(p=0.2))
        self.v = nn.Sequential(nn.Linear(2 * encoder_hidden_size, 50, bias=False), nn.SELU(), nn.Dropout(p=0.2))
        # print("SUCCESSFULLY LOADED LUONG GATE ATTENTION LAYER")

        self.rnn = torch.nn.GRU(150, decoder_hidden_size)

        self.linear = torch.nn.Linear(2 * decoder_hidden_size + 50, decoder_hidden_size)
        self.hidden2label = torch.nn.Linear(decoder_hidden_size, num_classes)
        self.linear1 = torch.nn.Linear(2 * encoder_hidden_size, decoder_hidden_size)

    def forward(self, inputs, last_hidden, encoder_outputs, current_encoder_outputs, time_step, max_len, mask,
                inputs_mask=None):
        #print("\nINSIDE DECODER FORWARD")
        embedded = self.label_embedding(inputs).unsqueeze(0)
        #embedded = self.dropout(embedded)

        #print("FINISHED LABEL EMBEDDING")
        
        input = F.leaky_relu(self.linear1(current_encoder_outputs)).permute(1,0,2)

        # print("FINISHED LEAKY RELU LINEAR1")
        # print('input shape:',input.shape)
        input1 = torch.cat((embedded, input), 2)
        output, hidden = self.rnn(input1, last_hidden) # contextualized clause repr
        # print("FINISHED GRU\n")
        # print("GRU OUTPUT shape:", output.shape)
        # print("ENCODER OUTPUT shape:", encoder_outputs.shape)    
        #print("RNN OUTPUT:", output)

        # Q, K, V inputs for multihead attention layer
        q = self.q(output)
        k = self.k(encoder_outputs)
        v = self.v(encoder_outputs)
        # print("\nQ, K, V Shape:")
        # print(q.shape,k.shape,v.shape)
        # Make sure that embedding size of q,k,v match with attention layer embed_dim
        context, attn_weights = self.attention(q, k, v)
        #print("Attention context output shape:", context.shape)
        #context = context.unsqueeze(0) # for luong_gate_attention class
        #print("\nFINISHED ATTENTION LAYER\n")
        output = torch.cat([context, input, output], 2)
        output = F.leaky_relu(self.linear(output))
        # mapping the hidden state of the decoder to an output label
        output = self.hidden2label(output).squeeze(0)
        output = F.log_softmax(output, dim=1)
        # print("Decoder forward output shape:", output.shape)
        # print("FINISHED DECODER FORWARD\n")
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
        output1 ,hidden = self.gru(source) # contextualized clause representation
        batch_size = source.size(0)
        max_len = max(source_len)  # in other sq2seq, max_len should be target.size()
        outputs = Variable(torch.zeros(max_len, batch_size,10)).to(DEVICE)

        #  encoder_outputs.shape   75 * 32 * 200
        hidden = hidden[:1]
        output = Variable(torch.zeros((batch_size))).long().to(DEVICE)
        mask = torch.zeros(batch_size, 10).long().to(DEVICE)
        for t in range(max_len):
            current_encoder_outputs = output1[:,t, :].unsqueeze(1)
            # print("INSIDE SEQ2SEQ BEFORE DECODER")
            # print("\nBEFORE DECODER OUTPUT:", output)
            # print("\nINPUT TARGET:", target)
            output, hidden = self.decoder(output, hidden, output1.permute(1,0,2), current_encoder_outputs, t,max_len, mask, source_len)
            #print("\nAFTER DECODER OUTPUT:", output)
            # print("\nINSIDE SEQ2SEQ AFTER DECODER")
            outputs[t] = output
            is_teacher = random.random() < 1 - epoch * 0.05
            top1 = output.data.max(1)[1]
            if testing:
                output = Variable(top1).to(DEVICE)
            elif is_teacher:
                target = torch.LongTensor(target)
                # print("TARGET LONGTENSOR SHAPE:", target.shape)
                output = Variable(target.permute(1,0)[t]).to(DEVICE)
            else:
                output = Variable(top1).to(DEVICE)
        return outputs
