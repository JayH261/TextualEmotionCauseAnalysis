import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from networks.simple_model import *
from transformers import AdamW, get_linear_schedule_with_warmup
from lion_pytorch import Lion
from utils.utils import *
from pytorch_pretrained_bert.optimization import BertAdam
import random
import apex

def inference_one_batch(configs, batch, model,epoch,):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
    pred_e = model(bert_token_b, bert_segment_b, bert_masks_b,bert_clause_b, doc_len_b,epoch,True)
    doc_couples_pred_b = inference_pair(pred_e,doc_id_b)
    return doc_couples_b, doc_couples_pred_b, doc_id_b

def inference_one_epoch(configs, batches, model,pair,epoch):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    for batch in batches:
        doc_couples, doc_couples_pred, doc_id_b = inference_one_batch(configs, batch, model,epoch)
        doc_id_all.extend(doc_id_b)
        doc_couples_pred_all.extend([doc_couples_pred])

    metric_ec, metric_e, metric_c = eval_func(pair, doc_couples_pred_all)
    return metric_ec, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all

def inference_pair( predict,doc_id_b):
    pre_test_1, pre_test_2, pre_test_3 = [], [], []

    result = predict.argmax(dim=2).permute(1,0).cpu().numpy()  # batch *seq_len

    for i in range(result.shape[0]):
        e1, e2, e3, c1, c2, c3 = [], [], [], [], [], []
        for j in range(result.shape[1]):
            if result[i][j] != 0:
                if int((result[i][j] + 2) / 3) == 1:
                    if result[i][j] % 3 == 1:
                        e1.append(j + 1)
                    elif result[i][j] % 3 == 2:
                        c1.append(j + 1)
                    else:
                        e1.append(j + 1)
                        c1.append(j + 1)
                elif int((result[i][j] + 2) / 3) == 2:
                    if result[i][j] % 3 == 1:
                        e2.append(j + 1)
                    elif result[i][j] % 3 == 2:
                        c2.append(j + 1)
                    else:
                        e2.append(j + 1)
                        c2.append(j + 1)
                else:
                    if result[i][j] % 3 == 1:
                        e3.append(j + 1)
                    elif result[i][j] % 3 == 2:
                        c3.append(j + 1)
                    else:
                        e3.append(j + 1)
                        c3.append(j + 1)

        for p in e1:
            for q in c1:
                pre_test_1.append(int(doc_id_b[i])*10000 + p * 100 + q)
        for p in e2:
            for q in c2:
                pre_test_1.append(int(doc_id_b[i])*10000 + p * 100 + q)
        for p in e3:
            for q in c3:
                pre_test_1.append(int(doc_id_b[i])*10000 + p * 100 + q)
    #if pre_test_1 != None:
        #print(pre_test_1)
    return pre_test_1

def get_optimizer(optimizer_name, model, loader):
    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.seq.parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 1e-5},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0},
        {'params': params_rest,'lr':0.0005,'weight_decay': 1e-7}
    ]
    
    if optimizer_name == "BertAdam":
        optimizer = BertAdam(params,
                             lr=1e-5,
                             warmup=0.1,
                             t_total=len(loader) // configs.gradient_accumulation_steps * configs.epochs)
    elif optimizer_name == "AdamW":
        paramsbert = []
        paramsbert0reg = []
        paramsothers = []
        paramsothers0reg = []
        for name, parameters in model.named_parameters():
            #print(name, ':', parameters.shape)
            if not parameters.requires_grad:
                continue
            if 'bert' in name:
                if '.bias' in name or 'LayerNorm.weight' in name:
                    paramsbert0reg += [parameters]
                else:
                    paramsbert += [parameters]
            else:
                paramsothers += [parameters]

        params = [dict(params=paramsbert,lr=2e-5, weight_decay=1e-2),
                      dict(params=paramsothers,lr=1e-4,weight_decay=1e-5),
                      dict(params=paramsbert0reg, weight_decay=0.0)]
        optimizer = AdamW(params,lr=3e-6,weight_decay = 0)
    elif optimizer_name == "Lion":
        paramsbert = []
        paramsbert0reg = []
        paramsothers = []
        paramsothers0reg = []
        for name, parameters in model.named_parameters():
            if not parameters.requires_grad:
                continue
            if 'bert' in name:
                if '.bias' in name or 'LayerNorm.weight' in name:
                    paramsbert0reg += [parameters]
                else:
                    paramsbert += [parameters]
            else:
                paramsothers += [parameters]

        params = [dict(params=paramsbert, weight_decay=1e-2),
                      dict(params=paramsothers,lr=1e-4,weight_decay=1e-5),
                      dict(params=paramsbert0reg, weight_decay=0.0)]
        optimizer = Lion(params, lr=0.0005, weight_decay = 0)
    return optimizer

def main(configs, fold_id):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] =str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    
    # Train data loader and pairs
    train_loader = build_training_data(configs, fold_id = fold_id)
    train_pair = get_ecpair(configs, fold_id=fold_id, data_type='train')
    
    test_loader = build_inference_data(configs, fold_id = fold_id, data_type = "test")
    test_pair = get_ecpair(configs, fold_id=fold_id, data_type='test')

    model = Network(configs).to(DEVICE)

    #optimizer = get_optimizer("BertAdam", model, train_loader)
    optimizer = get_optimizer("AdamW", model, train_loader)
    #optimizer = get_optimizer("Lion", model, train_loader)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()
    train_metric_ec, train_metric_e, train_metric_c = (-1,-1,-1), None, None
    test_metric_ec, test_metric_e, test_metric_c = (-1,-1,-1), None, None
    early_stop_flag = None

    # TRAINING
    for epoch in range(1, configs.epochs+1):
        train_doc_couples_pred_all = []
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            # Input values from data batch
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch

            # Model Prediction
            pred_e = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, doc_len_b,epoch,False,y_emotions_b)  # seq_len * batch * 10
            
            loss = model.loss_pre(pred_e, y_emotions_b, doc_len_b)
            loss = loss / configs.gradient_accumulation_steps
            if train_step <= 20:
                print('epoch: ',epoch,loss)

            loss.backward()

            if train_step % configs.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            show_time = 200

            if train_step % show_time ==0:
                with torch.no_grad():
                    model.eval()
                    test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model,test_pair,1)
                    if test_ec[2] > test_metric_ec[2]:
                        early_stop_flag = 1
                        test_metric_ec, test_metric_e, test_metric_c = test_ec, test_e, test_c
                        print('within train step: epoch',epoch,'test  p,r,f1 is :',test_ec)
                    else:
                        early_stop_flag += 1

        with torch.no_grad():
            model.eval()
            test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model,test_pair,1)
            if test_ec[2] > test_metric_ec[2]:
                early_stop_flag = 1
                test_metric_ec, test_metric_e, test_metric_c = test_ec, test_e, test_c
                print('epoch',epoch,'test  p,r,f1 is :',test_ec)
            else:
                early_stop_flag += 1

        if epoch > configs.epochs / 2 and early_stop_flag >= 5:
            break
        print('epoch',epoch,'is finish')
    return test_metric_ec, test_metric_e, test_metric_c

if __name__ == '__main__':
    configs = Config()
    test_metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    test_metric_ec, test_metric_e, test_metric_c = main(configs, 1)
    print('Test F_ecp: {}'.format(test_metric_ec)) 
      
    print('===== Test Data Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(test_metric_ec[2]), float_n(test_metric_ec[0]), float_n(test_metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(test_metric_e[2]), float_n(test_metric_e[0]), float_n(test_metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(test_metric_c[2]), float_n(test_metric_c[0]), float_n(test_metric_c[1])))
    write_b({'test_ecp': test_metric_ec, 'test_emo': test_metric_e, 'test_cau': test_metric_c}, 'results/{}_metrics.pkl'.format(time.time()))