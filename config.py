import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("running on the CPU")


TORCH_SEED = 1024
TRAIN_FILE = 'fold%s_train.json'
TEST_FILE  = 'fold%s_test.json'
DATA_DIR = '.'

# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = 'utils/bert-base-uncased'
        
        self.feat_dim = 768

        self.gnn_dims = '192'
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 10
        self.lr = 1e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 1
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 1e-5
        self.warmup_proportion = 0.05
        self.adam_epsilon = 1e-8