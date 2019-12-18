import tensorflow as tf
import numpy as np
import os, argparse, time, random
from data import read_corpus, read_pretrain_embedding, random_embedding, vocab_build
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只顯示 warning跟error (default: 0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='', help='model for test and demo')
args = parser.parse_args()


## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "B-TIM": 7, "I-TIM": 8,
            }


## get char embeddings
if args.pretrain_embedding == 'random': # 使用隨機向量
    vocab_path = os.path.join('.', args.train_data, 'word2id.pkl')
    corpus_path = os.path.join('.', args.train_data, 'train_data')
    word2id = vocab_build(vocab_path, corpus_path, min_count=1) # 將每個字給定id
    embeddings = random_embedding(word2id, args.embedding_dim) # 使用隨機向量
else:
    vocab_path = os.path.join('.', args.train_data, 'word2id.pkl')
    embedding_path = os.path.join('.', 'embedding', args.pretrain_embedding) 
    word2id, embeddings = read_pretrain_embedding(vocab_path, embedding_path) # 將每個字給定id，使用預訓練向量


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path); test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model # timestamp:當前時間(1970紀元後經過的浮點秒數)
output_path = os.path.join('.', args.train_data+"_save", timestamp) # 預設 './data_path_save/timestamp'
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries") # 預設 './data_path_save/timestamp/summaries'
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/") # 預設 './data_path_save/timestamp/checkpoints/'
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model") # 預設 './data_path_save/timestamp/checkpoints/model'
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results") # 預設 './data_path_save/timestamp/results'
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt") # 預設 './data_path_save/timestamp/results/log.txt'
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config) # 建立model
    model.build_graph() # 建立graph
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # 模型訓練

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config) # 建立model
    model.build_graph() # 建立graph
    print("test data: {}".format(test_size))
    model.test(test_data) # 模型測試

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph() # 建立graph
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG, TIM = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}\nTIM: {}'.format(PER, LOC, ORG, TIM))
