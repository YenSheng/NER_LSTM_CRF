from gensim.models import Word2Vec
import argparse


## hyperparameters
parser = argparse.ArgumentParser(description='word2vec training')
parser.add_argument('--train_data', type=str, default='data_path/train_data', help='train data source')
parser.add_argument('--output', type=str, default='embedding/word2vec.txt', help='output path')
parser.add_argument('--dim', type=int, default=300, help='dim of word')
args = parser.parse_args()


with open(args.train_data, 'r', encoding='utf8') as fr: # 使用train_data做預訓練
    lines = fr.readlines()

character_list = []
sentences = []
for line in lines:
    if line != '\n':
        character = line.split(' ')[0]
        character_list.append(character)
    else:
        sentences.append(character_list)
        character_list = []

        
model = Word2Vec(sentences, min_count=1 , size=args.dim)
model.wv.save_word2vec_format(args.output ,binary = False)
