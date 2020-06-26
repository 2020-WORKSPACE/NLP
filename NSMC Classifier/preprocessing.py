import json
import os

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from konlpy.tag import Okt

def tokenize(doc, okt):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)
            if t[1] in ['Noun', 'Verb', 'Adjective', 'Adverb', 'VerbPrefix', 'KoreanParticle']]

def read_data(file):
    with open(file, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()][1:]
    return data

def preprocessing():    

    okt = Okt()
    
    train_docs = None

    # train_data tagging
    if os.path.isfile('train_docs.json'):
        with open('train_docs.json') as f_read:
            train_docs = json.load(f_read)
    else:
        train_data = read_data('nsmc/ratings_train.txt')
        train_docs = [(tokenize(row[1], okt), row[2]) for row in train_data]
        with open('train_docs.json', 'w', encoding='utf-8') as f_write:
            json.dump(train_docs, f_write, ensure_ascii=False, indent='\t')

    test_docs = None

    # test_data tagging
    if os.path.isfile('test_docs.json'):
        with open('test_docs.json') as f_read:
            test_docs = json.load(f_read)
    else:
        test_data = read_data('nsmc/ratings_test.txt')
        test_docs = [(tokenize(row[1], okt), row[2]) for row in test_data]
        with open('test_docs.json', 'w', encoding='utf-8') as f_write:
            json.dump(test_docs, f_write, ensure_ascii=False, indent='\t')

    return train_docs, test_docs

def load_data_nsmc():
    
    train_docs, test_docs = preprocessing()
    
    tokens = [t for d in train_docs for t in d[0]]
    text = nltk.Text(tokens, name='NMSC')

    #############################################################
    # 자주 사용하는 토큰 5000개를 사용해서 데이터를 벡터로 표현
    # 즉, 1개의 문장은 크기 5000의 벡터
    # BOW (Bag of Words) 방식으로 구현하며, CountVectorization을 사용한다.
    #############################################################
    selected_words = [f[0] for f in text.vocab().most_common(5000)]

    def term_frequency(doc):
        return [doc.count(word) for word in selected_words]

    float_x_train = None
    float_x_test = None
    float_y_train = None
    float_y_test = None

    # float_x_train
    if os.path.isfile('x_train.npy'):
        float_x_train = np.load('x_train.npy')
    else:
        x_train = [term_frequency(d) for d, _ in train_docs]
        float_x_train = np.asarray(x_train).astype('float32')
        np.save('x_train.npy', float_x_train)

    # float_x_test
    if os.path.isfile('x_test.npy'):
        float_x_test = np.load('x_test.npy')
    else:
        x_test = [term_frequency(d) for d, _ in test_docs]
        float_x_test = np.asarray(x_test).astype('float32')
        np.save('x_test.npy', float_x_test)

    # float_y_train
    if os.path.isfile('y_train.npy'):
        float_y_train = np.load('y_train.npy')
    else:
        y_train = [c for _, c in train_docs]
        float_y_train = np.asarray(y_train).astype('float32')
        np.save('y_train.npy', float_y_train)

    # float_y_test
    if os.path.isfile('y_test.npy'):
        float_y_test = np.load('y_test.npy')
    else:
        y_test = [c for _, c in test_docs]
        float_y_test = np.asarray(y_test).astype('float32')
        np.save('y_test.npy', float_y_test)

    float_x_train = torch.from_numpy(float_x_train)
    float_y_train = torch.from_numpy(float_y_train)
    float_x_test = torch.from_numpy(float_x_test)
    float_y_test = torch.from_numpy(float_y_test)
    return float_x_train, float_y_train, float_x_test, float_y_test