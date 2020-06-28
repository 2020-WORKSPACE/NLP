import torch
from random import shuffle
from collections import Counter
import argparse
import random
import nltk
import math
import numpy
import operator
import csv
import json
import pandas
import re


def cleanText(readData):
    #텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?:;^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text



def NN(target, activated, inputMatrix, outputMatrix):
################################  Input  ################################
#     target   : Index of a class (type:int)                            #
#   activated  : Index of a title, description (type:list of int)       #
#  inputMatrix : Weight matrix of input  (type:torch.tesnor(V,300))     #
# outputMatrix : Weight matrix of output (type:torch.tesnor(8,300))     #
#########################################################################

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector  (type:torch.tensor(V,300))        #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(8,300))        #
#########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    # input이 총 n개이기 때문에 n개의 one hot vector를 만든다
    input = torch.zeros(len(activated), len(inputMatrix[:,0]))

    for j in range(0, len(activated)):
        for k in range(len(activated)):
            input[j][activated[k]] += 1/len(activated)

    input2 = torch.zeros(1, len(inputMatrix[:,0]))

    for j in range(0, len(activated)):
        input2 += input[j][:]



    # hidden layer 계산 (1,V) * (V,300) = (1,300)
    h = torch.mm(input2, inputMatrix)

    # (1,300) * (300,8) = (1,8)
    c = torch.mm(h, outputMatrix.t())

    #(1,8)의 결과물을 확률로 표현
    e = torch.exp(c)
    # 마지막 확률을 나타내는 p로
    p = e / torch.sum(e, dim=1, keepdim=True)

    # Loss 계산은 크로스엔트로피; target index는 line[0]값에서 1을 뺀 값 (0 ~ 7)
    loss = -torch.log(p[0][target])


    # gradient 계산
    # 어차피 index값 제외하면 0임
    softmax_gra = p
    softmax_gra[0][target] -= 1

    grad_out = torch.mm(softmax_gra.t(), h)
    grad_emb = torch.mm(softmax_gra, outputMatrix)

    return loss, grad_emb, grad_out


def testNN(activated, inputMatrix, outputMatrix):
################################  Input  ################################
#     target   : Index of a class (type:int)                            #
#   activated  : Index of a title, description (type:list of int)       #
#  inputMatrix : Weight matrix of input (type:torch.tesnor(V,300))      #
# outputMatrix : Weight matrix of output (type:torch.tesnor(8,300))     #
#########################################################################

###############################  Output  ################################
# predict index                                                         #
#########################################################################


    # input이 총 n개이기 때문에 n개의 one hot vector를 만든다
    input = torch.zeros(len(activated), len(inputMatrix[:,0]))

    for j in range(0, len(activated)):
        for k in range(len(activated)):
            input[j][activated[k]] += 1/len(activated)

    input2 = torch.zeros(1, len(inputMatrix[:,0]))

    for j in range(0, len(activated)):
        input2 += input[j][:]



    # hidden layer 계산 (1,V) * (V,300) = (1,300)
    h = torch.mm(input2, inputMatrix)

    # (1,300) * (300,8) = (1,8)
    c = torch.mm(h, outputMatrix.t())

    #(1,8)의 결과물을 확률로 표현
    e = torch.exp(c)
    # 마지막 확률을 나타내는 p로
    p = e / torch.sum(e, dim=1, keepdim=True)

    # argmax한 이후 1이 되는 인덱스 값 구하기(모델이 정답으로 예측한 클래스)
    values, indices = p[0].max(dim=0)
    indices = int(indices)



    predict_index = indices
    # answer_index는 0 ~ 7의 값으로 반환됨
    return predict_index




def classification(data, Bigram_dict, dimension=300, learning_rate=0.025):
    print("classification 시작")
    # Xavier initialization of weight matrices
    W_emb = torch.randn(len(Bigram_dict), dimension) / (dimension**0.5)
    # output은 (1 8) 행렬이 되도록
    W_out = torch.randn(8, dimension) / (dimension**0.5)


    losses=[]
    order = 0
    for arr1 in data:
        for arr2 in arr1:


            # 특수문자 제거
            arr2['utterance'] = cleanText(arr2['utterance']).lower()

            # tokenize
            seq = nltk.word_tokenize(arr2['utterance'])


            # target index
            if arr2['emotion'] == "neutral":
                target = 0
            elif arr2['emotion'] == "joy":
                target = 1
            elif arr2['emotion'] == "sadness":
                target = 2
            elif arr2['emotion'] == "fear":
                target = 3
            elif arr2['emotion'] == "anger":
                target = 4
            elif arr2['emotion'] == "surprise":
                target = 5
            elif arr2['emotion'] == "disgust":
                target = 6
            else:
                # non-neutral
                target = 7



            # input vector로 합쳐질 index들
            activated = []

            for i in range(len(seq) - 1):
                key = seq[i] + '-' + seq[i+1]
                if key in Bigram_dict:
                    activated.append(Bigram_dict[key])

            # learning
            #print("NN으로 진입")
            L, G_emb, G_out = NN(target, activated, W_emb, W_out)
            W_emb[activated] -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())



            if order%1000==0 :
        	    avg_loss=sum(losses)/len(losses)
        	    print("Loss : %f" %(avg_loss,))
        	    losses=[]

            order += 1

    print("1st classification done")
    return W_emb, W_out

def classification2(ep, W_emb, W_out, data, Bigram_dict, dimension=300, learning_rate=0.025):
    ep = ep+2
    print("#%d classification Start"%ep)


    losses=[]
    order = 0
    for arr1 in data:
        for arr2 in arr1:


            # 특수문자 제거
            arr2['utterance'] = cleanText(arr2['utterance']).lower()

            # tokenize
            seq = nltk.word_tokenize(arr2['utterance'])


            # target index
            if arr2['emotion'] == "neutral":
                target = 0
            elif arr2['emotion'] == "joy":
                target = 1
            elif arr2['emotion'] == "sadness":
                target = 2
            elif arr2['emotion'] == "fear":
                target = 3
            elif arr2['emotion'] == "anger":
                target = 4
            elif arr2['emotion'] == "surprise":
                target = 5
            elif arr2['emotion'] == "disgust":
                target = 6
            else:
                # non-neutral
                target = 7



            # input vector로 합쳐질 index들
            activated = []

            for i in range(len(seq) - 1):
                key = seq[i] + '-' + seq[i+1]
                if key in Bigram_dict:
                    activated.append(Bigram_dict[key])

            # learning
            #print("NN으로 진입")
            L, G_emb, G_out = NN(target, activated, W_emb, W_out)
            W_emb[activated] -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())



            if order%1000==0 :
        	    avg_loss=sum(losses)/len(losses)
        	    print("Loss : %f" %(avg_loss,))
        	    losses=[]

            order += 1

    print("#%d classification done"%ep)
    return W_emb, W_out



def main():

    with open('/Users/daniel/workspace/python/EmotionLines/Friends/friends_train.json') as json_file:
        data = json.load(json_file)

    with open('/Users/daniel/workspace/python/EmotionLines/Friends/friends_dev.json') as json_file:
        data3 = json.load(json_file)


    # dictionary 미리 선언
    Bigram_dict = {}
    gram_index = 0

    label_dict = {}

    for arr1 in data:
        for arr2 in arr1:

            # 전처리
            # 특수문자 제거
            arr2['utterance'] = cleanText(arr2['utterance']).lower()

            # tokenize
            seq = nltk.word_tokenize(arr2['utterance'])

            # n-gram dictionary 만들기
            for i in range(len(seq) - 1):
                key = seq[i] + '-' + seq[i+1]
                if key not in Bigram_dict:
                    Bigram_dict[key] = gram_index
                    gram_index += 1



    for arr1 in data3:
        for arr2 in arr1:

            # 전처리
            # 특수문자 제거
            arr2['utterance'] = cleanText(arr2['utterance']).lower()

            # tokenize
            seq = nltk.word_tokenize(arr2['utterance'])

            # n-gram dictionary 만들기
            for i in range(len(seq) - 1):
                key = seq[i] + '-' + seq[i+1]
                if key not in Bigram_dict:
                    Bigram_dict[key] = gram_index
                    gram_index += 1

        # Bigram_dict 완성




    print("Bi-gram dictionary 완성")
    print(len(Bigram_dict))
    print(Bigram_dict)
    # training section
    print("train start")
    emb, out = classification(data, Bigram_dict, dimension=300, learning_rate=0.05)
    for epoch in range(10):
        emb, out = classification2(epoch, emb, out, data, Bigram_dict, dimension=300, learning_rate=0.05)


    print("training done")





    # testing section
    with open('/Users/daniel/workspace/python/EmotionLines/Friends/friends_test.json') as json_file:
        data2 = json.load(json_file)
    print("test file read done")



    f = open('output.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['','speaker','utterance','emotion', 'Pos/Neg'])
    wrrow = 1
    whole = 0
    right = 0
    right_pos = 0

    for arr in data2:
        for arr2 in arr:

            arr2['utterance'] = cleanText(arr2['utterance']).lower()

            seq = nltk.word_tokenize(arr2['utterance'])

            # target index
            if arr2['emotion'] == "neutral":
                target = 0
            elif arr2['emotion'] == "joy":
                target = 1
            elif arr2['emotion'] == "sadness":
                target = 2
            elif arr2['emotion'] == "fear":
                target = 3
            elif arr2['emotion'] == "anger":
                target = 4
            elif arr2['emotion'] == "surprise":
                target = 5
            elif arr2['emotion'] == "disgust":
                target = 6
            else:
                # non-neutral
                target = 7

            # input vector로 합쳐질 index들
            activated = []

            for i in range(len(seq) - 1):
                key = seq[i] + '-' + seq[i+1]
                if key in Bigram_dict:
                    activated.append(Bigram_dict[key])


            predict_index = int(testNN(activated, emb, out))

            whole += 1

            if target == predict_index:
                right += 1

            predict = ''
            if predict_index==0:
                predict = 'neutral'
            elif predict_index==1:
                predict = 'joy'
            elif predict_index==2:
                predict = 'sadness'
            elif predict_index==3:
                predict = 'fear'
            elif predict_index==4:
                predict = 'anger'
            elif predict_index==5:
                predict = 'surprise'
            elif predict_index==6:
                predict = 'disgust'
            elif predict_index==7:
                predict = 'non-neutral'

            predict_pos = True
            if (predict == "sadness") or (predict == "fear") or (predict == "anger") or (predict == "disgust"):
                predict_pos = False

            target_pos = True
            if (target == "sadness") or (target == "fear") or (target == "anger") or (target == "disgust"):
                target_pos = False

            if predict_pos == target_pos:
                right_pos += 1

            if predict_pos == True:
                pos_neg = 'Pos'
            else:
                pos_neg = 'Neg'

            data4 = arr2['utterance'] + ": " + predict + "\n"
            wr.writerow([wrrow, arr2['speaker'], arr2['utterance'], predict, pos_neg])
            wrrow += 1



    ratio = right/whole*100
    ratio_pos = right_pos/whole*100


    f.close()

    print("test set 정답률: %d%%"%ratio)
    print("pos/neg 정답률: %d%%"%ratio_pos)


    ############################################################################
    print("test data classification 시작")

    answer = []
    s = open('/Users/daniel/workspace/python/EmotionLines/en_data.csv','r',encoding='unicode-escape')
    rdr = csv.reader(s)
    print(rdr)
    predict = ''
    for line in rdr:
        if line[4] != 'utterance':

            line[4] = cleanText(line[4]).lower()

            seq = nltk.word_tokenize(line[4])

            activated = []

            for i in range(len(seq) - 1):
                key = seq[i] + '-' + seq[i+1]
                if key in Bigram_dict:
                    activated.append(Bigram_dict[key])


            predict_index = int(testNN(activated, emb, out))


            if predict_index==0:
                predict = 'neutral'
            elif predict_index==1:
                predict = 'joy'
            elif predict_index==2:
                predict = 'sadness'
            elif predict_index==3:
                predict = 'fear'
            elif predict_index==4:
                predict = 'anger'
            elif predict_index==5:
                predict = 'surprise'
            elif predict_index==6:
                predict = 'disgust'
            elif predict_index==7:
                predict = 'non-neutral'

            answer.append(predict)
    s.close()
    print(len(answer))
    c = open('/Users/daniel/workspace/python/EmotionLines/en_sample.csv', 'w', newline = '')
    inputlines = 0
    wr = csv.writer(c)
    wr.writerow(['Id', 'Predicted'])
    for index in range(len(answer)):
        wr.writerow([index, answer[index]])

    c.close()

main()
