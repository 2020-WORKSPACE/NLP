from preprocessing import *
from DNN_model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import os
import time

################################################
# GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
################################################
print("GPU Available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

x_train = None
y_train = None 
x_test = None
y_test = None

#######################################
# x_train.txt, y_train.txt, x_test.txt, y_test.txt에서
# Bag of Words 형태로 임베딩 벡터를 얻을 수 있지만
# nlp_kr_npy.zip 파일을 압축해제하면 각 데이터에 대한
# npy 파일들을 얻을 수 있습니다.
#######################################
if os.path.isfile("x_train.npy") and os.path.isfile("y_train.npy") and os.path.isfile("x_test.npy") and os.path.isfile("y_test.npy"): # 이미 과거에 일련의 파일을 생성해 둠. 불완전한 if문
    
    print("load from saved files...\n")

    x_train = np.load('x_train.npy')
    x_train = torch.from_numpy(x_train).to(device)
    print("1. finished reading x_train")

    y_train = np.load('y_train.npy')
    y_train = torch.from_numpy(y_train).view(-1, 1).to(device)
    print("2. finished reading y_train")
    
    x_test = np.load('x_test.npy')
    x_test = torch.from_numpy(x_test).to(device)
    print("3. finished reading x_test")
    
    y_test = np.load('y_test.npy')
    y_test = torch.from_numpy(y_test).view(-1, 1).to(device)
    print("4. finished reading y_test")

else:
    print("load from nsmc files...\n")
    x_train, y_train, x_test, y_test = load_data_nsmc()
    print("finished loading from nsmc files")

# data 개수 확인
print('The number of training data: ', len(y_train))
print('The number of test data: ', len(y_test))

# 학습 모델 생성
model = DNN_model().to(device)

#######################################
# loss 함수 생성
# 0과 1로 분류하는 것이 목표이기 때문에,
# loss 함수는 Binary Cross Entropy Loss를 사용하겠습니다.
#######################################
criterion = nn.BCELoss()

#######################################
# optimizer 생성
# optimizer는 단순하게 SGD 방식을 택하겠습니다.
# learning rate : 0.001
# momentum      : 0.9
# weight decay  : 1e-04
#######################################
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum= 0.9, weight_decay=1e-04)

#######################################
# learning rate decay
# 160000번째, 320000번째 iteration에서 learning rate decay 동작
#######################################
decay_epoch = [160000, 320000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)

#######################################
# start training
#######################################
print("-----start training-----")

start_time = time.time()
global_steps = 0
epoches = 30

train_loss = 0
train_batch_cnt = 0
for epoch in range(0, epoches):
    for x_data, y_data in zip(x_train, y_train):

        optimizer.zero_grad()
        outputs = model(x_data)

        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_batch_cnt += 1
        
        if train_batch_cnt % 3000 == 1:
            ave_loss = train_loss / train_batch_cnt
            training_time = (time.time() - start_time) / 60
            print('========================================')
            print("epoch:", epoch, "\ttrain_iteration:", train_batch_cnt)
            print("training dataset average loss: %.3f" % ave_loss)
            print("training_time: %.2f minutes" % training_time)
        
        #######################################
        # epoch가 3의 배수가 될 때마다 model을 저장
        # 구글 colab을 사용하면, 중간에 연결이 끊기는 문제가 발생합니다.
        # 그래서 보험을 위해 모델을 중간 중간 저장하도록 합니다.
        # validation test를 하면서 저장하는 것이 좀 더 올바르지만,
        # RAM 부족 문제가 생길 수 있어 단순하게 접근하였습니다.
        #######################################
        if epoch % 3 == 0 and train_batch_cnt == len(y_train):
            torch.save(model, 'DNN_model_'+str(epoch))

        step_lr_scheduler.step()

#######################################
# iteration이 끝난 후, 마지막 모델 저장
#######################################
print('finished training model --- now saving')
torch.save(model, 'DNN_model')

# 모델을 evaluation mode로 설정
model.eval()

print("-----start evaluating-----")
test_correct_cnt = 0
test_entire_cnt = 0
for x_data, y_data in zip(x_test, y_test):

    x_data = x_data.to(device)
    y_data = y_data.to(device)

    pred = model.forward(x_data)
    pred = 0 if pred < 0.5 else 1
    test_correct_cnt = test_correct_cnt + (1 if pred == y_data else 0)
    test_entire_cnt += 1.0

print('========================================')
print("test accyracy: %.2f percent" % (100 * test_correct_cnt / test_entire_cnt))