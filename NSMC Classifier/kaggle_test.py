#########################################################
# 여기는 Kaggle 위한 테스트 용 코드입니다.
#########################################################

from preprocessing import *
from DNN_model import *
import pandas as pd

import os

# GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
print("GPU Available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#######################################
# 모델을 evaluation mode로 설정
# model은 main.py에서 학습을 마친 후 얻은 모델을 사용
#######################################
model = torch.load('DNN_model')
model.eval()

# 한글이 깨지는 문제 때문에, encoding 방식을 변경
kaggle_data = pd.read_csv('ko_data.csv', encoding='CP949').values

#id 제외, 텍스트 부분만 추출
kaggle_data = kaggle_data[:, 1]

#######################################
# x_train.txt에서 얻은 토큰 및 태깅 정보가 필요하므로
# 여기서 필요한 코드를 재사용
#######################################

# train nsmc에서 docs를 가져온다.
train_docs, test_docs = preprocessing()
tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

# train_txt에서 선택된 상위 5000개의 단어들
selected_words = [f[0] for f in text.vocab().most_common(5000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


#######################################
# kaggle_data를 하나씩 뽑고, 각 문장을 토큰화해서 
# selected_words를 이용해서 크기 5000의 벡터로 만든 다음, evaluate
#
# output.txt에 예측한 결과를 담는다.
# 이후, 엑셀 프로그램을 통해, 위 output.txt에 저장된 데이터로
# csv 파일을 만든 후 Kaggle에 제출
# Kaggle Score 83.05
#######################################
okt = Okt()

with open('output.txt','w') as out:
    for x_data in kaggle_data:
        x_data = term_frequency((tokenize(x_data, okt)))
        x_data = torch.tensor(x_data).to(device)
        pred = model.forward(x_data.float())

        if pred < 0.5:
            out.write('0\n')
        else:
            out.write('1\n')
