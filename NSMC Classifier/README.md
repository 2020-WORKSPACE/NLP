# DNN 기반 NSMC 분류기 설명

1. nlp_kr_npy.zip 파일을 압축해제 하시면 train, test에 해당하는 데이터를 Bag of Words로 문장을 임베딩한 데이터들을 얻을 수 있습니다. 

2. GPU를 사용하실 경우, 메모리를 꽤 많이 차지하므로, colab을 사용하시면 됩니다.

3. main.py를 실행하시면 학습이 완료되고, DNN_model로 모델 저장 및 evaluation을 진행합니다.

4. kaggle_test.py를 실행하기 위해선 ko_data.csv를 미리 다운받아 놓아야 합니다. kaggle_test.py를 실행하시면, output.txt가 생성됩니다. 여기서 각 문장마다 분류한 결과가 나옵니다.

5. 문장 분류를 처음 해보는 것이라 어려운 점이 많았습니다. 아래는 참고한 링크들입니다.

## 프로그래밍에 도움을 많이 받은 사이트들

1. NSMC 분류를 하는 관점을 설명한 사이트<br>
https://wikidocs.net/44249

2. DNN으로도 문장 분류를 할 수 있음 및 Bag of Word 방식의 문장 인코딩을 설명한 사이트<br> 
https://medium.com/oracledevs/text-classification-with-deep-neural-network-in-tensorflow-simple-explanation-be07c6cbe867

3. Pytorch 문법<br>
https://wikidocs.net/book/2788



* 추신 : Kaggle을 이용해서 Data Science 능력을 경쟁해볼 수 있다는 것이 재밌었습니다.<br>
방학동안 Kaggle을 이용해서 여러가지 머신러닝 기법들을 공부하겠습니다.
