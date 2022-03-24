# Assignment-03

**※ 과제의 출력물이 깃허브 상에서 보이지 않습니다. 파일을 다운로드하면 출력결과를 확인하실 수 있습니다.**

## NLU - 문장 유사도 계산 (STS)
과제 목표
- 한국어 문장의 유사도 분석 모델 훈련 및 서비스화
- 두 개의 한국어 분장을 입력받아 두 문장의 의미적 유사도를 출력하는 모델 생성


## 과제 산출물 설명

**기업과제3_6팀.ipynb**
- KLUE STS data를 이용한 모델 학습 및 검증
- 데이터 전처리를 통한 모델 성능 비교 및 최종 모델 선정

**dev_set_score.csv** 
- KLUE STS 데이터셋 컬렘에 predict real label, predict binary labe 추가한 csv 파일
```
  - predict real label : 모델이 추론한 실수 레이블 [float]
  - predict binary label : 모델이 추론한 True(1)/False(0) 레이블 [int]
```

**기업과제3_6팀_dev_set_score.ipynb**
- dev_set_score.csv 생성 과정 출력  
- dev_set_score의 예측값과 실제값 F1 score 와 Pearson's R 결과물 출력

**code** 
- KLUE/RoBERTa Base STS Regression model 코드 

**demo** 
- 모델 프로토타이핑을 FastAPI, Streamlit로 구현 
- [FastAPI, Streamlit 코드](https://github.com/wanted-AI-06/Assignment-03/tree/main/demo)


## 모델 성능

최고 성능 모델 요약
- [최고 성능 모델 다운로드](https://drive.google.com/file/d/1Y9GFVzcmTH0Zas_ekt0PNz4xToqpvBnj/view?usp=sharing)

|model| pearsonr | F1-score |
|---|-------|--------:|
|roberta-base| 0.894| 0.859 |  
|custom| 0.902| 0.870 |
