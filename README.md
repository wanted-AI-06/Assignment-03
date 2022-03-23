# Assignment-03
# NLU - 문장 유사도 계산 (STS)
* <font size=5> 과제 목표</font>
    * <font size=4> 한국어 문장의 유사도 분석 모델 훈련 및 서비스화</font>
    * <font size=4> 두 개의 한국어 분장을 입력받아 두 문장의 의미적 유사도를 출력하는 모델 생성</font>

# 과제 제출물
* <font size=5>기업과제3_6팀.ipynb</font>

    <font size=4>KLUE STS data를 이용한 모델 학습 및 검증</font>  
    <font size=4>데이터 전처리를 통한 모델 성능 비교 및 최종 모델 선정</font>

* <font size=5>dev_set_score.csv</font>

    검증 결과 score  
    예시 {guid, true_real_label, true_binary_label, predict_real_label, predict_binary_label} column 구성

* <font size=5>기업과제3_6팀_dev_set_score.ipynb</font>  

    dev_set_score.csv 생성 과정 출력  
    dev_set_score의 예측값과 실제값 F1 score 와 Pearson's R 결과물 출력

* <font size=5>demo</font>  

    [REST API 서버 코드](https://github.com/wanted-AI-06/Assignment-03/tree/main/demo)


# Summary


|model| pearsonr | F1-score |
|---|-------|--------:|
|roberta-base| 0.894| 0.859 |  
|custom| 0.902| 0.870 |
