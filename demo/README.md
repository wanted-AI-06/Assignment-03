# “초”경량 재활용쓰레기 이미지 분류기

[KLUE STS dev](https://github.com/wanted-AI-06/Assignment-03)의 Best model을 이용한 프로토타이핑

## 실행 방법
### 가상환경 설정
```
conda create -n sts python=3.8
conda activate sts
pip install -r requirements.txt
```
- sts 대신 다른 이름으로 대체 가능


### FastAPI, Streamlit 실행

윈도우 
```
# FastAPI 백그라운드 실행
start /b python main.py

# Streamlit 포그라운드 실행
streamlit run front.py
```
http://localhost:8501/에서 streamlit으로 데모 실행 가능


## 시연 영상
![sts_demo.gif](./sts_demo.gif)


## Reference
`dataset.py` and `model.py` in this repo is based on [LostCow/KLUE](https://github.com/LostCow/KLUE)
