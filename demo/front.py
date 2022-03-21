import json
import time
import streamlit as st
import requests

st.set_page_config(layout="wide")

def main():

    st.title("두 문장의 의미 유사도(STS) 측정")

    text1 = st.text_input("첫 번째 문장 입력:")
    text2 = st.text_input("두 번재 문장 입력:")

    data = json.dumps({'s1': text1, 's2': text2})

    if text1 and text2:
        
        st.markdown("## 의미 유사도 측정 결과")

        with st.spinner('processing..'):
            start = time.time()
            response = requests.post("http://localhost:8001/", data=data)
            label = response.json()
            end = time.time()

        st.write(label)
        st.write(f'추론 요청부터 추론 완료까지 {end-start} 초가 걸립니다.')

main()