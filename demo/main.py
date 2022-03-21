import os
from pathlib import Path
from re import S
from typing import Dict, Sequence

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from model import RobertaForStsRegression
from dataset import KlueStsWithSentenceMaskDataset

app = FastAPI()

class Data(BaseModel):
    s1: str
    s2: str


def get_model():
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 학습한 모델 로드
    ckpt = torch.load(os.path.join(Path(__file__).parent, 'pytorch_model.bin'), map_location=device)

    # 토크나이저 및 설정 로드
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    config = AutoConfig.from_pretrained('klue/roberta-base')
    config.num_labels = 1

    # 모델 로드
    model = RobertaForStsRegression(config=config)
    model.load_state_dict(ckpt)
    model.eval()

    return model, tokenizer, device


model, tokenizer, device = get_model()


@app.post('/') # app.HTTP_Method('uri')
def sts_classfier(data:Data):
    d = []
    s1 = data.s1.strip()
    s2 = data.s2.strip()
    d = [{"guid":0, "sentence1":s1, "sentence2":s2, "real-label":0, "binary-label":0}]

    inputs = KlueStsWithSentenceMaskDataset(d, tokenizer, 510)
    data_loader = DataLoader(inputs, 1, drop_last=False)
    for batch in data_loader:
        input_data = {
            key: value.to(device) for key, value in batch.items() if not key == "labels"
        }
        
        with torch.no_grad():
            output = model(**input_data)[0]

        pred = output.detach().cpu().numpy()
        real_label = pred[0][0]

        bin_label = 0
        if real_label > 3.2:
            bin_label = 1

    return {'Senetence1':s1, 'Senetence2': s2, 'Real-Label':str(real_label), 'Binary-Label':str(bin_label)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="0.0.0.0", port=8001, reload=True)
