import json

#train 데이터셋을 train,valid 데이터셋으로 split

def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)

data=read_json('/content/drive/MyDrive/Assignment_03/data/klue-sts-v1.1_train.json')

train_length=int(len(data)*0.9)
train=data[:train_length]
vaild=data[train_length:]

print('data_set:',len(data))
print('train_set:',len(train),', valid_set:',len(vaild))

with open('train_split.json', 'w') as f:
  json.dump(train, f, ensure_ascii=False)
with open('valid_split.json','w') as f:
  json.dump(vaild, f, ensure_ascii=False)