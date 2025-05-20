import torch
import torch.nn as nn
import pandas as pd
import numpy as np
#read data
raw_df=pd.read_csv('train.csv')
raw_pf=np.array(raw_df)

#label
label=raw_df['label']

#feature
raw_df=raw_df.drop(['label'],axis=1)
feature=raw_df.values

#contribute to test and train
train_feature=feature[:int(len(feature)*0.8)]
train_label=label[:int(len(feature)*0.8)]
test_feature=feature[int(len(feature)*0.8):]
test_label=label[int(len(feature)*0.8):]

#turn into tensor
train_featrue=torch.tensor(train_feature).to(torch.float)
train_label=torch.tensor(train_label)
test_featrue=torch.tensor(test_feature).to(torch.float)
test_label=torch.tensor(test_label)

#consttruct model
model=nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444,512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
)

#loss function
lossfunction=nn.CrossEntropyLoss()

#opttimizer
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.0001)

#train
for i in range(100):
    optimizer.zero_grad()
    predict=model(train_featrue)
    loss=lossfunction(predict,train_label)
    loss.backward()
    optimizer.step()
    print('epoch:',i,'loss:',loss.item())