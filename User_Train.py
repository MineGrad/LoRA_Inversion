#Compute gradient in the adapter layers
GLOBAL_VAR=0
import copy
import torch
from torch import optim
import torch.nn as nn
device=torch.device("cpu")
def train(target_encoder, sample, model, input_idx, labels):

    y=labels[sample]
    y=torch.tensor(y).reshape(1)

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.SGD(model.parameters(), lr=0.001)


    for name, param in model.named_parameters():
        if 'A' in name or 'B' in name or 'mlphead.head' in name:   # Replace 'layer_name' with the specific layers you want gradients for
            param.requires_grad = True
        else:
            param.requires_grad = False
    iter=1

#data=input_ids[sample].reshape(batch_size,1,3,32,32)
    data=input_idx[sample]#.unsqueeze(0)
    print(data.shape)
 #y=label.reshape(1)
    for i in range(iter):

        model.train()
        model.zero_grad()
        output=model(data)
        loss=criterion(output.to(device),y.to(device))
    #optimizer.zero_grad()
        loss.backward()
    weight_grad=[]
    for t in target_encoder:
        for name, param in model.named_parameters():     
            if name==f'encoder{t}.attn.B.weight':
                weight_grad.append(param.grad)
    return weight_grad
    
def train_secagg( sample, model, input_idx, labels):

    y=labels[sample]
    y=torch.tensor(y).reshape(1)

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.SGD(model.parameters(), lr=0.001)


    for name, param in model.named_parameters():
        if 'A' in name or 'B' in name or 'mlphead.head' in name:   # Replace 'layer_name' with the specific layers you want gradients for
            param.requires_grad = True
        else:
            param.requires_grad = False
    iter=1

#data=input_ids[sample].reshape(batch_size,1,3,32,32)
    data=input_idx[sample]#.unsqueeze(0)
    print(data.shape)
 #y=label.reshape(1)
    for i in range(iter):

        model.train()
        model.zero_grad()
        output=model(data)
        loss=criterion(output.to(device),y.to(device))
    #optimizer.zero_grad()
        loss.backward()
    weight_grad={}
    for name, param in model.named_parameters():
        if 'attn.B.weight' in name:
            weight_grad[name]=param.grad
    return weight_grad