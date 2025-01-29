import torch
from torch import nn
import numpy as np
import random
import math
import copy



class Attention_Layer:
    def __init__(self, embedding_dim,patch_dim,num_heads):
        self.EMBED_DIM=embedding_dim
        self.PATCH_DIM=patch_dim
        self.NUM_HEADS=num_heads
        self.HEAD_DIM=patch_dim//num_heads
       
    def attack_parameter(self, w):
        
#print(Q.shape)
        Q_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        K_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        V_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        #print(Q_head[1*PATCH_VECTOR_LEN:(1+1)*PATCH_VECTOR_LEN,1*PATCH_VECTOR_LEN:(1+1)*PATCH_VECTOR_LEN].shape)

        for i in range(self.NUM_HEADS):    
                Q_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=10**5*torch.eye(self.HEAD_DIM)
                K_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=torch.eye(self.HEAD_DIM)
                V_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=1*torch.eye(self.HEAD_DIM)

        w[0:self.EMBED_DIM]=Q_head
        w[self.EMBED_DIM:2*self.EMBED_DIM]=K_head
        w[2*self.EMBED_DIM:3*self.EMBED_DIM]=V_head
        return w




class Pos_Encoding:
    def __init__(self, c, c2, num_heads):
        self.c=c
        self.c2=c2
        self.num_heads=num_heads
        
    def tampering(self,w, head_dim):
        w['embedding.position_embeddings']=torch.zeros(w['embedding.position_embeddings'].size())
        #if w['embedding.position_embeddings'].shape[1]<=32:
        for i in range(0,w['embedding.position_embeddings'].shape[1]):
            if i<31:
                w['embedding.position_embeddings'][0][i][2*i]=self.c
                w['embedding.position_embeddings'][0][i][2*i+1]=-self.c
            else:
            
                w['embedding.position_embeddings'][0][i][2*31]=self.c
                w['embedding.position_embeddings'][0][i][2*31+1]=-self.c
                    
     


        for h in range(1, self.num_heads):
            for i in range(0,w['embedding.position_embeddings'].shape[1]):
                if i<31:
                    w['embedding.position_embeddings'][0][i][h*head_dim+2*i]=self.c2
                    w['embedding.position_embeddings'][0][i][h*head_dim+2*i+1]=-self.c2
                else:
                    w['embedding.position_embeddings'][0][i][h*head_dim+2*31]=self.c2
                    w['embedding.position_embeddings'][0][i][h*head_dim+2*31+1]=-self.c2
        return w['embedding.position_embeddings']
    
class Design:
    def __init__(self,r, c, c2, embedding_dim, patch_dim, num_heads, num_patch):
        super().__init__()
        self.attn_layer=Attention_Layer(embedding_dim, patch_dim,num_heads)
        self.pos_enc=Pos_Encoding(c,c2,num_heads)
        
    def position_encoding(self,w,head_dim):
        w=self.pos_enc.tampering(w,head_dim)
        return w
    def attention(self, w):
        w=self.attn_layer.attack_parameter(w)
        return w
    
    
    
   
    
   
    