# -*- coding: utf-8 -*-


import torch
from torch import nn
import math

VOCAB_SIZE = 50265  # Standard RoBERTa vocab size


class Create_WordEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, dropout):
        super().__init__()

        # Word embeddings
        #self.cls_token = nn.Parameter(torch.randn(size=(1,embedding_dim)), requires_grad=True)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim), requires_grad=False)
        self.token_type_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim), requires_grad=False)
        self.LN = nn.LayerNorm(embedding_dim, eps=1e-6)
        # Dropout for embedding
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        word_embeds = self.word_embeddings(x)
        global x_before_word_embed
        x_before_word_embed=word_embeds
        seq_len = word_embeds.size(1)#+1
        pos_embeds = self.position_embeddings[:, :seq_len, :]
        token_type_embeds = self.token_type_embedding[:, :seq_len, :]

        x = word_embeds + pos_embeds + token_type_embeds
        x = self.LN(x)

        return x



class Residual(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

    def forward(self, x, y):
        return x + y

class Attention(nn.Module):
    def __init__(self, max_seq_len, num_head, embedding_dim, dim_head, r):
        super().__init__()

        self.QKV = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.scale = math.sqrt(dim_head)
        self.msa = nn.Linear(embedding_dim, embedding_dim, bias=False)


        self.NUM_HEADS = num_head
        self.HEAD_DIM = dim_head
        self.Bq=nn.Linear(r,embedding_dim, bias=False)
        self.Aq=nn.Linear(embedding_dim,r, bias=False)
        self.Bk=nn.Linear(r,embedding_dim, bias=False)
        self.Ak=nn.Linear(embedding_dim,r, bias=False)
        self.Bv=nn.Linear(r,embedding_dim, bias=False)
        self.Av=nn.Linear(embedding_dim,r, bias=False)
        self.B=nn.Linear(r,embedding_dim, bias=False)
        self.A=nn.Linear(embedding_dim,r, bias=False)
        self.res=Residual(embedding_dim)
        self.LN1 = nn.LayerNorm(embedding_dim, eps=1e-6)
    def forward(self, y, i, mask=None):

        global y_before_attn_LN
        y_before_attn_LN=y


        global y_after_res

        QKV = self.QKV(y).reshape(y.shape[0], y.shape[1], 3, self.NUM_HEADS, self.HEAD_DIM)
        QKV = QKV.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, seq_len, dim_head)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        combined_weights_query = self.Bq.weight @ self.Aq.weight
        delQ = torch.matmul(y, combined_weights_query).reshape(y.shape[0], y.shape[1], self.NUM_HEADS, self.HEAD_DIM)
        delQ = delQ.permute(0, 2, 1, 3)  # Shape: (1, batch_size, num_heads, seq_len, dim_head)
        combined_weights_key = self.Bk.weight @ self.Ak.weight
        delK = torch.matmul(y, combined_weights_key).reshape(y.shape[0], y.shape[1], self.NUM_HEADS, self.HEAD_DIM)
        delK = delK.permute(0, 2, 1, 3)
        combined_weights_value = self.Bv.weight @ self.Av.weight
        delV = torch.matmul(y, combined_weights_value).reshape(y.shape[0], y.shape[1], self.NUM_HEADS, self.HEAD_DIM)
        delV = delV.permute(0, 2, 1, 3)
        Q = Q + delQ
        K = K + delK
        V = V + delV

        dot_product = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(dot_product, dim=-1)

        weighted_values = torch.matmul(attention_weights, V)

        weighted_values = weighted_values.permute(0, 2, 1, 3).reshape(y.shape[0], -1, self.NUM_HEADS * self.HEAD_DIM)

        combined_weights = self.B.weight @ self.A.weight

        MSA_output=self.msa(weighted_values) + torch.matmul(weighted_values, combined_weights) #combined_weights(concat_heads) #((self.B(self.A))(concat_heads))


        MSA_output=self.res(MSA_output,y)
        y_after_res= MSA_output

        MSA_output=self.LN1(MSA_output)

        return MSA_output




class MLP(nn.Module):
    def __init__(self, embedding_dim, r):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu=nn.GELU()

        self.fc2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.res=Residual(embedding_dim)
        self.LN2 = nn.LayerNorm(embedding_dim, eps=1e-6)


    def forward(self, x, i):




        global y_after_mlp_LN
        global y_after_mlp_res
        global y_before_mlp
        y_before_mlp=x

        y = self.fc1(x)
        y = self.gelu(y)
        y = self.fc2(y)
        x = self.res(x,y)
        y_after_mlp_res=x

        x = self.LN2(x)



        y_after_mlp_LN=x

        return x

class MLPHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)



    def forward(self, x):
        cls_token_output = x  # Use the first token for classification (similar to CLS token)

        cls_token_output = self.dense(cls_token_output)

        cls_output = self.head(cls_token_output)

        return cls_output

class Encoder(nn.Module):
    def __init__(self, max_seq_len, num_head, embedding_dim, dim_head, r):
        super().__init__()
        self.attn = Attention(max_seq_len, num_head, embedding_dim, dim_head, r)
        self.mlp = MLP(embedding_dim, r)


    def forward(self, x, i, mask=None):

        global y_before_attn
        y_before_attn=x

        y = self.attn(x, i, mask)

        global y_after_attn
        y_after_attn=y




        y = self.mlp(y,i)

        global y_after_mlp
        y_after_mlp=y



        global y_after_mlpres
        y_after_mlpres=x

        return y

class RoBERTa(nn.Module):
    def __init__(self, r, embedding_dim, vocab_size, max_seq_len, num_head, dim_head, dropout, num_classes):
        super().__init__()
        self.embedding = Create_WordEmbedding(embedding_dim, vocab_size, max_seq_len, dropout)

        self.mlphead = MLPHead(embedding_dim, num_classes)
        self.num_blocks = 12
        for i in range(1, self.num_blocks + 1):
            setattr(self, f'encoder{i}', Encoder(max_seq_len, num_head, embedding_dim, dim_head, r))



    def forward(self, x, mask=None):
        x = self.embedding(x)

        global y_patch
        global yp
        y_patch=x

        for i in range(1, self.num_blocks + 1):
            x = getattr(self, f'encoder{i}')(x, i-1, mask)


            yp=x


        x = self.mlphead(x[:,0])

        return x