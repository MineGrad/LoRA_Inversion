import torch
from torch import nn
import math

VOCAB_SIZE = 30522  # Standard RoBERTa vocab size


class Create_WordEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, dropout):
        super().__init__()

        # Word embeddings
        #self.cls_token = nn.Parameter(torch.randn(size=(1,embedding_dim)), requires_grad=True)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim), requires_grad=False)
        self.token_type_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim), requires_grad=False)
        self.LN = nn.LayerNorm(embedding_dim, eps=1e-6)
        # Dropout for embedding
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Generate position embeddings and word embeddings
        word_embeds = self.word_embeddings(x)
        #print(word_embeds.shape)
        #word_embeds = (word_embeds - word_embeds.mean()) / (word_embeds.std() + 1e-6)  # Normalize to mean 0, std 1
        global x_before_word_embed
        x_before_word_embed=word_embeds
        #print(f"x_before_word_embed : {x_before_word_embed[0]}")
        
        seq_len = word_embeds.size(1)#+1
        
        pos_embeds = self.position_embeddings[:, :seq_len, :]
        token_type_embeds = self.token_type_embedding[:, :seq_len, :]

        x = word_embeds + pos_embeds + token_type_embeds
#         print(f"layer_norm input : {x[0]}")
#         print("std of first LN input:", torch.std(x[0]))
#         print("mean of first LN input:", torch.mean(x[0]))
        x = self.LN(x)
#         print("std of first LN output:", torch.std(x[0]))
#         print("mean of first LN output:", torch.mean(x[0]))
#         print(f"layer_norm output : {x[0]}")
        #x = self.dropout(x)
        # Normalize embeddings to have mean 0.5 and std 0.5


        #x = self.dropout(x)
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
        #self.adapt1 = nn.Linear(embedding_dim, r)
        #self.relu = nn.ReLU()
        #self.adapt2 = nn.Linear(r, embedding_dim)
        self.B=nn.Linear(r,embedding_dim, bias=False)
        self.A=nn.Linear(embedding_dim,r, bias=False)
        self.res=Residual(embedding_dim)
        self.LN1 = nn.LayerNorm(embedding_dim, eps=1e-6)
    def forward(self, y, i, mask=None):

        global y_before_attn_LN
        y_before_attn_LN=y
        #print(f"y_before_attn_qkv{i} : {y_before_attn_LN[0]}")


        global y_after_res

        QKV = self.QKV(y).reshape(y.shape[0], y.shape[1], 3, self.NUM_HEADS, self.HEAD_DIM)
        QKV = QKV.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, seq_len, dim_head)
        Q, K, V = QKV[0], QKV[1], QKV[2]

        # Calculate scaled dot-product attention
        dot_product = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(dot_product, dim=-1)
        #print("softmax")
        # Get weighted values
        weighted_values = torch.matmul(attention_weights, V)
        #print("dot product of values")
        weighted_values = weighted_values.permute(0, 2, 1, 3).reshape(y.shape[0], -1, self.NUM_HEADS * self.HEAD_DIM)
        #print(f"weighted values :{weighted_values.shape}")
        # Output projection and adapter layers
        #print("valued", weighted_values)
#         print("concat_heads")
#         print(concat_heads.shape)
        combined_weights = self.B.weight @ self.A.weight
        #print("combined shape:", combined_weights.shape)
        MSA_output=self.msa(weighted_values) + torch.matmul(weighted_values, combined_weights) #combined_weights(concat_heads) #((self.B(self.A))(concat_heads))
#         print("MSA output")
#         print(MSA_output)
        MSA_output=self.res(MSA_output,y)
        y_after_res= MSA_output
#         print(f"y_after_attn_res{i} : {y_after_res[0]}")
#         print("std of attn LN input:", torch.std(MSA_output[0]))
#         print("mean of attn LN input:", torch.mean(MSA_output[0]))
        MSA_output=self.LN1(MSA_output)
        #print(f"MSA_output after LN{i} : {MSA_output[0]}")

#         print("std of attn LN output:", torch.std(MSA_output[0]))
#         print("mean of attn LN output:", torch.mean(MSA_output[0]))
        return MSA_output




class MLP(nn.Module):
    def __init__(self, embedding_dim, r):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu=nn.GELU()
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.res=Residual(embedding_dim)
        self.LN2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        #self.adapt1 = nn.Linear(embedding_dim, r)
        #self.relu=nn.ReLU()
        #self.adapt2 = nn.Linear(r, embedding_dim)
        #self.res=Residual(embedding_dim)

    def forward(self, x, i):




        global y_after_mlp_LN
        global y_after_mlp_res
        global y_before_mlp
        y_before_mlp=x
        #print(f"y_before_mlp{i} : {y_before_mlp[0]}")
        y = self.fc1(x)
        y = self.gelu(y)
        y = self.fc2(y)
        x = self.res(x,y)
        y_after_mlp_res=x
#         print(f"y_after_mlp_res{i} : {y_after_mlp_res[0]}")
#         print("std of MLP LN input:", torch.std(x[0]))
#         print("mean of MLP LN input:", torch.mean(x[0]))
        x = self.LN2(x)
#         print("std of MLP LN output:", torch.std(x[0]))
#         print("mean of MLP LN output:", torch.mean(x[0]))
        

        y_after_mlp_LN=x
        #print(f"y_after_mlp_LN{i} : {y_after_mlp_LN[0]}")
        return x

class MLPHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)



    def forward(self, x):
        cls_token_output = x  # Use the first token for classification (similar to CLS token)
        #print(f"cls_token final : {x}")
        cls_token_output = self.dense(cls_token_output)
        #print(f"cls token after dense : {cls_token_output}")
        cls_output = self.head(cls_token_output)
        #print(f"final output : {cls_output}")
        return cls_output

class Encoder(nn.Module):
    def __init__(self, max_seq_len, num_head, embedding_dim, dim_head, r):
        super().__init__()
        self.attn = Attention(max_seq_len, num_head, embedding_dim, dim_head, r)
        self.mlp = MLP(embedding_dim, r)


    def forward(self, x, i, mask=None):

        global y_before_attn
        y_before_attn=x
#         print(f"shape of x = {x.shape}")
#         print(f"y_before_attn{i} : {y_before_attn[0]}")
        y = self.attn(x, i, mask)

        global y_after_attn
        y_after_attn=y
        #print(f"y_after_attn{i} : {y_after_attn[0]}")


        #print(f"y_after_res{i} : {y_after_res[0]}")
        y = self.mlp(y,i)

        global y_after_mlp
        y_after_mlp=y
        #print(f"y_after_mlp{i} : {y_after_mlp[0]}")


        global y_after_mlpres
        y_after_mlpres=x
        #print(f"y_after_mlpres{i} : {y_after_mlpres[0]}")
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
        #print("X.shape", x.shape)
        global y_patch
        global yp
        y_patch=x
#         print(f"y_patch : {y_patch[0]}")
#         print(f"y_patch{y_patch}")
        for i in range(1, self.num_blocks + 1):
            x = getattr(self, f'encoder{i}')(x, i-1, mask)
            #print(x)

            yp=x
            #print(f"yp{i} : {yp[0]}")
        #print("all printed")

        x = self.mlphead(x[:,0])
        #print(f"final_mlphead : {x[0]}")
        return x