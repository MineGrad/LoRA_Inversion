import torch
import torch.nn.functional as F

def recover1(noise_std, patch_id, w_glob, weight_grad,weight, div_factor, new_scaling,D):
    rec=[]
    for t in range(1,len(patch_id)+1):
        #norm=torch.norm(weight_grad[0][:,t-1],p=2)
        #C=0.5*norm #sensitivity
        #factor=norm/C
        noise =noise_std*torch.randn(D)
#weight_grad[0][:,t-1]=weight_grad[0][:,t-1]/max(1,factor)
#print(weight_grad[0][:,t-1])
        weight_grad[0][:,t-1]=weight_grad[0][:,t-1]+noise
        rec.append(weight_grad[0][:,t-1].reshape(1,768) ) 
        rec[t-1]=torch.div(rec[t-1],weight)
        rec[t-1]=torch.mul(rec[t-1],div_factor)
        rec[t-1]=torch.div(rec[t-1],new_scaling)

        rec[t-1]=(rec[t-1]-w_glob['embedding.position_embeddings'][0][patch_id[t-1]].reshape(1,768))
#print(f"rec_without_clamp : {rec}")
        rec[t-1] = rec[t-1].clamp(-1,1)
    return rec

def recover_secagg(noise_std, patch_id, w_glob, weight_grad,weight, div_factor, new_scaling,D):
    rec=[]
    for t in range(1,len(patch_id)+1):
        norm=torch.norm(weight_grad[:,t-1],p=2)
        C=0.5*norm #sensitivity
        factor=norm/C
        noise =noise_std*torch.randn(D)
#weight_grad[0][:,t-1]=weight_grad[0][:,t-1]/max(1,factor)
#print(weight_grad[0][:,t-1])
        weight_grad[:,t-1]=weight_grad[:,t-1]+noise
        rec.append(weight_grad[:,t-1].reshape(1,768) ) 
        rec[t-1]=torch.div(rec[t-1],weight)
        rec[t-1]=torch.mul(rec[t-1],div_factor)
        rec[t-1]=torch.div(rec[t-1],new_scaling)

        rec[t-1]=(rec[t-1]-w_glob['embedding.position_embeddings'][0][patch_id[t-1]].reshape(1,768))
#print(f"rec_without_clamp : {rec}")
        rec[t-1] = rec[t-1].clamp(-1,1)
    return rec

def recover2(patch_id, rec, w_glob, seq_tokenizer):
    word_embed = w_glob['embedding.word_embeddings.weight'] 
#recovered_embeddings = rec
    recovered_words=[]
    word_embed = torch.tensor(word_embed.detach().cpu().numpy()) 

    for i in range(0, len(patch_id)):

        recovered_embeddings = torch.tensor(rec[i], dtype=torch.float32)  # Shape: [17, 768]

# Step 2: Compute cosine similarity between each recovered embedding and all word embeddings
        cos_sim = F.cosine_similarity(recovered_embeddings.unsqueeze(1), word_embed.unsqueeze(0), dim=2)
# Step 3: Find the index of the most similar embedding for each recovered embedding
        most_similar_indices = torch.argmax(cos_sim, dim=1)
#print(most_similar_indices)
        vocab = seq_tokenizer.get_vocab()
        index_to_token = {index: token for token, index in vocab.items()}
#print(index_to_token)
#words = [vocab[idx.item()] for idx in most_similar_indices]
        words = [index_to_token[idx.item()] if isinstance(idx, torch.Tensor) else index_to_token[idx] for idx in       most_similar_indices]
# Print the results
        for idx, word in enumerate(words):
            print(f"Recovered embedding {i+1} corresponds to the word: {word}")
            recovered_words.append(word)
    return recovered_words
    