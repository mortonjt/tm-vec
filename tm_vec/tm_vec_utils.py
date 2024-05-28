import re

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


# Function to extract ProtTrans embedding for a sequence
def featurize_prottrans(sequences, model, tokenizer, device):
    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]
    ids = tokenizer.batch_encode_plus(seqs,
                                      add_special_tokens=True,
                                      padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(seqs)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    return features[0]


# Embed a protein using tm_vec (takes as input a prottrans embedding)
def embed_tm_vec(prottrans_embedding, model_deep, device):
    prot_embedding = torch.tensor(prottrans_embedding).unsqueeze(0).to(device)
    padding = torch.zeros(prot_embedding.shape[0:2]).type(
        torch.BoolTensor).to(device)
    tm_vec_embedding = model_deep(prot_embedding,
                                  src_mask=None,
                                  src_key_padding_mask=padding)

    return (tm_vec_embedding.cpu().detach().numpy())


#Predict the TM-score for a pair of proteins (inputs are TM-Vec embeddings)
def cosine_similarity_tm(output_seq1, output_seq2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    dist_seq = cos(output_seq1, output_seq2)

    return (dist_seq)


def encode(sequences, model_deep, model, tokenizer, device, batch_size=16):
    embed_all_sequences = []
    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="Embedding sequences",
                  miniters=len(sequences) // 1000):
        protrans_sequences = featurize_prottrans(sequences[i:i + batch_size],
                                                 model, tokenizer, device)
        for seq in protrans_sequences:
            embedded_sequence = embed_tm_vec(seq, model_deep, device)
            embed_all_sequences.append(embedded_sequence)

    return np.concatenate(embed_all_sequences, axis=0)
