import gc
import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from tm_vec.model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import (cosine_similarity_tm, embed_tm_vec,
                                 featurize_prottrans)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_PROT_LEN = 1024
CACHE_DIR = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache"

#TM-Vec model paths
model_folder = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_retrained/prott5"
tm_vec_model_cpnts = [
    os.path.join("model_matched.ckpt"),
    os.path.join("model.ckpt"),
]
tm_vec_model_config_file = os.path.join(model_folder, "params.json")
tm_vec_model_config = trans_basic_block_Config.from_json(
    tm_vec_model_config_file)

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                        do_lower_case=False,
                                        cache_dir=CACHE_DIR)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                       cache_dir=CACHE_DIR)
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()
# model = torch.compile(model, mode="reduce-overhead", dynamic=True)

logger.info("Model loaded.")

file = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/malidup_sequences_and_tm_scores.csv"
sequences = pd.read_csv(file)

for i, cpnt in tqdm(enumerate(tm_vec_model_cpnts)):

    model_deep = trans_basic_block.load_from_checkpoint(
        cpnt, config=tm_vec_model_config, map_location=device)
    model_name = cpnt.split('/')[-1].split('.')[0]

    #Load the TM-Vec model
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()

    tm_score_predictions = []
    embeddings_dict = {}

    for n in tqdm(range(sequences.shape[0])):

        sequence_str_1 = sequences.loc[n, 'Sequence 1']
        sequence_str_2 = sequences.loc[n, 'Sequence 2']

        if len(sequence_str_1) > MAX_PROT_LEN or len(
                sequence_str_2) > MAX_PROT_LEN:
            continue

        sequence_1 = np.expand_dims(sequence_str_1, axis=0)
        protrans_sequence_1 = featurize_prottrans(sequence_1, model, tokenizer,
                                                  device)
        sequence_2 = np.expand_dims(sequence_str_2, axis=0)
        protrans_sequence_2 = featurize_prottrans(sequence_2, model, tokenizer,
                                                  device)

        embedded_sequence_1 = embed_tm_vec(protrans_sequence_1, model_deep,
                                           device)
        embedded_sequence_2 = embed_tm_vec(protrans_sequence_2, model_deep,
                                           device)

        #Predict the TM-score for sequence 1 and 2, using the TM-Vec embeddings
        predicted_tm_score = cosine_similarity_tm(
            torch.tensor(embedded_sequence_1),
            torch.tensor(embedded_sequence_2))

        tm_score_predictions.append(predicted_tm_score.item())

    sequences[f'tm_vec_{model_name}'] = np.array(tm_score_predictions)

sequences.to_csv(
    '/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/malidup_sequences_and_tm_scores_preds.csv',
    index=False)
