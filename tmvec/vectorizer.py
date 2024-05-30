from typing import List

import numpy as np
import torch
from tqdm import tqdm

from tmvec.embedding import ProtT5Encoder
from tmvec.model import trans_basic_block, trans_basic_block_Config


class TMVec:
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 cache_dir: str = None,
                 protlm_path: str = None,
                 protlm_tokenizer_path: str = None,
                 local_files_only: bool = False):

        self.model_path = model_path
        self.config_path = config_path
        self.protlm_path = None
        self.protlm_tokenizer_path = None
        self.cache_dir = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self.compile_model = None
        self.config = trans_basic_block_Config.from_json_file(self.config_path)
        self.model = trans_basic_block.load_from_checkpoint(
            self.model_path, config=self.config, map_location=self.device)
        if self.protlm_tokenizer_path is None:
            self.protlm_tokenizer_path = "Rostlab/prot_t5_xl_uniref50"
        if str(self.device) == "cuda":
            if self.protlm_path is None:
                self.protlm_path = "Rostlab/prot_t5_xl_uniref50"
            self.backend = "torch"
            self.compile_model = True
        elif str(self.device) == "cpu":
            if self.protlm_path is None:
                self.protlm_path = "valentynbez/prot-t5-xl-uniref50-onnx"
            self.backend = "onnx"
            self.compile_model = False
        self.embedder = ProtT5Encoder(self.protlm_path,
                                      self.protlm_tokenizer_path,
                                      self.cache_dir,
                                      local_files_only=local_files_only,
                                      backend=self.backend,
                                      compile_model=self.compile_model,
                                      threads=1)

    def embed_single_protin(self, sequence: List[str]):
        embedding = self.embedder.get_sequence_embeddings(sequence)[0]
        return embedding

    def embedding_to_vector(self, embedding: np.ndarray):
        prot_embedding = torch.tensor(embedding).unsqueeze(0).to(self.device)
        padding = torch.zeros(prot_embedding.shape[0:2]).type(
            torch.BoolTensor).to(self.device)
        tm_vec_embedding = self.model(prot_embedding,
                                      src_mask=None,
                                      src_key_padding_mask=padding)

        return tm_vec_embedding.cpu().detach().numpy()

    def vectorize_proteins(self, sequences: List[str]):
        embed_all_sequences = []
        for seq in tqdm(sequences,
                        desc="Vectorizing proteins",
                        unit="proteins",
                        total=len(sequences)):
            prottrans_embedding = self.embed_single_protin([seq])
            embedded_sequence = self.embedding_to_vector(prottrans_embedding)
            embed_all_sequences.append(embedded_sequence)

        return np.concatenate(embed_all_sequences, axis=0)

    def delete_vectorizer(self):
        """
        Removes TMVec model to free up memory
        for alignment model.
        """

        del self.model
        torch.cuda.empty_cache()
