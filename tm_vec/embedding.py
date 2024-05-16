import logging
from abc import abstractmethod

import torch
from transformers import EsmModel, EsmTokenizer, T5EncoderModel, T5Tokenizer

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ProtLM:
    """
    Base class for protein LLMs based on BERT architecture.
    Includes methods for tokenization and batching of sequences.
    Tested on ESM and ProtT5
    """
    def __init__(self, model_path, cache_dir, compile_model):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self.tokenizer = None
        self.model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def finalize_torch(self):
        self.model.eval()
        self.model.to(self.device)
        if self.compile_model:
            self.model = torch.compile(self.model,
                                       mode="max-autotune",
                                       dynamic=True)

    # # convert to half precision if cuda
    # if self.device.type == "cuda":
    #     self.model.half()

    def tokenize(self, sequences):
        inp = self.tokenizer(sequences, padding=True, return_tensors="pt")
        return inp

    def forward_pass(self, inp):
        with torch.no_grad():
            out = self.model(**inp)
        return out

    def batch_embed(self, sequences):

        inp = self.tokenize(sequences)
        out = self.forward_pass(inp)
        embs = out.last_hidden_state.cpu().numpy()

        return inp, embs

    @abstractmethod
    def remove_special_tokens(self, embedding, attention_mask):
        # add a verbose error
        raise NotImplementedError(
            "Method for removing special tokens is not implemented, "
            "the embeddings should not be used as is.")

    def get_sequence_embeddings(self, sequences):
        """
        Get embeddings from the last layer for sequences.
        """
        inp, embs = self.batch_embed(sequences)
        # remove special tokens
        embs = self.remove_special_tokens(embs, inp["attention_mask"])
        return embs


class ESMEncoder(ProtLM):
    def __init__(self,
                 model_path: str,
                 cache_dir: str,
                 compile_model: bool = False,
                 local_files_only: bool = True,
                 pooling_layer: bool = False):

        super().__init__(model_path, cache_dir, compile_model)
        self.model = EsmModel.from_pretrained(self.model_path,
                                              cache_dir=self.cache_dir)
        self.tokenizer = EsmTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            add_pooling_layer=pooling_layer)
        super().finalize_torch()

    def remove_special_tokens(self, embeddings, attention_mask):
        """
        Remove special tokens from the embedding
        """
        embeddings = list(embeddings)

        clean_embeddings = []

        for seq_num in range(len(embeddings)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emb = embeddings[seq_num][:seq_len - 1]
            # remove first <cls> token
            clean_embeddings.append(seq_emb[1:])

        return clean_embeddings


class ProtT5Encoder(ProtLM):
    def __init__(self,
                 model_path: str,
                 cache_dir: str,
                 compile_model: bool = False,
                 local_files_only: bool = True,
                 pooling_layer: bool = False):

        super().__init__(model_path, cache_dir, compile_model)
        self.model = T5EncoderModel.from_pretrained(self.model_path,
                                                    cache_dir=self.cache_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only)
        super().finalize_torch()

    def remove_special_tokens(self, embeddings, attention_mask):
        """
        Remove special tokens from the embedding
        """
        clean_embeddings = []

        for seq_num in range(len(embeddings)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emb = embeddings[seq_num][:seq_len - 1]
            clean_embeddings.append(seq_emb)

        return clean_embeddings
