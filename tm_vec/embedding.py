from transformers import EsmTokenizer, EsmModel
import torch
from abc import abstractmethod
from tqdm import tqdm
import h5py
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTModel:
    """
    Base class for protein LLMs based on BERT architecture.
    Includes methods for tokenization and batching of sequences.
    Tested on ESM and ProtT5
    """


    def __init__(self, model_path, cache_dir, device):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.device = device

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
        embs = out.last_hidden_state

        return inp, embs

    @abstractmethod
    def remove_special_tokens(self, embedding, attention_mask):
        # add a verbose error
        raise NotImplementedError("Method for removing special tokens is not implemented, the embeddings should not be used as is.")

    def get_sequence_embeddings(self, sequences):
        """
        Get embeddings from the last layer for sequences.
        """
        inp, embs = self.batch_embed(sequences)
        # remove special tokens
        embs = self.remove_special_tokens(embs, inp["attention_mask"])
        return embs



class ESMEncoder(Model):

    def __init__(self,
                 model_path: str,
                 cache_dir: str,
                 device : str,
                 local_files_only: bool = True,
                 pooling_layer : bool = False):

        super().__init__(model_path, cache_dir, device)
        self.model = EsmModel.from_pretrained(self.model_path, cache_dir=self.cache_dir)
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_path, cache_dir=self.cache_dir, local_files_only=local_files_only,
                                                      add_pooling_layer=pooling_layer)
        # # convert to half precision if cuda
        # if self.device.type == "cuda":
        #     self.model.half()

        self.model.to(self.device)
        self.model.eval()
        # compile model
        self.model = torch.compile(self.model, mode="max-autotune", dynamic=True)


        def remove_special_tokens(self, embedding, attention_mask):
            """
            Remove special tokens from the embedding
            """
            embeddings = []

            for seq_num in range(len(embs)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emb = embs[seq_num][: seq_len - 1]
                # remove first <cls> token
                embeddings.append(seq_emb[1:])

            return embeddings


class ProtT5Encoder:
    def __init__(self,
                 model_path: str,
                 cache_dir: str,
                 device : str,
                 local_files_only: bool = True):

        super().__init__(model_path, cache_dir, device)
        self.model = ProtT5ForConditionalGeneration.from_pretrained(self.model_path, cache_dir=self.cache_dir)
        self.tokenizer = ProtT5Tokenizer.from_pretrained(self.model_path, cache_dir=self.cache_dir, local_files_only=local_files_only)
        # # convert to half precision if cuda
        # if self.device.type == "cuda":
        #     self.model.half()

        self.model.to(self.device)
        self.model.eval()
        # compile model
        self.model = torch.compile(self.model, mode="max-autotune", dynamic=True)


        def remove_special_tokens(self, embedding, attention_mask):
            """
            Remove special tokens from the embedding
            """
            embeddings = []

            for seq_num in range(len(embs)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emb = embs[seq_num][: seq_len - 1]

            return embeddings






