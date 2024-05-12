from transformers import EsmTokenizer, EsmModel
import gc
import torch
from abc import abstractmethod
from tqdm import tqdm
import h5py
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:

    def __init__(self, model_path, cache_dir, device):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.device = device

    def batch_input(self, sequences, batch_size):
        for i in range(0, len(sequences), batch_size):
            yield sequences[i:i + batch_size]

    def tokenize(self, sequences):
        inp = self.tokenizer(sequences, padding=True, return_tensors="pt")
        return inp

    @abstractmethod
    def batch_embed(self, sequences, batch_size):
        raise NotImplementedError



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
        # convert to half precision if cuda
        if self.device.type == "cuda":
            self.model.half()

        self.model.to(self.device)
        # compile model
        self.model = torch.compile(self.model, mode="max-autotune", dynamic=True)
        self.model.eval()
        gc.collect()

    def batch_embed(self, sequences):

        embeddings = []
        inp = self.tokenize(sequences)
        inp = {k: v.to(self.device) for k, v in inp.items()}

        with torch.no_grad():
            out = self.model(**inp)

        embs = out.last_hidden_state

        for seq_num in range(len(embs)):
            seq_len = (inp["attention_mask"][seq_num] == 1).sum()
            seq_emb = embs[seq_num][: seq_len - 1]
            # remove first <cls> token
            embeddings.append(seq_emb[1:].cpu().numpy())

        return embeddings



