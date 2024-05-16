class ProtLM:
    def __init__(self, model_path, cache_dir, compile_model):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self.tokenizer = None
        self.model = None
        self.device = None
        self.forward_pass = None

    def finalize_torch(self):
        import torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)
        if self.compile_model:
            self.model = torch.compile(self.model,
                                       mode="max-autotune",
                                       dynamic=True)

        def forward_pass(self, inp):
            """
            Forward pass of the model.
            """
            with torch.no_grad():
                out = self.model(**inp)
            return out

        self.forward_pass = forward_pass

    def tokenize(self, sequences):
        inp = self.tokenizer.batch_encode_plus(sequences,
                                               padding=True,
                                               return_tensors="pt")
        return inp

    def batch_embed(self, sequences):
        inp = self.tokenize(sequences)
        out = self.forward_pass(inp)
        embs = out.last_hidden_state.cpu().numpy()
        return inp, embs

    def remove_special_tokens(self, embeddings, attention_mask):
        clean_embeddings = []
        embeddings = list(embeddings)
        for emb, mask in zip(embeddings, attention_mask):
            seq_len = (mask == 1).sum()
            seq_emb = emb[:seq_len - 1]
            clean_embeddings.append(seq_emb)

        return clean_embeddings

    def get_sequence_embeddings(self, sequences):
        inp, embs = self.batch_embed(sequences)
        embs = self.remove_special_tokens(embs, inp["attention_mask"])
        return embs


class ProtT5Encoder(ProtLM):
    def __init__(self,
                 model_path,
                 cache_dir,
                 compile_model=False,
                 local_files_only=True,
                 pooling_layer=False):
        from transformers import T5EncoderModel, T5Tokenizer
        super().__init__(model_path, cache_dir, compile_model)
        self.model = T5EncoderModel.from_pretrained(model_path,
                                                    cache_dir=cache_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, local_files_only=local_files_only)
        self.finalize_torch()


class ESMEncoder(ProtLM):
    def __init__(self,
                 model_path,
                 cache_dir,
                 compile_model=False,
                 local_files_only=True,
                 pooling_layer=False):
        from transformers import EsmModel, EsmTokenizer

        super().__init__(model_path, cache_dir, compile_model)
        self.model = EsmModel.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer = EsmTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            add_pooling_layer=pooling_layer)
        self.finalize_torch()

    def remove_special_tokens(self, embeddings, attention_mask):
        clean_embeddings = super().remove_special_tokens(
            embeddings, attention_mask)
        # remove <cls> token
        clean_embeddings = [emb[1:] for emb in clean_embeddings]
        return clean_embeddings


class Ankh(ProtLM):
    def __init__(self,
                 model_path,
                 cache_dir,
                 compile_model=False,
                 local_files_only=True,
                 pooling_layer=False):
        from transformers import AutoTokenizer, T5EncoderModel
        super().__init__(model_path, cache_dir, compile_model)
        self.model = T5EncoderModel.from_pretrained(model_path,
                                                    cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            add_pooling_layer=pooling_layer)
        self.finalize_torch()

        def tokenize(self, sequences):
            inp = self.tokenizer.batch_encode_plus(
                sequences,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            return inp
