import time
import json
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from pathlib import Path
import os
from llama import ModelArgs, Tokenizer, LLaMA

class LLaMAInference:
    def __init__(self, llama_path, model, model_type, device_map="auto", **kwargs):

        state_dict = os.path.join(llama_path, model, "state_dict.pth")
        params_file = os.path.join(llama_path, model, "params.json")
        tokenizer_path = os.path.join(llama_path, "tokenizer.model")

        assert os.path.exists(os.path.join(llama_path, model)), f"Model {model} does not exist"
        assert os.path.exists(state_dict), f"Model {model} does not exist"
        assert os.path.exists(params_file), f"Model {model} does not exist"
        assert os.path.exists(tokenizer_path), f"Missing tokenizer in {llama_path}"

        with open(params_file, "r") as f:
            params = json.load(f)

        model_args = dict(
            max_seq_len=2048,
            max_batch_size=1,
            **params
        )
        model_args.update(kwargs)
        model_args = ModelArgs(**model_args)

        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        with init_empty_weights():
            torch.set_default_tensor_type(torch.HalfTensor)
            model = model_type(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = load_checkpoint_and_dispatch(
            model,
            state_dict,
            device_map=device_map,
            no_split_module_classes=["TransformerBlock"]
        )

        self.generator = LLaMA(self.model, self.tokenizer)

    def generate(self, texts, temperature=0.8, top_p=0.95, max_length=256, repetition_penalty=1, stop_ids=None, stop_words=None):
        start_time = time.time()
        results, stats = self.generator.generate(
            texts,
            max_gen_len=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_ids=stop_ids,
            stop_words=stop_words
        )
        end_time = time.time()
        stats["total_seconds"] = end_time - start_time
        stats["tok/s"] = max(stats["num_generated_tokens"]) / stats["total_seconds"]
        return results, stats
