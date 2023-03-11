import argparse
import json
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from pathlib import Path
from llama import ModelArgs, Tokenizer, Transformer, LLaMA

class LLaMAInference:
    def __init__(self, model_path, tokenizer_path):
        params = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": -1}
        model_args = ModelArgs(
            max_seq_len=2048,
            max_batch_size=1,
            **params
        )

        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        with init_empty_weights():
            torch.set_default_tensor_type(torch.HalfTensor)
            model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map="auto",
            no_split_module_classes=["TransformerBlock"]
        )
        
        self.generator = LLaMA(self.model, self.tokenizer)

    def generate(self, texts, max_length=128):
        results = self.generator.generate(
            texts, max_gen_len=256, temperature=0.8, top_p=0.95
        )
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    args = parser.parse_args()

    llama = LLaMAInference(args.model_path, args.tokenizer_path)
    llama.generate(["This is a test: "])
    