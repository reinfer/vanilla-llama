import argparse
import json
import torch
from accelerate import init_empty_weights
from tqdm import tqdm
from pathlib import Path
from llama import ModelArgs, Tokenizer, Transformer

class LLaMAInference:
    def __init__(self, model_path, tokenizer_path):
        checkpoints = sorted(Path(model_path).glob("*.pth"))
        with open(Path(model_path) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args = ModelArgs(
            max_seq_len=2048,
            max_batch_size=1,
            **params
        )

        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        with init_empty_weights():
            torch.set_default_tensor_type(torch.HalfTensor)
            self.model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        key_to_dim = {
            "w1": 0,
            "w2": -1,
            "w3": 0,
            "wo": -1,
            "wq": 0,
            "wk": 0,
            "wv": 0,
            "output": 0,
            "tok_embeddings": -1,
            "ffn_norm": None,
            "attention_norm": None,
            "norm": None,
            "rope": None,
        }

        converted_state_dict = {}

        for i, ckpt in tqdm(enumerate(checkpoints), total=len(checkpoints)):
            checkpoint = torch.load(ckpt, map_location="cpu")
            for parameter_name, parameter in self.model.named_parameters():
                converted_state_dict[parameter_name] = torch.zeros_like(parameter, device="cpu")
                short_name = parameter_name.split(".")[-2]
                if key_to_dim[short_name] is None and i == 0:
                    converted_state_dict[parameter_name] = checkpoint[parameter_name]
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    converted_state_dict[parameter_name][size * i : size * (i + 1), :] = checkpoint[
                        parameter_name
                    ]
                elif key_to_dim[short_name] == -1:
                    size = checkpoint[parameter_name].size(-1)
                    converted_state_dict[parameter_name][:, size * i : size * (i + 1)] = checkpoint[
                        parameter_name
                    ]
                del checkpoint[parameter_name]
            del checkpoint

        torch.save(converted_state_dict, "converted_state_dict.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    args = parser.parse_args()

    llama = LLaMAInference(args.model_path, args.tokenizer_path)
    