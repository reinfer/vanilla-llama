import argparse
from inference import LLaMAInference

if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    args = parser.parse_args()

    llama = LLaMAInference(args.llama_path, args.model)
    print(llama.generate(["My name is Federico"]))