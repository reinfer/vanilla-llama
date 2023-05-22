import time
import argparse
from inference import LLaMAInference

if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    args = parser.parse_args()
    
    start_loading = time.time()
    llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")

    start_generation = time.time()
    print(llama.generate(["Chat:\nHuman: Hi i am an human\nAI:"], stop_ids=[13]))
    print(f"Inference took {time.time() - start_generation:.2f} seconds")
