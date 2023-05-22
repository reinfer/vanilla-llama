from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Union

from inference import LLaMAInference

def create_app(args):
    app = FastAPI()
    llama = LLaMAInference(
        args.llama_path,
        args.model,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len
    )

    class GenerateRequest(BaseModel):
        prompt: Union[List[str], str]
        temperature: float = 0.8
        top_p: float = 0.95
        stop_ids: List[int] = None
        stop_words: List[str] = None
        max_length: int = 512
        repetition_penalty: float = 1.0
    
    def verify_token(req: Request):
        if args.token == "":
            return True
        
        token = req.headers["Authorization"]
        if token != args.token:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )
        return True

    @app.get("/generate")
    def generate(gen_args: GenerateRequest, authorized: bool = Depends(verify_token)):
        if isinstance(gen_args.prompt, str):
            gen_args.prompt = [gen_args.prompt]

        if len(gen_args.prompt) > args.max_batch_size:
            return {"error": "Batch size too small"}

        generated, stats = llama.generate(
            gen_args.prompt,
            max_length=gen_args.max_length,
            temperature=gen_args.temperature,
            top_p=gen_args.top_p,
            repetition_penalty=gen_args.repetition_penalty,
            stop_ids=gen_args.stop_ids,
            stop_words=gen_args.stop_words
        )

        return {"generated": generated, "stats": stats}

    return app


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["7B", "13B", "30B", "65B"])
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--token", type=str, default="")

    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
