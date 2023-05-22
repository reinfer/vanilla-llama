# vanilla-llama 🦙

> 📢 `vanilla-llama` is a plain-pytorch implementation of `LLaMA` with minimal differences with respect to the original Facebook's implementation. You can run `vanilla-llama` on 1, 2, 4, 8 or 100 GPUs

**Couldn't be more easy to use 🔥**

**Comes with an inference server included 🔋**

```python
from inference import LLaMAInference

llama = LLaMAInference(llama_path, "65B")
print(llama.generate(["My name is Federico"]))
```

## Features 🏆

- Easy to use and fine-tune 🔥
- Uses [🤗 accelerate](https://github.com/huggingface/accelerate) to distribute the model on all available GPUs
- Comes with batteries included🔋
- Nice one line loading and generation 😎

## Examples 🤔 

Stop generation on specific tokens (`13` is the new-line token)

```python
llama.generate(["Chat:\nHuman: Hi i am an human\nAI:"], stop_ids=[13])
```

Stop generation on specific texts

```python
llama.generate(["Question: is the sky blue?\nAnswer:"], stop_words=["Question"])
```

Batch generation

```python
llama.generate(["My name is Federico", "My name is Zuck"])
```

Repetition Penalty
```python
llama.generate(["This is a list of awesome things:\n"], repetition_penalty=(1.0 / 0.85))
```

## Inference server 🌐 

Install server requirements

```
pip install -r server_requirements.txt
```

Run the server
```
python server.py --llama-path <CONVERTED-WEIGHTS-PATH> --model <MODEL>
```

Test it!

```
curl -X GET http://localhost:3000/generate -H "Content-Type: application/json" -d '{"prompt": "REST server are very useful becouse"}'
```

## Installation ⚙️ 

Clone this repository

```
git clone https://github.com/galatolofederico/vanilla-llama.git
cd vanilla-llama
```

Install the requirements

```
python3 -m venv env
. ./env/bin/activate
pip install -r requirements.txt
```

## Convert LLaMA weights 🏭 

To convert LLaMA weights to a plain pytorch state-dict run

```
python convert.py --llama-path <ORIGINAL-LLAMA-WEIGHTS> --model <MODEL> --output-path <CONVERTED-WEIGHTS-PATH>
```

## Run example ✨ 

Run the provided example

```
python example.py --llama-path <CONVERTED-WEIGHTS-PATH> --model <MODEL>
```


## Many thanks to these amazing projects ❤️

- [llama hf implementation](https://github.com/zphang/transformers/tree/llama_push)
- [LLaMA: INT8 edition](https://github.com/tloen/llama-int8)
- [pyllama](https://github.com/juncongmoo/pyllama)
- [shawwn fork](https://github.com/shawwn/llama)
