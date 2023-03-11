# vanilla-llama 🦙

> 📢 `vanilla-llama` is a plain-pytorch implementation of `LLaMA` with minimal differences with respect to the original Facebook's implementation. You can run `vanilla-llama` on 1, 2, 4, 8 or 100 GPUs

**🔥Couldn't be more easy to use**

**🔋Comes with an inference server included**

```python
from inference import LLaMAInference

llama = LLaMAInference(llama_path, "65B")
print(llama.generate(["My name is Federico"]))
```

## Features

- Easy to use and fine-tune 🔥
- Uses (🤗 accelerate)[https://github.com/huggingface/accelerate] to distribute the model on all available GPUs
- Comes with batteries included🔋
- Nice one line loading and generation 😎

## Examples

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

## Installation

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

## Convert LLaMA weights

To convert LLaMA weights to a plain pytorch state-dict run

```
python convert.py --llama-path <ORIGINAL-LLAMA-WEIGHTS> --model <MODEL> --output-path <CONVERTED-WEIGHTS-PATH>
```

## Run example

Run the provided example

```
python example.py --llama-path <CONVERTED-WEIGHTS-PATH> --model <MODEL>
```