import os
import json
import jsonlines
import pandas 
import datasets
import argparse
import datasets
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer


from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/hyzou/ckpt/huggingface/Qwen/Qwen2.5-Math-1.5B")
def to_messages(line):
    messages=[
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": line["prompt"]},
    ]
    text=line["prompt"]
    return text, messages, line['answer']


def save_dataset(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(dataset, Dataset):
        dataset.save_to_disk(save_dir)
    elif isinstance(dataset, DatasetDict):
        dataset.save_to_disk(save_dir)
    else:
        raise ValueError("")

def main(args):
    data=[]
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            data.append(line)
        
    print(len(data))
    save_data=[]
    lengths=[]
    for line in data:
        text, messages, answer=to_messages(line)
        s={"dataset": "rl", "context": text, "context_messages": messages, "answer": str(answer), "source": "math.3k"}
        if len(tokenizer.encode(text))>=310: continue
        save_data.append(s)
    
    
    evaldata=[]
    with open("./data/eval/RL.jsonl", 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            evaldata.append(line)
    eval_save_data=[]
    for line in evaldata:
        text, messages, answer=to_messages(line)
        if len(tokenizer.encode(text))>=310: continue
        s={"dataset": "rl", "context": text, "context_messages": messages, "answer": str(answer).strip(), "source": line['source']}
        eval_save_data.append(s)
    
    training_data=save_data
    valiation_data=eval_save_data
    
    print(json.dumps(training_data[-1], indent=4))    
    print(json.dumps(valiation_data[-1], indent=4))
    training_data=pd.DataFrame(training_data)
    validation_data=pd.DataFrame(valiation_data)
    
    print("training_data_length", len(training_data))
    print("validation_data_length", len(validation_data))

    data_dict={
        "train": training_data,
        "test": validation_data,
    }
    dataset_dict=DatasetDict({
        split_name: Dataset.from_pandas(df) for split_name,df in data_dict.items()
    })
    save_dataset(dataset_dict, args.output_path)



if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/xfli/o1/reference/cs2916/homework1/data/train/math3k_rl_prompt.jsonl")
    parser.add_argument("--output_path", type=str, default="/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/xfli/o1/reference/cs2916/homework1/data/train/math3k_rl_prompt")
    args=parser.parse_args()
    main(args)
