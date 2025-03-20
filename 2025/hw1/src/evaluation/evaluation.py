from vllm import LLM, SamplingParams
import os
import re
import json
import jsonlines
import argparse
from tqdm import tqdm
import sys
import warnings
import pdb
import ray
warnings.filterwarnings("ignore")
import torch
from transformers import AutoTokenizer
from math_verify import parse, verify

def get_prompts(dev_set, apply_chat_template):
    prompt2answer={}
    processed_prompts=[]
    with open(f"./data/eval/{dev_set}.jsonl", 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            chat=[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, {"role": "user", "content": line['question']}]
            prompt=apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            processed_prompts.append(prompt)
            prompt2answer[prompt]=str(line['answer'])
    print(processed_prompts[-1])
    return processed_prompts, prompt2answer
    
def math_equal(gold, answer):
    try:
        gold=parse(gold)
        answer=parse(answer)
        return verify(gold, answer)
    except:
        return False

def math_eval(response, gt):
    matches = re.findall(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})", response)
    if len(matches)==0: pred=""
    else: pred=matches[-1][:-1]
    return math_equal(pred, gt)


def main(args):
    num_gpus = torch.cuda.device_count()
    another_args = {'max_num_batched_tokens': 32768}
    apply_chat_template=AutoTokenizer.from_pretrained(args.model_path).apply_chat_template
    print("?????")
    
    llm = LLM(model=args.model_path, tensor_parallel_size=num_gpus, **another_args, trust_remote_code=True)
    eval_acc={}
    eval_responses={}
    sampling_params = SamplingParams(n=1, temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)
    
    for dev_set in ["AMC23", "GSM8k", "MATH500", "OlympiadBench"]:
        processed_prompts, prompt2answer = get_prompts(dev_set, apply_chat_template)
        outputs = llm.generate(processed_prompts, sampling_params)
        eval_results=[]
        eval_response=[]
        for output in outputs:
            prompt=output.prompt
            response=output.outputs[0].text
            answer=prompt2answer[prompt]
            is_correct=math_eval(response, answer)
            eval_results.append(is_correct)
            eval_response.append({"question": prompt, "response": response, "results": is_correct, "answer": answer})
                
        acc=sum(1 for result in eval_results if result is True)/len(eval_results)
        print(f"{dev_set} acc: {acc}")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument('--output_name', type=str, default='./data/output/eval/temp')
    parser.add_argument('--temperature', type=int, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=2048)
    args = parser.parse_args()

    main(args)
    