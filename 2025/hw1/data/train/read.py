import os
import json
import jsonlines

with open("/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/xfli/o1/reference/cs2916/homework1/data/train/prompt.jsonl", 'r', encoding='utf-8') as f:
    data=json.load(f)

with open("/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/xfli/o1/reference/cs2916/homework1/data/train/prompt.jsonl", 'w', encoding='utf-8') as f:
    for line in data:
        json.dump(line, f)
        f.write('\n')
