#加载P-Tuning 的checkpoint
from transformers import AutoModel,AutoTokenizer,AutoConfig
import torch
import os
tokenizer = AutoTokenizer.from_pretrained('/mnt/dvc2/lijinxuan/ChatGLM2-6B/chatglm2-6b',trust_remote_code=True)
config = AutoConfig.from_pretrained("/mnt/dvc2/lijinxuan/ChatGLM2-6B/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("/mnt/dvc2/lijinxuan/ChatGLM2-6B/chatglm2-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join('/mnt/dvc2/lijinxuan/ChatGLM2-6B/ChatGLM2-6B/ptuning/output/agen-chatglm2-6b-pt-128-2e-2/checkpoint-3000',
                                            "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model.eval()

response,history = model.chat(tokenizer,'你好，你是谁',history=[])
print(response)
