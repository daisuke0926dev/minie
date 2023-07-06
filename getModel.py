from transformers import AutoTokenizer, AutoModelForCausalLM

def getTokenizer():
    return AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
    
    
def getModel():
    # 標準
    #model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")
    # 自動
    #model = AutoModelForCausalLM.from_pretrained（"rinna/japanese-gpt-neox-3.6b-instruction-sft", device_map='auto'）
    # 自動（VRAM16GB以下でも8GBはNG）
    #model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", torch_dtype=torch.float16, device_map='auto')
    # CPU指定
    # return AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft").to("cpu")
    # GPU指定
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft").to("cuda")
    # GPU指定（VRAM16GB以下でも8GBはNG）
    #model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", torch_dtype=torch.float16).to("cuda")