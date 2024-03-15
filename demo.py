# from modelscope import (
#     snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# )
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.1.0'
# model_dir = snapshot_download(model_id, revision=revision)
torch.manual_seed(1234)

model_dir = "/home3/xyd/Qwen-VL/Qwen-VL-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:1", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 第一轮对话
# query = tokenizer.from_list_format([
#     {'image': '/home3/xyd/qwen/Qwen-VL-Chat/imgs/桁架拱桥.jpg'}, # Either a local path or an url
#     {'text': '这张图片是一座桁架拱桥，它主要的结构包括实腹段、腹杆和拱肋等。拱肋是拱桥主拱圈的骨架，形状是一个弧形段。请帮我在图片中框出拱肋的位置。'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

query = tokenizer.from_list_format([
    {'image': '/home3/xyd/Qwen-VL/imgs/圬工拱桥.jpg'}, # Either a local path or an url
    {'text': '这是圬工拱桥还是桁架拱桥?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# # 第二轮对话
# response, history = model.chat(tokenizer, '框出图中拱肋的位置', history=history)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
  image.save('/home3/xyd/Qwen-VL/results/拱肋.jpg')
else:
  print("no box")