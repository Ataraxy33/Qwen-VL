#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-VL-Chat', cache_dir='.')