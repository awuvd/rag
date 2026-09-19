"""
week2_D12.py —— T5：把 1023 个知识块向量化，存入 Chroma 向量库
================================================================
作用：读 T3 切好的知识块 → 用 BGE 模型编码成向量 → 存进向量库。
      完成后，就能"按语义检索"了（T7 要做的事）。

运行：py week2_D12.py
输入：data/chunks.json      （T3 的产物，1023 个知识块）
输出：chroma_db/            （向量库，持久化在磁盘上）

运行结果：col.count() == 1023

【本文件踩过的 3 个坑（都是很典型的）】
  1. 环境变量必须在 import 之前设置，且必须用 os.environ 写
     （写成 HF_HUB_OFFLINE=1 只是建了个 Python 变量，库看不见）
  2. add() 是【批量】接口：4 个参数都要传"列表"，不是"单个元素"
  3. 循环里要用 append【累积】，用 = 只会保留最后一个
"""

import os

# ⚠️ 坑 1：这两件事必须在 import sentence_transformers 之前做！
#   因为 huggingface_hub 在【被导入的那一刻】就读取环境变量，决定去哪加载模型。
#   HF_HUB_OFFLINE='1' → 强制离线：模型已缓存在本地，别去网上检查更新（否则会连超时重试）
#   （另外注意：必须用 os.environ['X']='值' 写，直接写 X=1 只是 Python 变量，库读不到）
os.environ['HF_HUB_OFFLINE'] = '1'

import json                                    # 读 chunks.json
from sentence_transformers import SentenceTransformer   # 文本向量化模型
import chromadb                                # 向量数据库

# ============================================================
# ① 读入知识块（T3 的产物）
# ============================================================
with open(r"E:\llm-study-plan\rag\data\chunks.json", 'r', encoding='utf-8') as f:
    txt = json.load(f)
# json.load 会自动把 JSON 数组读成 Python list：
#   txt       → list，长度 1023
#   txt[i]    → dict，键有 id / chapter / title / chunk_index / text

# ② 把每个块的 text 抽出来，组成"字符串列表"
#    因为 model.encode() 要的输入是 list[str]，不是 list[dict]
texts = []
for i in range(int(len(txt))):          # （int() 是多余的，len() 本来就返回整数）
    texts.append(txt[i]['text'])
# 验证：确认提取成功（应该是 <class 'str'>）
print(type(texts[0]))

# ============================================================
# ③ 加载 BGE 中文向量化模型
# ============================================================
# 这一步做的事：把训练好的模型权重读进内存（约 102M 参数）。
# 模型已缓存在 ~/.cache/huggingface/，所以离线也能加载（约 19 秒）。
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
device = 'cpu'      # ⚠️ 这行其实没被用到（变量定义了没用）。想指定设备要传给构造函数：
                    #    SentenceTransformer("...", device='cpu')

# ============================================================
# ④ 编码：文本 → 向量（T5 的核心）
# ============================================================
# encode() 内部做了 4 件事（详见函数速查手册）：
#   分词 → 过 Transformer → 池化（多token合成1个句子向量）→ 归一化
#
# normalize_embeddings=True 的意义：
#   把向量长度缩到 1（单位向量）→ 之后"点积"就直接等于"余弦相似度"
#   → 检索更快、更准
vectors = model.encode(texts, normalize_embeddings=True)
# vectors 是 numpy 数组，形状 (1023, 768)：1023 块 × 768 维
print(vectors.shape)

# ============================================================
# ⑤ 连接向量库（Persistent = 持久化到磁盘，关掉程序数据还在）
# ============================================================
client = chromadb.PersistentClient(path=r"E:\llm-study-plan\rag\chroma_db")

# ⚠️ 先删掉旧的同名集合，再重建 —— 这样脚本【可以重复运行】。
#    否则第二次跑时，集合里已有同名 id，add() 会报"id 重复"。
try:
    client.delete_collection("nndl_book")
except:
    pass            # 集合不存在时会抛异常，忽略即可（更规范的写法是 except Exception:）

col = client.create_collection(name='nndl_book')   # 建一张"表"

# ============================================================
# ⑥ 组装 4 个列表，准备批量入库
# ============================================================
# ⚠️ 坑 3：循环里必须用 append【累积】！用 = 只会保留最后一个元素。
#   所以三个空列表要在循环【之前】建好。
ids = []            # 每块的唯一 id（如 'ch01_001'）
documents = []      # 每块的原文（检索时要返回给用户看）
meta = []           # 每块的元信息（用于溯源）

for i in range(int(len(txt))):
    # 元信息是"自己组装"的：从原来 5 个键里，挑 3 个有用的
    # （text 不放进来，因为它已经通过 documents 传了，而且很长、浪费空间）
    meta.append({'chapter': txt[i]['chapter'], 'title': txt[i]['title'], 'chunk_index': txt[i]['chunk_index']})
    ids.append(txt[i]["id"])         # 注意：数据字段叫 id（单数）
    documents.append(txt[i]['text'])

# ⚠️ 坑 2：向量库要的是"普通 Python 列表"，而 vectors 是 numpy 数组，
#   所以必须 .tolist() 转换。
embeddings = vectors.tolist()

# ============================================================
# ⑦ 批量入库（只调用【一次】！）
# ============================================================
# 4 个参数都必须传"长度 1023 的列表"，而且【第 i 个互相对应】：
#   第 i 个 id + 第 i 段原文 + 第 i 份元信息 + 第 i 个向量 = 同一块知识
col.add(
    ids=ids,                    # 注意参数名是 ids（复数），不是 id
    documents=documents,
    metadatas=meta,
    embeddings=embeddings,
)

# ============================================================
# ⑧ 验证
# ============================================================
print(col.count())      # 应该输出 1023
