"""
week3_D17.py —— T8：检索 + 生成串联 → 🎯 MVP
================================================
这是整个项目的**里程碑文件**：把已有零件拼成一个能回答问题的完整系统。

运行：py week3_D17.py
输入：终端里输入你的问题
输出：终端打印「回答」+「依据来自哪几章」

【完整流程 —— 这就是 RAG 的全部】
    用户问题
       ↓
    ① search()        检索：问题 → 向量 → 在 1023 个块里找 top-5
       ↓
    ② build_prompt()  拼提示词：把「资料 + 问题」拼成一个字符串
       ↓
    ③ ask_llm()       生成：调大模型，让它【只根据资料】回答
       ↓
    ④ 返回「答案 + 出处」

【复用的零件（都不是新写的）】
    · T5（week2_D12）：1023 个知识块已向量化入库
    · T7（week2_D13）：检索流程
    · T6（week3_D15）：ask_llm 大模型调用

【本文件踩过的坑】
    1. build_prompt 返回的是 tuple，调用时【必须解包】
       → 否则整个 tuple 会被当成 prompt 塞进 API，报 422
    2. 模型加载/连库要放在【模块级】（只执行一次），
       放函数里每次调用都重加载，慢 10 秒
    3. 用 ask_llm 时【别把 tuple 当字符串传】，API 那边会报
       "invalid type: string ... expected ContentBlock"
"""

import os

# ⚠️ 必须在 import sentence_transformers 之前设置！
#   因为 huggingface_hub 在【被导入的那一刻】读取环境变量，决定去哪加载模型。
#   HF_HUB_OFFLINE='1' → 强制离线：模型已缓存在本地，不要去联网检查更新
#   （否则会联网超时重试，白白等很久）
os.environ['HF_HUB_OFFLINE'] = '1'

import chromadb
from sentence_transformers import SentenceTransformer
from week3_D15 import ask_llm      # 复用 T6 写好的大模型调用函数

# ============================================================
# ① 模块级初始化（文件被导入时执行【一次】，不是每次调用都执行）
# ============================================================
# 💡 为什么放这里而不是函数里？
#    加载模型约 10 秒、连向量库也要时间。
#    如果在 search() 里写，每问一次问题就重加载一次 → 卡死。
#    放模块级 = 整个程序生命周期只做一次。

model = SentenceTransformer('BAAI/bge-base-zh-v1.5')      # 中文向量化模型（和入库时同一个）

client = chromadb.PersistentClient(path=r"E:\llm-study-plan\rag\chroma_db")

# 用 get_collection（不是 get_or_create）：
#   名字写错会【直接报错】，问题立刻暴露；
#   而 get_or_create 会默默创建一个空集合 → 查询返回 0 条，还找不到原因
col = client.get_collection('nndl_book')


# ============================================================
# ② search() —— 检索：问题 → 最相关的 top_k 个知识块
# ============================================================
def search(question, top_k=5):
    """把问题编码成向量，在向量库里找最相似的 top_k 个块。

    参数：
        question : str  用户的问题
        top_k    : int  返回几条（默认 5）

    返回（3 个平行列表，第 i 项互相对应）：
        docs  : list[str]           命中的原文
        metas : list[dict]          元信息 {chapter, title, chunk_index}
        dists : list[float]         距离（越小 = 越相似）
    """
    # ---- 编码问题 ----
    # ⚠️ 必须用【和入库时完全相同的模型 + 相同配置】！
    #    数据可以不同（问题是 1 条、库是 1023 条），但"尺子"必须相同。
    # normalize_embeddings=True：归一化后"点积"就等于"余弦相似度"
    q_vec = model.encode([question], normalize_embeddings=True)

    # 向量是 numpy 数组，Chroma 只认普通 Python 列表 → tolist()
    q_vec1 = q_vec.tolist()

    # ---- 检索 ----
    # ⚠️ 必须传 query_embeddings（自己编码好的向量），不能传 query_texts！
    #    传 query_texts 的话，Chroma 会用【它自带的英文模型】编码，
    #    和你的 BGE 向量空间完全不匹配 → 结果全乱，而且【不报错】。
    res = col.query(query_embeddings=q_vec1, n_results=top_k)

    # ---- 解析返回结构 ----
    # ⚠️ query() 返回的是【两层嵌套列表】：
    #       [ ['文本1', '文本2', ...] ]
    #         ↑ 外层：第几个查询（query 支持一次问多个问题）
    #            ↑ 内层：那一个问题的 top_k 条结果
    #    我们只问了 1 个问题 → 取 [0]
    docs = res['documents'][0]
    metas = res['metadatas'][0]
    dists = res['distances'][0]

    # 💡 原来的这里有个 for 循环算 ch/title，但算完从没被用到 → 已删掉
    return docs, metas, dists


# ============================================================
# ③ build_prompt() —— 拼提示词（RAG 质量的关键）
# ============================================================
def build_prompt(docs, metas, question):
    """把「检索到的资料」和「用户问题」拼成提示词。

    返回： (user_prompt, system_prompt)  ← ⚠️ 是【两个值】，调用时要解包！

    拼出来的 user_prompt 长这样：
        【资料】
        [1]（第8章 注意力机制与外部记忆）
            以作用在内部的意识（思考、回忆等）...
        [2]（第8章 注意力机制与外部记忆）
            指自上而下的聚焦式注意力...

        【问题】
        什么是注意力机制
    """
    # ---- 把每一段资料拼成一行 ----
    lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        # ⚠️ f-string 里取字典用【双引号】，因为外层是单引号
        #    （Python 3.12 之前，大括号里不能用和外层相同的引号）
        # [1] 编号 → 让模型能引用；第X章 → 让答案能溯源
        lines.append(f'[{i}]（第{meta["chapter"]}章 {meta["title"]}）\n{doc}')

    # 多段之间用【空行】隔开，模型更好读（比单换行清晰）
    context = '\n\n'.join(lines)

    user_prompt = f'【资料】\n{context}\n\n【问题】\n{question}'

    # ---- system 提示词：定规矩（决定会不会瞎编）----
    system_prompt = (
        "你是一个严谨的助手。请【只根据提供的资料】回答问题。\n"
        "· 资料里没有的内容，明确说\"资料中没有提到\"，绝对不要编造\n"
        "· 回答时标注依据来自哪一章"
    )
    # 💡 这三句话是 RAG 防幻觉的核心：
    #    ① "只根据资料" → 限制知识来源
    #    ② "没有就说没有" → 给它一个"拒答"的出口
    #    ③ "标注依据" → 方便你核对（也是我们后面做量化评估的基础）

    return user_prompt, system_prompt


# ============================================================
# ④ rag_answer() —— 主流程：检索 → 拼 → 生成 → 返回
# ============================================================
def rag_answer(question):
    """输入问题，返回 (回答, 出处列表)。

    返回：
        answer  : str        模型的回答（可能为 None，如果 API 调用失败）
        sources : list[str]  出处，如 ['第8章 注意力机制与外部记忆']
                             超范围问题时返回空列表 []
    """
    # ---- 1. 检索 ----
    docs, metas, dists = search(question)

    # ---- 2. 超范围拦截（⭐ 三层防御里的"代码层"）----
    # 思路：如果"最相关的那段"距离都很远，说明教材里根本没有相关内容。
    #       这时【用代码直接拦住】，而不是求模型"听话"。
    #
    # 阈值怎么定？用实测数据标定：
    #   · 教材内问题：距离约 0.47 ~ 0.50（很相关）
    #   · 教材外问题：距离通常 > 1.0（毫不相关）
    #   → 取 0.9 作为分界（在两者之间，留了余量）
    if min(dists) > 0.9:
        return '这个问题不在教材范围内', []

    # ---- 3. 拼提示词 ----
    # ⚠️ 关键：build_prompt 返回【两个值】，必须解包成两个变量！
    #    写成 `x = build_prompt(...)` 的话，x 会是 tuple，
    #    再传给 ask_llm 就会让 API 报 422（invalid type: string）
    user_prompt, system_prompt = build_prompt(docs, metas, question)

    # ---- 4. 调大模型生成 ----
    # 显式写 system_prompt= 而不是按位置传，可读性更好
    answer = ask_llm(user_prompt, system_prompt=system_prompt)

    # ---- 5. 拼出处 ----
    # 从元信息里提取"第X章 标题"
    # dict.fromkeys(...) 的作用：**保序去重**
    #   （top-5 里可能有 3 段都来自第 8 章，去重后只显示一次）
    sources = list(dict.fromkeys(
        f"第{m['chapter']}章 {m['title']}" for m in metas
    ))

    return answer, sources


# ============================================================
# ⑤ 测试入口（只有直接运行本文件时才执行；被 import 时跳过）
# ============================================================
if __name__ == '__main__':
    question = input('你的问题是什么：')
    answer, sources = rag_answer(question)

    print('\n【回答】')
    print(answer)

    # 超范围问题返回的 sources 是空列表 → 不打印"依据"
    if sources:
        print('\n【依据】')
        for s in sources:
            print('  ·', s)

    # ============================================================
    # 📌 测试建议（自己手动试这几类问题）：
    # ============================================================
    # 1. 教材内的问题
    #      "什么是注意力机制"        → 应命中第8章，带出处
    #      "循环神经网络有什么问题"   → 应命中第6章
    #
    # 2. ⭐ 教材外的问题（验证"超范围拦截"）
    #      "红烧肉怎么做"            → 应回答"这个问题不在教材范围内"
    #      "什么是量子神经网络火锅"   → 同上，不能编造
    #
    # 3. ⭐ 忠实度测试（注入假信息）
    #      手动改掉某段资料里的一个数字，再问 → 看模型是否跟着你的资料走
    #
    # 达标标准：
    #   · 教材内问题 → 答得对 + 有出处
    #   · 教材外问题 → 明确拒答，不瞎编
