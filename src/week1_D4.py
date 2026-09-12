"""
week1_D4.py —— 第 1 周 D4 任务：文档切分（把整本书切成"知识块"）
=================================================================
作用：读入 D3 提取的全文（book.txt），按章节切成小块（chunk），
      给每块加上元信息（id / 所属部分 / 标题 / 块序号），
      存成 JSON，供后续"向量化(D6)"和"检索(D7)"使用。

运行方式：py week1_D4.py

输入：data/book.txt        ← D3 的产物（444 页提取出的全文）
输出：data/chapters.json   ← 各"部分"的位置信息（中间产物）
      data/chunks.json     ← 知识块（RAG 的核心数据）

核心思路：
  ① 用正则找到 15 个"章节标题"（并排除目录、正文提及等假货）
  ② 再找 5 个"附录"和最后的"参考文献"（否则它们会被算进第 15 章）
  ③ 把所有"部分"按行号排序，用"相邻两段的行号"夹出每段正文
  ④ 每段正文再按 500 字切块，块间重叠 50 字（防止句子被切断）
  ⑤ 每块带上元信息，最后存成 JSON

【本版修复】原版只识别"第 N 章"，导致第 15 章之后的内容
（附录 A~E 线性代数/微积分/数学优化/概率论/信息论 + 参考文献）
全部被误标为"第15章 序列生成模型"，会造成溯源错误。现已修正。
"""

import json
import re

# ============================================================
# 0. 读入 D3 提取的全文，按行拆成列表
# ============================================================
# 路径前必须加 r（原始字符串），否则 \r、\b 等会被当成转义符
with open(r"E:\llm-study-plan\rag\data\book.txt", "r", encoding="utf-8") as bk:
    book = bk.read()
lines = book.split("\n")        # 整本书 → 一行一个元素的列表

# ============================================================
# 1. 定位所有"部分"（章 / 附录 / 参考文献）
# ============================================================
# sections 里每个元素代表"一段独立的正文"，按行号排好序后用于切分
sections = []

# ---- 1.1 找"第 N 章"（正文的 15 章）----
# 正则要求：
#   ^第\s+(\d+)\s+章\s+   —— 行首 + "第 N 章"（数字前后必须有空格）
#   ([^0-9]+)$            —— 后面跟"不含数字"的标题，直到行尾（排除目录里的页码）
pattern_ch = re.compile(r"^第\s+(\d+)\s+章\s+([^0-9]+)$")

x = 1               # "期待的章号"：必须是 1、2、3……依次出现
for i, line in enumerate(lines):
    m = pattern_ch.search(line)
    if m:
        # 只接受"正好是下一个该出现的章号"——用"章节递增"这个规律，
        # 自动跳过"正文里提到第X章"之类的假货（它们的章号会突然变大或回退）
        if int(m.group(1)) == x:
            sections.append({
                "key": f"ch{int(m.group(1)):02d}",   # id 前缀，如 ch08
                "num": int(m.group(1)),              # 编号：章 = 1~15
                "title": m.group(2),                 # 标题
                "line_no": i,                        # ⭐ 标题所在行号（切分要用）
            })
            x += 1

# ---- 1.2 找"附录 A~E"----
# 正则：^附\s*录\s*([A-E])\s+  +  ([^0-9]+)$
#   ([A-E])   —— 附录字母（只认 A~E）
#   ([^0-9]+)$ —— 不含数字的标题（自动排除目录里带页码的"附录 A 线性代数 384"）
pattern_app = re.compile(r"^附\s*录\s*([A-E])\s+([^0-9]+)$")

for i, line in enumerate(lines):
    m = pattern_app.search(line)
    if m:
        letter = m.group(1)                      # A / B / C / D / E
        sections.append({
            "key": f"app{letter}",               # id 前缀，如 appA
            "num": 100 + (ord(letter) - ord("A") + 1),   # 编号：附录 = 101~105
            "title": f"附录 {letter} {m.group(2)}",       # 如"附录 A 线性代数"
            "line_no": i,
        })

# ---- 1.3 找"参考文献"（只取最后一个）----
# ⚠️ 注意：每章末尾都有一个小节的"参考文献"（全书共 16 处同名行），
#    只有最后那一处（行号最大）才是全书的总参考文献。
pattern_ref = re.compile(r"^参考文献\s*$")
ref_lines = [i for i, line in enumerate(lines) if pattern_ref.search(line)]

if ref_lines:
    sections.append({
        "key": "ref",                            # id 前缀
        "num": 200,                              # 编号：参考文献 = 200
        "title": "参考文献",
        "line_no": ref_lines[-1],                # 取最后一个
    })

# ---- 1.4 按行号排序（顺序 = 在书里出现的先后）----
sections.sort(key=lambda s: s["line_no"])

# ============================================================
# 2. 段内切分：把每段正文切成 ~500 字的块
# ============================================================
size = 500          # 每块最大字符数
overlap = 50        # 相邻块重叠字符数（防止一句话被切断）
step = size - overlap   # 每次向后滑动 450 字

chunks = []         # 最终产物：所有知识块

for index, sec in enumerate(sections):
    start = sec["line_no"]        # 本段标题所在行

    # "向后看一位"：下一段的标题行 = 本段正文的结束边界
    # 最后一段没有"下一段"，所以直接切到文件末尾（否则索引会越界报错）
    if index + 1 < len(sections):
        end = sections[index + 1]["line_no"]
    else:
        end = len(lines)

    # 把本段的"多行"拼成"一整段字符串"（+1 是为了跳过标题行本身）
    text = "\n".join(lines[start + 1 : end])

    # 滑动窗口切块：
    #   range 生成"每块的起始位置" 0, 450, 900...
    #   enumerate(..., start=1) 再配一个"块编号" j（从 1 开始）
    for j, pos in enumerate(range(0, len(text), step), start=1):
        piece = text[pos : pos + size]

        # 过滤"碎块"：滑动到最后常常剩下一小截（如 6 个字符），
        # 这种块没有检索价值，反而会污染检索结果，直接跳过
        if len(piece) < 50:
            continue

        chunks.append({
            # id 必须全局唯一：前缀(ch08/appA/ref) + 块号补 3 位 → "ch08_012"
            # （补零是为了排序整齐，否则 ch1_10 会排在 ch1_2 前面）
            "id": f'{sec["key"]}_{j:03d}',
            "chapter": sec["num"],      # 编号（章1-15 / 附录101-105 / 参考文献200）
            "title": sec["title"],      # 标题（用于溯源）
            "chunk_index": j,           # 是本段的第几块
            "text": piece,              # ⭐ 这一块的正文（检索的真正内容）
        })

# ============================================================
# 3. 保存成 JSON 文件
# ============================================================
# ensure_ascii=False —— 必须写！否则中文会变成 \u7b2c 这种乱码
# indent=2           —— 加缩进换行，人能看懂（不加则挤成一整行）
with open(r"E:\llm-study-plan\rag\data\chapters.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=2)

with open(r"E:\llm-study-plan\rag\data\chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# ============================================================
# 4. 读回来验证（"写成功"≠"内容对"，要读一遍确认）
# ============================================================
with open(r"E:\llm-study-plan\rag\data\chunks.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)

print("部分数（15章 + 5附录 + 1参考文献）:", len(sections))
print("总块数:", len(loaded))
print("id 是否唯一:", len(set(c["id"] for c in loaded)) == len(loaded))
print()
print("各部分块数:")
for sec in sections:
    n = sum(1 for c in loaded if c["id"].startswith(sec["key"] + "_"))
    print(f'  {sec["key"]:<6} {sec["title"]:<20} {n:>3} 块')
print()
print("前 2 块:")
for c in loaded[:2]:
    print(f'  {c["id"]} | {c["title"]} | 长度={len(c["text"])}')
