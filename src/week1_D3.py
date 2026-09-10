"""
week1_D3.py —— 第 1 周 D3 任务：PDF 文本提取
------------------------------------------------
作用：把《神经网络与深度学习》（邱锡鹏）整本 PDF 的文字提取出来，
      保存成一个 txt 文件，供后续切分 / 向量化 / 检索使用。

运行方式：py week1_D3.py
输出：E:\llm-study-plan\rag\data\book.txt
"""

# ========== 1. 导入库 ==========
# pypdf 是用来读取 PDF 的库（已安装）
from pypdf import PdfReader

# ========== 2. 设置路径 ==========
# 用 r"..." 表示原始字符串，避免 Windows 路径里的反斜杠 \ 被当成转义符
pdf_path = r"E:\llm-study-plan\local\data\神经网络与深度学习 (邱锡鹏) (z-library.sk, 1lib.sk, z-lib.sk.pdf"

# 提取出的全文要保存到哪里
out_path = r"E:\llm-study-plan\rag\data\book.txt"

# ========== 3. 打开 PDF ==========
# PdfReader 会把整个 PDF 加载进内存，reader.pages 就是所有页面的列表
reader = PdfReader(pdf_path)
print("总页数：", len(reader.pages))

# ========== 4. 逐页提取文字 ==========
# 用一个列表 all_text 收集每一页的文字，最后再一次性合并（比反复拼接字符串快）
all_text = []
for page in reader.pages:
    # extract_text() 返回这一页的文字内容
    text = page.extract_text()

    # 注意：如果某一页是纯图片、没有文字，extract_text() 可能返回 None
    # 这里做个判断，避免后面合并时因为 None 报错
    if text:
        all_text.append(text)

# 把每一页的文字用换行符连接成完整的一本书
full_text = "\n".join(all_text)

# ========== 5. 保存成 txt 文件 ==========
# encoding="utf-8" 必须指定，否则中文会乱码
with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("已保存, 总字符数:", len(full_text))

# ========== 6. 检查结果 ==========
# 打印前几行，确认中文没有乱码、内容正常
print("\n--- 前几行预览 ---")
print("\n".join(full_text.split("\n")[:10]))
