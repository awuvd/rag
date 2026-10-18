"""
week3_D15.py —— T6：跑通大模型 API（封装成可复用函数）
========================================================
作用：封装一个 ask_llm(prompt) 函数，输入一句话、返回模型的回答。
      后面 T8（检索+生成串联）会 import 它来生成答案。

运行：py week3_D15.py
输入：.env 文件里的 DEEPSEEK_API_KEY；运行后终端里输入你的问题
输出：终端打印模型的回答

【本文件踩过的 2 个坑（都是典型坑）】
  坑 1：.env 文件名被 Windows 记事本自动加成了 .env.txt
         → dotenv 找不到 → os.getenv 返回 None → 报 "Missing credentials"
         （教训：Windows 上建"点开头"的文件要格外小心扩展名）
  坑 2：模型名要用【API 名】deepseek-flash，
         而不是【版本名】DeepSeek-V4.1-Flash
         （版本名是给人看的，API 名才是给程序用的）

【设计说明】
  · client 在模块顶层只建一次 → 之后每次调用都复用，不用反复创建
  · 密钥从 .env 读，绝不写死在代码里（代码要 push 到 GitHub！）
  · 函数只 return 结果、不 print → 这样 T8 拿到的是干净的字符串
    需要看耗时/token 时，传 verbose=True
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# ① 配置（模块加载时执行一次）
# ============================================================

# 从 .env 读取密钥。
# 不传路径也能读到：dotenv 会从【本脚本所在目录】逐层往上找 .env
#    src/ → 没有 → rag/ → 找到 ✅
load_dotenv()

API_KEY = os.getenv('DEEPSEEK_API_KEY')

# ⚠️ 提前检查，给出清楚的报错。
# 否则 key 是 None 时会一路传到 API 调用处，报出很难懂的 "Missing credentials"
if not API_KEY:
    raise ValueError(
        '没找到 DEEPSEEK_API_KEY。\n'
        '  请检查：① rag/.env 文件是否存在（注意别被记事本加成 .env.txt）\n'
        '         ② 内容格式是否为  DEEPSEEK_API_KEY=sk-xxxx（等号两边不要空格、不要引号）'
    )

BASE_URL = 'https://api.deepseek.com'   # 请求发到哪家（换服务商只改这一行）
MODEL = 'deepseek-flash'                # ⚠️ API 名（不是版本名 DeepSeek-V4.1-Flash）

# client 只创建一次，后续所有调用复用它
# （每次调用都新建 client 也能跑，但没必要——它内部会维护连接池）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60,          # 超时 60 秒，防止网络卡住时无限等待
)


# ============================================================
# ② 核心函数：问大模型
# ============================================================
def ask_llm(prompt, system_prompt=None, temperature=0.3, verbose=False):
    """
    调用大模型，返回它的回答文本。

    参数：
        prompt        : str  你的问题（必填）
        system_prompt : str  系统提示词，用来设定"人设/规则"。
                             T8 里会用它写"只根据以下资料回答"。
                             不传则只发用户消息。
        temperature   : float 随机性。0 = 最确定，1 = 最有创意。
                             ⚠️ RAG 场景要"基于资料、别发挥" → 建议 0 ~ 0.3
        verbose       : bool 是否打印耗时和 token 用量（调试用）

    返回：
        str  模型的回答；
             出错时返回 None（并打印错误原因）
    """
    # ---- 组装 messages（一个列表，每项是一个 dict）----
    messages = []
    if system_prompt:
        # system 消息用来"定规矩"，比如"你只能根据给定资料回答"
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # ---- 发请求（用 try 包住，避免网络/额度问题直接崩掉）----
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
    except Exception as e:
        # 常见错误及原因（方便排查）：
        #   401 AuthenticationError  → key 错/失效
        #   402 / 429                → 余额不足 / 请求太频繁
        #   APITimeoutError          → 网络慢或超时
        print(f'[ask_llm 调用失败] {type(e).__name__}: {e}')
        return None
    cost = time.time() - t0

    # ---- 取回答文本 ----
    # 返回结构是一层层嵌套的，要剥到最里面：
    #   resp → .choices[0] → .message → .content
    answer = resp.choices[0].message.content

    # ---- 调试信息（可选）----
    if verbose:
        u = resp.usage        # token 用量，体现"成本意识"
        print(f'[耗时 {cost:.2f}s | prompt {u.prompt_tokens} + '
              f'completion {u.completion_tokens} = {u.total_tokens} tokens]')

    return answer


# ============================================================
# ③ 交互式测试（只有直接运行本文件时才执行；被 import 时不会执行）
# ============================================================
if __name__ == '__main__':
    print('=' * 50)
    print('T6 测试：问 DeepSeek 一个问题')
    print('（直接回车则跑内置的 3 个测试）')
    print('=' * 50)

    question = input('\n你有什么问题：').strip()

    if question:
        # 你输入了问题 → 问它
        print('\n回答：')
        print(ask_llm(question, system_prompt='你是一个严谨的助手。', verbose=True))

    else:
        # 没输入 → 跑内置测试
        print('\n--- 测试 1：普通问题 ---')
        print(ask_llm('用一句话解释什么是神经网络', verbose=True))

        print('\n--- 测试 2：带 system 提示（模拟 T8 的 RAG 场景）---')
        print(ask_llm(
            prompt='什么是注意力机制？',
            system_prompt='你是一个严谨的助手，只用一句话回答。',
            verbose=True,
        ))

        print('\n--- 测试 3：问不存在的概念（看会不会瞎编）---')

        print(ask_llm(
            prompt='什么时候上火星"？',
            system_prompt='只根据你的知识回答；如果这个概念不存在，就直接说不存在。',
            verbose=True,
        ))
        print('\n💡 测试 3 是 T8 要重点验证的能力：')
        print('   教材里没有的内容，模型应该说"没有"而不是编造。')
