from openai import OpenAI
import os

# 初始化 OpenAI 客户端
# 使用阿里云 DashScope 的兼容模式 API
client = OpenAI(api_key="sk-882e296067b744289acf27e6e20f3ec0",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化变量用于存储模型的推理内容和最终回答内容
reasoning_content = ""  # 存储模型的推理过程
content = ""  # 存储模型的最终回答

# 标记是否已经开始输出最终回答
is_answering = False

# 创建聊天完成请求
# 使用 qwq-32b 模型，并启用流式输出
completion = client.chat.completions.create(
    model="qwq-32b",
    messages=[
        {"role": "user", "content": "9.9和9.11 哪个更大？"}
    ],
    stream=True,
    # 取消下面的注释可以在最后一个数据块中返回 token 使用情况
    # stream_options={
    #     "include_usage": True
    # }
)

# 打印推理内容的分隔标题
print("\n" + "=" * 20 + "reasoning content" + "=" * 20 + "\n")

# 处理流式响应的每个数据块
for chunk in completion:
    # 如果 chunk.choices 为空，打印使用情况统计
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        # 获取当前数据块中的增量内容
        delta = chunk.choices[0].delta
        
        # 处理推理内容部分
        # QwQ 模型的特殊功能：可以输出推理过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            print(delta.reasoning_content, end='', flush=True)  # 实时打印推理内容
            reasoning_content += delta.reasoning_content  # 累积推理内容
        else:
            # 处理最终回答内容部分
            # 当首次收到回答内容时，打印分隔标题
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "content" + "=" * 20 + "\n")
                is_answering = True
                
            # 打印回答内容
            print(delta.content, end='', flush=True)  # 实时打印回答内容
            content += delta.content  # 累积回答内容