"""
MemU 记忆层集成演示脚本

本脚本展示了如何使用 MemU (https://memu.pro) 为 AI 应用添加长期记忆功能。
主要实现了以下核心流程：
1. 初始化 MemU 客户端
2. 检索用户历史记忆
3. 将记忆转换为系统提示词
4. 使用记忆上下文进行 AI 对话
5. 保存新对话到记忆系统

这是 MemU 记忆层的完整工作流程演示。
"""

import os
import time
from memu import MemuClient

# ============================================================================
# 第一部分：初始化 MemU 客户端
# ============================================================================
# 功能：创建与 MemU API 服务的连接
# 说明：
#   - base_url: MemU 服务的 API 端点地址
#   - api_key: 从环境变量中读取的 API 密钥，用于身份验证
#   注意：需要在环境变量中设置 MEMU_API_KEY
memu_client = MemuClient(
    base_url="https://api.memu.so",
    api_key=os.getenv("MEMU_API_KEY")
)

print("✅ MemU client initialized successfully!")

# ============================================================================
# 第二部分：检索用户记忆
# ============================================================================
def retrieve_user_memories(user_id, agent_id):
    """
    从 MemU 服务中检索指定用户和代理的历史记忆
    
    参数:
        user_id (str): 用户的唯一标识符，用于区分不同用户
        agent_id (str): AI 代理的唯一标识符，用于区分不同代理实例
    
    返回:
        dict: 包含记忆分类的字典，格式为 {'categories': [...]}
              每个分类包含名称、摘要等信息
              如果检索失败，返回空字典 {'categories': []}
    
    功能说明:
        - 调用 MemU 的 retrieve_default_categories API
        - 获取该用户在该代理下的所有默认记忆分类
        - 这些记忆包含了之前对话中提取的关键信息
        - 用于在对话开始前"唤醒"AI 对用户的了解
    """
    try:
        # 调用 MemU API 获取用户的记忆分类
        memories = memu_client.retrieve_default_categories(
            user_id=user_id,
            agent_id=agent_id
        )
        
        print('📚 Retrieved memories:', memories)
        return memories
    except Exception as error:
        # 如果检索失败（如网络错误、API 错误等），返回空记忆结构
        print('❌ Error retrieving memories:', error)
        return {'categories': []}

# 示例用法：定义用户和代理 ID，并检索记忆
user_id = "user_123"
agent_id = "assistant_001"
user_memories = retrieve_user_memories(user_id, agent_id)

# ============================================================================
# 第三部分：构建系统提示词
# ============================================================================
def build_system_prompt(memories):
    """
    将检索到的记忆转换为 LLM 可理解的系统提示词
    
    参数:
        memories (dict): 从 retrieve_user_memories 获取的记忆字典
    
    返回:
        str: 包含用户信息的完整系统提示词
    
    功能说明:
        - 将结构化的记忆数据转换为自然语言格式
        - 每个记忆分类的摘要会被添加到提示词中
        - 这样 LLM 在生成回复时就能参考用户的历史信息
        - 实现个性化的对话体验
    
    提示词格式示例:
        "You are a helpful AI assistant. Here's what you know about the user:
        
        **Profile:** 用户喜欢编程和人工智能...
        **Preferences:** 用户偏好简洁的回答...
        "
    """
    # 基础系统提示词
    system_prompt = "You are a helpful AI assistant. Here's what you know about the user:\n\n"
    
    # 遍历所有记忆分类，将摘要信息添加到提示词中
    if memories and 'categories' in memories:
        for category in memories['categories']:
            # 只添加有摘要的分类（摘要包含该分类的核心信息）
            if category.get('summary'):
                system_prompt += f"**{category['name']}:** {category['summary']}\n\n"
    
    return system_prompt

# 示例用法：构建包含记忆的系统提示词
system_prompt = build_system_prompt(user_memories)
print('🧠 System prompt built:', system_prompt)

# ============================================================================
# 第四部分：带记忆上下文的 AI 对话
# ============================================================================
import openai

def chat_with_ai(system_prompt, user_message, conversation_history=None):
    """
    使用包含记忆上下文的系统提示词与 AI 进行对话
    
    参数:
        system_prompt (str): 包含用户记忆的系统提示词
        user_message (str): 用户当前的消息
        conversation_history (list, optional): 本次会话的历史对话记录
                                              格式: [{"role": "user", "content": "..."}, ...]
    
    返回:
        str: AI 生成的回复内容
    
    功能说明:
        - 构建完整的消息列表：系统提示 + 历史对话 + 当前用户消息
        - 调用 OpenAI API 生成回复（可替换为其他 LLM）
        - AI 在生成回复时会参考系统提示词中的用户记忆
        - 实现基于历史记忆的个性化对话
    
    消息结构:
        [
            {"role": "system", "content": "包含记忆的系统提示..."},
            {"role": "user", "content": "历史消息1"},
            {"role": "assistant", "content": "历史回复1"},
            {"role": "user", "content": "当前消息"}
        ]
    """
    # 如果没有提供历史对话，初始化为空列表
    if conversation_history is None:
        conversation_history = []
        
    try:
        # 构建消息列表：系统提示词 + 历史对话 + 当前用户消息
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        # 调用 OpenAI API 生成回复
        # 注意：这里使用的是旧版 API，新版本应使用 openai.ChatCompletion.create
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )

        # 提取 AI 的回复内容
        ai_response = response.choices[0].message.content
        print('🤖 AI Response:', ai_response)
        
        return ai_response
    except Exception as error:
        # 如果 API 调用失败，返回错误提示
        print('❌ Error with AI:', error)
        return "Sorry, I encountered an error."

# 示例用法：进行一次对话
user_message = "Hi, how are you today?"
ai_response = chat_with_ai(system_prompt, user_message)

# ============================================================================
# 第五部分：保存对话到记忆系统
# ============================================================================
import time

def save_conversation(conversation, user_id, agent_id):
    """
    将对话保存到 MemU 记忆系统中，用于长期记忆存储
    
    参数:
        conversation (list): 对话记录列表，格式为：
                            [
                                {"role": "user", "content": "用户消息"},
                                {"role": "assistant", "content": "AI回复"}
                            ]
        user_id (str): 用户的唯一标识符
        agent_id (str): AI 代理的唯一标识符
    
    功能说明:
        - 调用 MemU 的 memorize_conversation API 保存对话
        - MemU 会异步处理对话，提取关键信息并更新记忆
        - 返回任务 ID，用于跟踪处理状态
        - 等待任务完成，确保记忆已成功保存
    
    记忆更新流程:
        1. 发送对话到 MemU API
        2. MemU 分析对话内容，提取关键信息
        3. 更新用户的记忆分类（如偏好、个人信息等）
        4. 下次调用 retrieve_user_memories 时会包含新信息
    """
    try:
        # 调用 MemU API 保存对话
        response = memu_client.memorize_conversation(
            conversation=conversation,
            user_id=user_id,
            user_name="Demo User",  # 用户显示名称
            agent_id=agent_id,
            agent_name="AI Assistant"  # 代理显示名称
        )

        print('💾 Conversation saved! Task ID:', response.task_id)
        
        # 等待 MemU 完成记忆处理（异步任务）
        wait_for_completion(response.task_id)
        
    except Exception as error:
        print('❌ Error saving conversation:', error)

def wait_for_completion(task_id):
    """
    轮询检查 MemU 记忆处理任务的状态，直到完成
    
    参数:
        task_id (str): 从 save_conversation 返回的任务 ID
    
    功能说明:
        - 定期查询任务状态（每 2 秒一次）
        - 当任务状态为 SUCCESS、FAILURE 或 REVOKED 时停止轮询
        - 确保记忆处理完成后再继续后续操作
    
    任务状态说明:
        - SUCCESS: 记忆处理成功完成
        - FAILURE: 处理失败
        - REVOKED: 任务被取消
        - 其他状态: 仍在处理中，继续等待
    """
    while True:
        try:
            # 查询任务当前状态
            status = memu_client.get_task_status(task_id)
            print('📊 Task status:', status.status)
            
            # 如果任务已完成（成功、失败或取消），退出循环
            if status.status in ['SUCCESS', 'FAILURE', 'REVOKED']:
                break
            
            # 等待 2 秒后再次检查状态
            time.sleep(2)
        except Exception as error:
            print('❌ Error checking task status:', error)
            break

# ============================================================================
# 示例：构建对话上下文并保存
# ============================================================================
# 将用户消息和 AI 回复组织成对话格式
conversation_context = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": ai_response}
]

# 保存对话到记忆系统
save_conversation(conversation_context, user_id, agent_id)