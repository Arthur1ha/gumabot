import logging
import os
import asyncio
import sys
import requests
from dotenv import load_dotenv
from memu import MemuClient
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    inference,
    JobContext,
    room_io,
)
from livekit.plugins import silero
from livekit.plugins import openai
from livekit.plugins import deepgram

logger = logging.getLogger("guma-agent")

#OPENAI-API
load_dotenv(override=True)
api_key = os.getenv("OPENAI_APIKEY")
base_url = os.getenv("BASE_URL")
memu_api_key = os.getenv("MEMU_API_KEY")  # MemU API 密钥
MEMU_API_BASE = "https://api.memu.so/api/v1"
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")  # Deepgram API 密钥
# print(f"OPENAI_APIKEY: {api_key}")
# print(f"BASE_URL: {base_url}")

# 初始化 MemU 客户端
if memu_api_key:
    memu_client = MemuClient(
        base_url="https://api.memu.so",
        api_key=memu_api_key
    )
    logger.info("[MEMU] ✅ MemU 客户端初始化成功")
else:
    memu_client = None
    logger.warning("[MEMU] ⚠️  MemU API 密钥未设置，记忆功能将被禁用")

# ============================================================================
# MemU 记忆层功能函数
# ============================================================================
def retrieve_user_memories(user_id: str, agent_id: str):
    """
    从 MemU 检索用户的历史记忆
    参数:
        user_id: 用户唯一标识符
        agent_id: 代理唯一标识符
    返回:
        dict: 包含记忆分类的字典，如果失败则返回空字典
    """
    if not memu_client:
        logger.warning("[MEMU] ⚠️  客户端未初始化，跳过记忆检索")
        return {'categories': []}
    
    try:
        logger.info("[MEMU] 🔍 开始检索用户记忆...")
        logger.info(f"[MEMU]   用户 ID: {user_id}")
        logger.info(f"[MEMU]   代理 ID: {agent_id}")
        
        memories = memu_client.retrieve_default_categories(
            user_id=user_id,
            agent_id=agent_id
        )
        categories = extract_categories(memories)
        if categories:
            category_count = len(categories)
            logger.info(f"[MEMU] ✅ 成功检索到 {category_count} 个记忆分类")
            for idx, category in enumerate(categories, 1):
                category_name = extract_value(category, 'name') or '未知分类'
                summary_val = extract_value(category, 'summary') or ''
                summary_preview = (summary_val[:50] + '...') if summary_val else '无摘要'
                logger.info(f"[MEMU]   分类 {idx}: {category_name} (摘要: {summary_preview})")
        else:
            logger.info("[MEMU] ℹ️  未找到历史记忆（新用户或首次对话）")
        
        return memories
    except Exception as error:
        logger.error(f"[MEMU] ❌ 检索记忆时发生错误: {error}")
        logger.error(f"[MEMU]   错误类型: {type(error).__name__}")
        return {'categories': []}


def build_system_prompt_with_memories(base_instructions: str, memories: dict) -> str:
    """
    将记忆信息整合到系统提示词中
    
    参数:
        base_instructions: 基础系统提示词
        memories: 从 MemU 检索的记忆字典
    
    返回:
        str: 包含记忆信息的完整系统提示词
    """
    system_prompt = base_instructions
    logger.info(f"system_prompt:{system_prompt}")
    
    categories = extract_categories(memories)
    if categories:
        memory_context = "\n\n以下是关于用户的信息：\n\n"
        added_categories = 0
        for category in categories:
            category_summary = extract_value(category, 'summary')
            if category_summary:
                category_name = extract_value(category, 'name') or '未知分类'
                memory_context += f"**{category_name}:** {category_summary}\n\n"
                added_categories += 1
        if added_categories > 0:
            system_prompt += memory_context
            logger.info(f"[MEMU] 📝 已将 {added_categories} 个记忆分类添加到系统提示词")
            logger.info(f"new_system_prompt:{system_prompt}")
        else:
            logger.info("[MEMU] ℹ️  记忆分类中没有可用摘要，未添加到提示词")
    else:
        logger.info("[MEMU] ℹ️  无记忆数据，使用基础系统提示词")
    
    return system_prompt

def extract_categories(memories):
    if not memories:
        return []
    if isinstance(memories, dict):
        return memories.get('categories', []) or []
    if hasattr(memories, 'categories'):
        return getattr(memories, 'categories') or []
    return []

def extract_value(item, key):
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

async def save_conversation_to_memu(conversation: list, user_id: str, agent_id: str, assistant=None, base_instructions: str=None):
    """
    异步保存对话到 MemU 记忆系统
    
    参数:
        conversation: 对话记录列表，格式为 [{"role": "user", "content": "..."}, ...]
        user_id: 用户唯一标识符
        agent_id: 代理唯一标识符
    """
    if not memu_client:
        logger.warning("[MEMU] ⚠️  客户端未初始化，跳过对话保存")
        return
    
    try:
        # 记录保存的对话信息
        message_count = len(conversation)
        logger.info("[MEMU] 💾 开始保存对话到 MemU...")
        logger.info(f"[MEMU]   用户 ID: {user_id}")
        logger.info(f"[MEMU]   代理 ID: {agent_id}")
        logger.info(f"[MEMU]   对话消息数: {message_count}")
        
        # 显示对话预览
        for idx, msg in enumerate(conversation): # 显示全部 
        # for idx, msg in enumerate(conversation[:4], 1):  # 只显示前4条
            role = msg.get('role', 'unknown')
            content_preview = msg.get('content', '')[:50] + '...' if len(msg.get('content', '')) > 50 else msg.get('content', '')
            logger.info(f"[MEMU]   消息 {idx} ({role}): {content_preview}")
        
        # 在后台线程中执行同步的 API 调用
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: memu_client.memorize_conversation(
                conversation=conversation,
                user_id=user_id,
                user_name="语音用户",
                agent_id=agent_id,
                agent_name="语音助手"
            )
        )
        
        # 记录保存结果
        task_id = getattr(response, 'task_id', 'N/A')
        logger.info(f"[MEMU] ✅ 对话已成功提交到 MemU")
        logger.info(f"[MEMU]   任务 ID: {task_id}")
        logger.info(f"[MEMU]   消息数: {message_count}")
        if assistant and base_instructions:
            asyncio.create_task(
                refresh_memories_and_update_prompt_with_task(
                    task_id,
                    user_id,
                    agent_id,
                    assistant,
                    base_instructions
                )
            )
        
    except Exception as error:
        logger.error(f"[MEMU] ❌ 保存对话时发生错误: {error}")
        logger.error(f"[MEMU]   错误类型: {type(error).__name__}")
        import traceback
        logger.error(f"[MEMU]   错误详情:\n{traceback.format_exc()}")

async def refresh_memories_and_update_prompt_with_task(task_id: str, user_id: str, agent_id: str, assistant, base_instructions: str, attempts: int = 10, delay: float = 2.0):
    if not memu_api_key:
        return await refresh_memories_and_update_prompt_fallback(user_id, agent_id, assistant, base_instructions, attempts, delay)
    for i in range(attempts):
        try:
            resp = requests.get(f"{MEMU_API_BASE}/memory/memorize/status/{task_id}", headers={"Authorization": f"Bearer {memu_api_key}"}, timeout=10)
            status = resp.json() if resp.ok else None
        except Exception:
            status = None
        if status == "completed":
            memories = retrieve_user_memories(user_id, agent_id)
            categories = extract_categories(memories)
            summaries = [c for c in categories if extract_value(c, 'summary')]
            if summaries:
                updated = build_system_prompt_with_memories(base_instructions, memories)
                assistant.instructions = updated
                logger.info(f"[MEMU] 🔧 更新后的系统提示词: {updated}")
                logger.info(f"[MEMU] ✅ 已刷新记忆并更新提示词，摘要条目: {len(summaries)}")
                return
        await asyncio.sleep(delay)
    return await refresh_memories_and_update_prompt_fallback(user_id, agent_id, assistant, base_instructions, 5, delay)

async def refresh_memories_and_update_prompt_fallback(user_id: str, agent_id: str, assistant, base_instructions: str, attempts: int, delay: float):
    for i in range(attempts):
        memories = retrieve_user_memories(user_id, agent_id)
        categories = extract_categories(memories)
        summaries = [c for c in categories if extract_value(c, 'summary')]
        if summaries:
            updated = build_system_prompt_with_memories(base_instructions, memories)
            assistant.instructions = updated
            logger.info(f"[MEMU] 🔧 更新后的系统提示词: {updated}")
            logger.info(f"[MEMU] ✅ 已刷新记忆并更新提示词，摘要条目: {len(summaries)}")
            return
        await asyncio.sleep(delay)
    logger.info("[MEMU] ℹ️  重试后仍无摘要，保持基础提示词")


# ============================================================================
# Assistant 类
# ============================================================================

class Assistant(Agent):
    def __init__(self, instructions: str = None) -> None:
        # base_instructions = """你是一个有用的语音人工智能助手。你热心地帮助用户解答他们的问题，从你广博的知识中提供信息。
        #     你的回答简洁明了，没有任何复杂的格式或标点符号，包括表情符号、星号或其他符号。你好奇、友善，而且有幽默感。将回复内容控制在20字以内"""
        
        # final_instructions = instructions if instructions else base_instructions
        # super().__init__(instructions=final_instructions)
        super().__init__(instructions=instructions)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # ========================================================================
    # MemU 记忆层集成：检索用户记忆
    # ========================================================================
    logger.info("=" * 60)
    logger.info("[MEMU] 🚀 开始 MemU 记忆层集成流程")
    logger.info("=" * 60)
    
    # 从房间信息或上下文获取用户标识（这里使用房间名作为示例）
    # 在实际应用中，您可能需要从 ctx.room 或其他来源获取真实的用户 ID
    user_id = ctx.room.name or "default_user"
    agent_id = "voice_assistant_001"
    
    logger.info(f"[MEMU] 📋 会话信息:")
    logger.info(f"[MEMU]   房间名: {ctx.room.name}")
    logger.info(f"[MEMU]   用户 ID: {user_id}")
    logger.info(f"[MEMU]   代理 ID: {agent_id}")
    
    # 检索用户历史记忆
    logger.info("")
    user_memories = retrieve_user_memories(user_id, agent_id)
    
    # 构建包含记忆的系统提示词
    logger.info("")
    logger.info("[MEMU] 🔨 构建系统提示词...")
    base_instructions = """你是一个有用的语音人工智能助手。你热心地帮助用户解答他们的问题，从你广博的知识中提供信息。
            你的回答简洁明了，没有任何复杂的格式或标点符号，包括表情符号、星号或其他符号。你好奇、友善，而且有幽默感。将回复内容控制在20字以内"""
    logger.info(f"[MEMU] 🧭 基础系统提示词: {base_instructions}")
    dynamic_instructions = build_system_prompt_with_memories(base_instructions, user_memories)
    logger.info(f"[MEMU] 🔧 动态系统提示词: {dynamic_instructions}")
    
    # 创建带记忆的 Assistant 实例
    logger.info("")
    logger.info("[MEMU] 🤖 创建带记忆的 Assistant 实例")
    assistant = Assistant(instructions=dynamic_instructions)
    logger.info("[MEMU] ✅ Assistant 创建完成")
    logger.info("=" * 60)
    
    # ========================================================================
    # 初始化 AgentSession
    # ========================================================================
    session = AgentSession(
        # stt=inference.STT(
        #     model="cartesia/ink-whisper", 
        #     language="zh"
        # ),
        # stt = openai.STT(
        #     model="gpt-4o-mini-transcribe",
        #     base_url=base_url, 
        #     api_key=api_key,
        #     language="zh"
        # ),

        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.4,
            api_key=deepgram_api_key
            # base_url = "https://api.deepgram.com/v1/listen"
        ),
        
        llm=openai.LLM.with_x_ai(
            model="grok-4.1", 
            base_url=base_url, 
            api_key=api_key
        ),
        tts = openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="用友好和对话的语气说话",
            base_url=base_url, 
            api_key=api_key
        ),
        turn_detection="vad",
        vad=silero.VAD.load(),
    )
    
    # ========================================================================
    # MemU 记忆层集成：监听对话并保存
    # ========================================================================
    conversation_buffer = []
    full_conversation = []
    turn_count = 0
    current_user_message = None
    current_agent_message = None
    
    # ========================================================================
    # AgentSession 官方事件（参考 https://docs.livekit.io/home/client/events/）
    # ========================================================================
    logger.info("")
    logger.info("[LiveKit] 📝 注册 AgentSession 事件监听器...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(state):
        """当代理状态变化（listening/thinking/speaking 等）时触发"""
        logger.info(f"[LiveKit] 🤖 Agent 状态变更 -> {state}")

    @session.on("user_state_changed")
    def on_user_state_changed(state):
        """当用户状态变化（listening/speaking 等）时触发"""
        logger.info(f"[LiveKit] 👤 User 状态变更 -> {state}")

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(payload):
        """当用户语音被转写为文本时触发"""
        text = getattr(payload, "text", None) or getattr(payload, "transcript", None) or str(payload)
        current_user_message and logger.debug("[MEMU] ⚠️ 覆盖上一条用户消息")
        logger.info(f"[LiveKit] 📝 用户转写文本: {text}")

    @session.on("conversation_item_added")
    def on_conversation_item_added(item):
        """当对话消息加入历史记录时触发"""
        
        # 检查 item 是否包含 'item' 属性，它是 ChatMessage 对象
        chat_message = getattr(item, 'item', None)
        if chat_message is None:
            logger.warning(f"[LiveKit] item 中不包含 ChatMessage 对象，无法处理")
            return

        # 获取消息的角色和内容
        role = getattr(chat_message, "role", "unknown")
        content = getattr(chat_message, "content", None)  # content 是一个列表

        # 如果 content 是列表，将其合并为一个字符串
        if isinstance(content, list):
            content = ''.join(content)  # 合并列表中的所有文本内容
        
        # 调试输出：检查 content 是否为 None
        if content is None:
            logger.warning(f"[LiveKit] 内容为空 (None)，无法处理此消息，role={role}")
        else:
            logger.info(f"[LiveKit] 💬 conversation_item_added -> role={role}, content={content}")  # 只显示前100个字符
            
            # 如果内容非空，进行解析并输出
            if isinstance(content, str) and content.strip():
                if role == "user":
                    # logger.info(f"用户提问: {content}")  # 显示用户问题
                    nonlocal current_user_message
                    current_user_message = content
                elif role == "assistant":
                    # logger.info(f"助手回答: {content}")  # 显示助手回答
                    nonlocal current_agent_message, conversation_buffer, turn_count
                    current_agent_message = content  # 赋值给 current_agent_message

                    # 如果用户提问存在，保存对话并清空当前消息
                    if current_user_message:
                        turn_count += 1
                        conversation_context = [
                            {"role": "user", "content": current_user_message},
                            {"role": "assistant", "content": current_agent_message}
                        ]
                        conversation_buffer.extend(conversation_context)
                        full_conversation.extend(conversation_context)
                        logger.debug(f"[LiveKit] 当前对话缓冲区内容: {conversation_buffer}")
                        current_user_message = None
                        current_agent_message = None

                        if len(conversation_buffer) >= 4:
                            logger.debug(f"[LiveKit] 缓冲区已满，准备保存对话到 MemU")
                            asyncio.create_task(
                                save_conversation_to_memu(
                                    full_conversation.copy(),
                                    user_id,
                                    agent_id,
                                    assistant,
                                    base_instructions
                                )
                            )
                            conversation_buffer.clear()
                            


    @session.on("close")
    def on_session_close(reason=None):
        """当 session 关闭时触发"""
        logger.info(f"[LiveKit] ⛔ AgentSession closed. reason={reason}")
        nonlocal conversation_buffer, full_conversation
        if len(full_conversation) >= 4:
            asyncio.create_task(
                save_conversation_to_memu(
                    full_conversation.copy(),
                    user_id,
                    agent_id,
                    assistant,
                    base_instructions
                )
            )
            conversation_buffer.clear()
            full_conversation.clear()
    
    # ========================================================================
    # 启动对话会话
    # ========================================================================
    logger.info("")
    logger.info("[MEMU] 🚀 启动 AgentSession...")
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
                # noise_cancellation=lambda params:krisp.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else krisp.BVC()
            ),
        ),
    )
    logger.info("[MEMU] ✅ AgentSession 启动完成")
    logger.info("[MEMU] 📡 现在正在监听对话事件...")
    
    # 调试：列出 session 对象的所有可用事件
    try:
        if hasattr(session, '_event_emitter'):
            emitter = session._event_emitter
            if hasattr(emitter, '_listeners'):
                events = list(emitter._listeners.keys())
                logger.info(f"[MEMU] 🔍 可用事件列表: {events}")
    except Exception as e:
        logger.debug(f"[MEMU] 无法列出事件: {e}")
    
    # 尝试监听所有可能的 transcript 相关事件
    possible_events = [
        "user_transcript", "agent_transcript", "transcript",
        "user_speech", "agent_speech", "speech",
        "user_message", "agent_message", "message"
    ]
    
    for event_name in possible_events:
        try:
            @session.on(event_name)
            def debug_event_handler(*args, **kwargs):
                logger.info(f"[MEMU] 🔔 事件 '{event_name}' 被触发！")
                logger.info(f"[MEMU]   参数数量: {len(args)}, 关键字参数: {list(kwargs.keys())}")
                if args:
                    logger.info(f"[MEMU]   第一个参数类型: {type(args[0]).__name__}")
                    if hasattr(args[0], 'text'):
                        logger.info(f"[MEMU]   文本内容: {args[0].text[:100]}")
        except Exception as e:
            logger.debug(f"[MEMU] 无法注册事件 '{event_name}': {e}")



    # await session.generate_reply(
    #     instructions="对用户打招呼并且表达你的帮助"
    # )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    agents.cli.run_app(server)
