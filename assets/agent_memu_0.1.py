import logging
import os
import asyncio
import sys
import time
from memu import MemuClient

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    inference,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
    voice
)

from livekit.plugins import silero

from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import openai

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")


#OPENAI-API
load_dotenv(override=True)
api_key = os.getenv("OPENAI_APIKEY")
base_url = os.getenv("BASE_URL")
memu_key = os.getenv("MEMU_API_KEY")

# Memu.so client
memu_client = MemuClient(
    base_url="https://api.memu.so",
    api_key=memu_key
)

# retrieve user memories
def retrieve_user_memories(user_id, agent_id):
    try:
        memories = memu_client.retrieve_default_categories(
            user_id=user_id,
            agent_id=agent_id
        )
        print('Retrieved memories:', memories)
        return memories
    except Exception as error:
        print('Error retrieving memories:', error)
        return {'categories': []}

# build context from memories
def build_system_prompt(memories, base_prompt):
    system_prompt = base_prompt + "\n\nHere's what you know about the user:\n\n"
    
    if memories and 'categories' in memories:
        for category in memories['categories']:
            if category.get('summary'):
                system_prompt += f"**{category['name']}:** {category['summary']}\n\n"
    
    return system_prompt

# save conversation to memory (synchronous)
def save_conversation_sync(conversation, user_id, agent_id):
    try:
        response = memu_client.memorize_conversation(
            conversation=conversation,
            user_id=user_id,
            user_name="Demo User",
            agent_id=agent_id,
            agent_name="AI Assistant"
        )
        print('Conversation saved! Task ID:', response.task_id)
    except Exception as error:
        print('Error saving conversation:', error)

# async wrapper to run the save in a background thread
async def save_conversation_async(conversation, user_id, agent_id):
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,  # Use the default thread pool executor
            save_conversation_sync,
            conversation,
            user_id,
            agent_id
        )
    except Exception as error:
        print(f'Error saving conversation in background: {error}')

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # User and agent identifiers (hardcoded for demo)
    user_id = "user_123"
    agent_id = "assistant_001"

    # Retrieve user memories
    user_memories = retrieve_user_memories(user_id, agent_id)

    # Build dynamic system prompt with memories
    base_instructions = """你是一个有用的语音人工智能助手。你热心地帮助用户解答他们的问题，从你广博的知识中提供信息。
    你的回答简洁明了，没有任何复杂的格式或标点符号，包括表情符号、星号或其他符号。你好奇、友善，而且有幽默感。"""
    dynamic_instructions = build_system_prompt(user_memories, base_instructions)

    # Create a dynamic Assistant instance with memory context
    class Assistant(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=dynamic_instructions)

    session = AgentSession(
        stt=inference.STT(
            model="deepgram/nova-2", 
            language="zh"
        ),
        llm=openai.LLM(
            model="gpt-5", 
            base_url=base_url, 
            api_key=api_key
        ),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="用友好和对话的语气说话",
            base_url=base_url, 
            api_key=api_key
        ),
        turn_detection="vad",
        vad=silero.VAD.load(),
    )

    # Define the event listener for when a turn is finished
    @session.on("turn_finished")
    async def on_turn_finished(turn):
        # We only want to save turns where the user and the agent both spoke
        if turn.user_speech is None or turn.agent_speech is None:
            return

        # Build the conversation context for saving
        conversation_context = [
            {"role": "user", "content": turn.user_speech.text},
            {"role": "assistant", "content": turn.agent_speech.text}
        ]

        # Save the conversation in the background
        asyncio.create_task(
            save_conversation_async(conversation_context, user_id, agent_id)
        )

    #启动对话
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="对用户打招呼并且表达你的帮助"
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    agents.cli.run_app(server)
