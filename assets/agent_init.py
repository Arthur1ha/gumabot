import logging
import os
import asyncio
import sys

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
)

from livekit.plugins import silero

from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import openai

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")


#OPENAI-API
load_dotenv(override=True)
api_key = os.getenv("OPENAI_APIKEY")  # 获取键为 API_KEY 的值
base_url = os.getenv("BASE_URL")
# print(api_key)
# print(base_url)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""你是一个有用的语音人工智能助手。你热心地帮助用户解答他们的问题，从你广博的知识中提供信息。
            你的回答简洁明了，没有任何复杂的格式或标点符号，包括表情符号、星号或其他符号。你好奇、友善，而且有幽默感。""",
        )

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # stt = openai.STT(
        #     model="gpt-4o-transcribe",
        #     api_key=api_key,
        #     base_url=base_url
        # ),
        stt=inference.STT(
            model="deepgram/nova-2", 
            language="zh"
        ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=openai.LLM(
            model="gpt-5", 
            base_url=base_url, 
            api_key=api_key
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts = openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="用友好和对话的语气说话",
            base_url=base_url, 
            api_key=api_key
        ),

        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        # turn_detection=MultilingualModel(),
        turn_detection="vad",
        vad=silero.VAD.load(),

        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        # preemptive_generation=True,

        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # # when it's detected, you may resume the agent's speech
        # resume_false_interruption=True,
        # false_interruption_timeout=1.0,
    )

    #启动对话
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
                # noise_cancellation=lambda params:krisp.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else krisp.BVC()
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