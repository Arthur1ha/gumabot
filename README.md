# Guma 语音智能助手（LiveKit + MemU）

## 项目概述
- 这是一个基于 LiveKit Agents 的语音助手，集成了语音转写（STT）、大语言模型（LLM）、语音合成（TTS），并通过 MemU 提供会话级“记忆层”。
- 程序主入口与全部逻辑在 `d:\study\git\gumabot\agent.py`，启动后即可在房间内进行语音对话，并持续将对话保存到 MemU，检索分类摘要并动态整合到系统提示词中。

## 功能特性
- 语音对话：VAD 端点检测后进行 STT → LLM → TTS 的完整闭环对话。
- 记忆层：
  - 提交对话到 MemU，异步生成用户“分类与摘要”。
  - 轮询记忆任务状态为完成后，检索默认分类并将摘要整合进系统提示词，后续回答可利用记忆上下文。
- 动态提示词：构建时输出基础提示词与整合后的动态提示词；记忆刷新后再次打印更新后的提示词，便于观测。
- 日志与事件：注册多类 LiveKit 事件，打印状态、转写、消息入栈与保存动作等信息，便于调试。

## 技术架构
- LiveKit Agents：
  - `AgentServer` + `AgentSession` 作为会话容器。
  - STT：示例采用 Deepgram `STTv2`（`flux-general-en`），也可切换 OpenAI/Groq 等。
  - LLM：示例使用 `openai.LLM.with_x_ai`（`grok-4.1`），亦可切换至 OpenAI 标准模型。
  - TTS：示例使用 `openai.TTS`（`gpt-4o-mini-tts`）。
  - VAD：`silero.VAD.load()`；`turn_detection="vad"`。
- MemU：
  - `memorize_conversation` 注册记忆任务（异步）。
  - 通过 `GET /memory/memorize/status/{task_id}` 轮询，完成后再检索默认分类。
  - 将有 `summary` 的分类整合进系统提示词。

## 目录与关键代码
- 主文件：`agent.py`
  - 记忆函数：
    - `retrieve_user_memories(user_id, agent_id)`：检索默认分类并打印摘要预览。
    - `build_system_prompt_with_memories(base_instructions, memories)`：把分类摘要拼进系统提示词。
    - `save_conversation_to_memu(conversation, user_id, agent_id, assistant, base_instructions)`：提交对话，记录任务 ID，后续刷新提示词。
    - `refresh_memories_and_update_prompt_with_task(task_id, ...)`：轮询任务状态为完成后，检索分类、更新提示词。
    - `extract_categories(memories)` 与 `extract_value(item, key)`：兼容 Pydantic 模型/字典的通用访问工具。
  - 会话：`entrypoint(ctx)` 构建提示词、创建 `Assistant`、初始化 `AgentSession`、注册事件并启动。

## 环境准备
- 依赖环境变量（可放入 `.env`）：
  - `OPENAI_APIKEY`：OpenAI 或 xAI 的 API Key。
  - `BASE_URL`：对应提供商的基础端点（如 xAI 的 `https://api.x.ai/v1`）。
  - `DEEPGRAM_API_KEY`：Deepgram 的 API Key（若使用 Deepgram STT）。
  - `MEMU_API_KEY`：MemU 的 API Key（启用记忆层必要）。
- 运行环境：Windows（示例里设置了 `WindowsSelectorEventLoopPolicy`）。

## 启动方式
- 直接运行：
  - 在 Windows 环境中执行 Python 程序，入口在 `if __name__ == "__main__": agents.cli.run_app(server)`。
- 房间与身份：
  - 默认使用 `ctx.room.name` 作为 `user_id`，`agent_id` 固定为 `voice_assistant_001`；根据业务可自行修改。

## 配置与定制
- STT 选择：
  - 采用 Deepgram `STTv2(model="flux-general-en")`；如需中文可改成 `nova-2` 或其他模型。
  - 或改为 OpenAI STT：`openai.STT(model="gpt-4o-transcribe", base_url="https://api.openai.com/v1")`。
  - 或改为 Groq STT：`groq.STT(model="whisper-large-v3-turbo", base_url="https://api.groq.com/openai/v1")`。
- LLM 选择：
  - 示例使用 xAI Grok：`openai.LLM.with_x_ai(model="grok-4.1", base_url=...)`。
  - 或使用 OpenAI：`openai.LLM(model="gpt-4o-mini", base_url="https://api.openai.com/v1")`。
- TTS 选择：
  - 示例使用 `openai.TTS(model="gpt-4o-mini-tts")`。
  - 可按供应商文档替换与配置语音参数。
- 系统提示词：
  - 统一在 `DEFAULT_BASE_INSTRUCTIONS` 中维护基础提示词，构建时会整合记忆摘要形成动态提示词。

## 记忆层工作流（MemU）
- 保存：当缓冲达到阈值（默认 4 条消息）或会话关闭时，提交 `full_conversation` 到 MemU。
- 轮询：提交后通过 `GET /memory/memorize/status/{task_id}` 轮询任务状态为 `completed`。
- 检索：完成后执行 `retrieve_default_categories(user_id, agent_id)` 获取分类与摘要。
- 注入：有 `summary` 的分类会被拼入系统提示词，并立即更新 `assistant.instructions`。
- 日志：会打印任务 ID、摘要条目数、更新后的提示词全文，便于确认效果。

## 常见问题排查
- 429（Agent Gateway STT 拒绝）：
  - 原因为 LiveKit 网关额度/权限不足；改用供应商直连 STT 或检查 LiveKit 配额。
- 500 multipart EOF（OpenAI STT 表单解析失败）：
  - 多为 `base_url` 指向非 OpenAI 音频端点或上传空音频；统一用 `https://api.openai.com/v1` 并确保音轨发布正常。
- `DefaultCategoriesResponse` 无 `get` 属性：
  - 返回是 Pydantic 模型，已通过 `extract_categories`/`extract_value` 统一访问解决。
- 检索为 0 类别：
  - 记忆任务异步处理未完成或对话信息缺少可提取摘要；已加入状态轮询与重试；建议在对话中明确表达个人偏好、事件等。

## 日志观察点
- 基础提示词与动态提示词：启动时打印，便于对比记忆整合效果。
- 更新后的提示词：保存与刷新成功后打印，用于确认已应用记忆摘要。
- 对话保存：打印任务 ID 与消息数。

## 提示
- 请勿将 API Key 明文写入代码或日志；使用 `.env` 管理密钥。
- 根据业务需要调整保存阈值、轮询次数与间隔，以权衡实时性与开销。

