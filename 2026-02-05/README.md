# Voxtral Realtime: Streaming Speech-to-Text POC

**Date:** February 5, 2026
**Topic:** Real-time streaming transcription with sub-200ms latency
**Source:** Mistral AI's Voxtral Transcribe 2 release

## Why This Matters

Mistral released Voxtral Transcribe 2 yesterday, and it's getting massive attention (887+ points on Hacker News). The breakthrough isn't just another STT model - it's the **streaming architecture** that enables true real-time applications.

Traditional speech-to-text works like this:
1. Record complete audio
2. Send to API
3. Wait for transcript

First token latency: **5-10+ seconds**

Voxtral Realtime works like this:
1. Stream audio chunks as they arrive
2. Get transcript tokens as they're generated

First token latency: **150-200ms**

This 30-50x improvement enables entirely new application categories: voice agents that respond in sub-second time, live captioning, real-time translation, and interactive voice interfaces that feel natural.

## What's Included

### 1. voxtral_realtime.py - Core Transcription Demo

Full implementation of streaming transcription using Voxtral Realtime API.

```bash
# Install dependencies
pip install mistralai pyaudio rich

# Set API key
export MISTRAL_API_KEY=your_key

# Run with microphone
python voxtral_realtime.py

# Or transcribe a file
python voxtral_realtime.py --file audio.wav

# Demo mode (no API key needed)
python voxtral_realtime.py --demo
```

Features:
- Real-time microphone capture
- WebSocket streaming to Voxtral API
- Live transcript display with Rich
- Performance metrics (latency, throughput)
- Configurable chunk duration for latency tuning

### 2. voice_agent.py - Voice Agent Framework

Demonstrates chaining Voxtral Realtime with an LLM for interactive voice conversations.

```bash
# With Mistral for both STT and LLM
python voice_agent.py

# Or use OpenAI for LLM responses
export OPENAI_API_KEY=your_key
python voice_agent.py --openai-key $OPENAI_API_KEY

# Architecture demo
python voice_agent.py --demo
```

Architecture:
```
[Microphone] -> [VAD] -> [Voxtral Realtime] -> [LLM] -> [Response]
     |           |              |               |           |
  Audio      Detect         Streaming       Process      Text
  Input      speech           STT          transcript   output
```

### 3. streaming_vs_chunked.py - Architecture Comparison

Interactive demo illustrating the latency difference between chunked and streaming approaches.

```bash
python streaming_vs_chunked.py --demo
python streaming_vs_chunked.py --compare
```

## Technical Details

### Streaming Architecture

Voxtral Realtime uses a WebSocket connection for bidirectional streaming:

```
Client                              Server
  |                                    |
  |--[Audio chunk 10ms]-------------->|
  |--[Audio chunk 10ms]-------------->|
  |<--------------[Transcript delta]--|
  |--[Audio chunk 10ms]-------------->|
  |<--------------[Transcript delta]--|
  ...
```

Key parameters:
- **Sample rate:** 8kHz, 16kHz, 22.05kHz, 44.1kHz, or 48kHz
- **Chunk duration:** 10ms recommended for low latency
- **Encoding:** PCM signed 16-bit little-endian

### Latency Modes

Voxtral Realtime offers configurable latency vs quality tradeoffs:

| Delay | Use Case | Quality Impact |
|-------|----------|----------------|
| 200ms | Voice agents, live captioning | Slightly reduced |
| 480ms | Real-time subtitling | Within 1-2% of offline |
| 2400ms | Maximum quality | Matches offline models |

### Language Support

13 languages: English, Chinese, Hindi, Spanish, Arabic, French, Portuguese, Russian, German, Japanese, Korean, Italian, Dutch

### Open Weights

Voxtral Realtime is available as open weights under Apache 2.0:
https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602

The 4B parameter model runs on edge devices for privacy-sensitive deployments.

## Requirements

```
mistralai>=1.0.0
pyaudio>=0.2.14
rich>=13.0.0
```

For microphone input on macOS:
```bash
brew install portaudio
pip install pyaudio
```

## API Pricing

- **Voxtral Mini Transcribe V2 (batch):** $0.003/minute
- **Voxtral Realtime (streaming):** $0.006/minute

Both are significantly cheaper than competitors while matching or exceeding quality.

## The Bigger Picture

This release signals where voice AI is heading:

1. **Sub-second voice agents become practical.** The 150-200ms first token latency + fast LLM response means <500ms total response time is achievable.

2. **Edge deployment for privacy.** Open weights + 4B parameters means running on-device without cloud dependencies.

3. **Multimodal real-time applications.** Streaming STT + streaming LLM + streaming TTS creates fluid voice interfaces.

4. **Cost disruption.** At $0.003-0.006/min with SOTA quality, the economics enable voice AI at scale.

## Related Trends This Week

- **Claude Code local fallback:** Users connecting to local models (Qwen3-Coder-Next, GLM-4.7-Flash) when API quota runs out. Hybrid architectures becoming standard.

- **Xcode AI integration:** Apple adding OpenAI and Anthropic agentic coding to Xcode. IDEs becoming AI-native.

- **Fluid.sh:** AI infrastructure agents using sandbox clones for safe exploration before generating IaC. Safety-first AI ops.

- **MCP proliferation:** ActivePieces trending with ~400 MCP servers. Tool use becoming the standard agent interface.

## Links

- Voxtral announcement: https://mistral.ai/news/voxtral-transcribe-2
- API docs: https://docs.mistral.ai/capabilities/audio_transcription
- Open weights: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- Audio playground: https://console.mistral.ai/build/audio/speech-to-text
