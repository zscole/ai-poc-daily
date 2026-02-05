#!/usr/bin/env python3
"""
Voice Agent POC - Streaming STT + LLM Integration

Demonstrates building a voice agent using Voxtral Realtime for streaming
speech-to-text combined with an LLM for response generation. This is the
pattern that enables sub-second voice assistants.

Architecture:
    [Microphone] -> [Voxtral Realtime] -> [LLM] -> [Response]
         |               |                  |           |
      Audio          Streaming           Process     Text/TTS
      Input           STT               Transcript   Output

Usage:
    pip install mistralai openai pyaudio rich
    export MISTRAL_API_KEY=your_key
    export OPENAI_API_KEY=your_key  # Optional, for GPT responses
    python voice_agent.py

Features:
    - Real-time streaming transcription (sub-200ms latency)
    - Endpoint detection (knows when you stop speaking)
    - LLM response generation
    - Conversation history management
"""

import argparse
import asyncio
import os
import sys
import time
from typing import AsyncIterator, List, Optional
from dataclasses import dataclass, field

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from mistralai import Mistral
    from mistralai.models import (
        AudioFormat,
        RealtimeTranscriptionSessionCreated,
        TranscriptionStreamDone,
        TranscriptionStreamTextDelta,
    )
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    """Manages conversation state for the voice agent."""
    history: List[Message] = field(default_factory=list)
    current_transcript: str = ""
    is_listening: bool = False
    is_processing: bool = False
    last_response: str = ""
    
    def add_user_message(self, content: str) -> None:
        self.history.append(Message(role="user", content=content))
        self.current_transcript = ""
    
    def add_assistant_message(self, content: str) -> None:
        self.history.append(Message(role="assistant", content=content))
        self.last_response = content
    
    def get_context(self, max_messages: int = 10) -> List[dict]:
        """Get conversation context for LLM."""
        messages = self.history[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in messages]


class VoiceAgentDisplay:
    """Rich display for voice agent state."""
    
    def __init__(self) -> None:
        self.state = ConversationState()
        self.console = Console() if RICH_AVAILABLE else None
    
    def render(self) -> Panel:
        if not RICH_AVAILABLE:
            return None
        
        content = Text()
        
        # Status
        if self.state.is_listening:
            content.append("[green]Listening...[/]\n\n")
        elif self.state.is_processing:
            content.append("[yellow]Processing...[/]\n\n")
        else:
            content.append("[dim]Ready (speak to start)[/]\n\n")
        
        # Conversation history (last 3 exchanges)
        recent = self.state.history[-6:]
        for msg in recent:
            if msg.role == "user":
                content.append("You: ", style="cyan bold")
                content.append(f"{msg.content}\n")
            else:
                content.append("Agent: ", style="green bold")
                content.append(f"{msg.content}\n")
        
        # Current transcript
        if self.state.current_transcript:
            content.append("\n")
            content.append("[dim]Hearing: [/]")
            content.append(self.state.current_transcript, style="italic")
        
        return Panel(
            content,
            title="Voice Agent",
            subtitle="Ctrl+C to exit",
            border_style="blue"
        )


async def iter_microphone_with_vad(
    sample_rate: int = 16000,
    chunk_duration_ms: int = 10,
    silence_threshold: float = 2.0,
) -> AsyncIterator[tuple[bytes, bool]]:
    """
    Yield microphone chunks with voice activity detection.
    Returns (audio_bytes, is_speech) tuples.
    
    Simple energy-based VAD - production would use webrtcvad or similar.
    """
    try:
        import pyaudio
        import struct
    except ImportError:
        print("Error: PyAudio required")
        return
    
    p = pyaudio.PyAudio()
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_samples,
    )
    
    silence_chunks = 0
    speech_started = False
    energy_threshold = 500  # Adjust based on environment
    
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, stream.read, chunk_samples, False
            )
            
            # Calculate RMS energy
            samples = struct.unpack(f"{chunk_samples}h", data)
            rms = (sum(s * s for s in samples) / chunk_samples) ** 0.5
            
            is_speech = rms > energy_threshold
            
            if is_speech:
                speech_started = True
                silence_chunks = 0
            elif speech_started:
                silence_chunks += 1
            
            # Detect end of utterance
            silence_duration = silence_chunks * chunk_duration_ms / 1000
            end_of_speech = speech_started and silence_duration > silence_threshold
            
            yield data, is_speech
            
            if end_of_speech:
                speech_started = False
                silence_chunks = 0
                break
                
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def get_llm_response(
    messages: List[dict],
    api_key: str,
    model: str = "mistral-small-latest",
    use_openai: bool = False,
) -> str:
    """
    Get response from LLM given conversation history.
    Supports both Mistral and OpenAI.
    """
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful voice assistant. Keep responses concise and "
            "conversational since they will be spoken aloud. Aim for 1-2 "
            "sentences unless more detail is specifically requested."
        )
    }
    
    all_messages = [system_prompt] + messages
    
    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=all_messages,
                max_tokens=150,
            )
            return response.choices[0].message.content
        except ImportError:
            print("OpenAI not available, falling back to Mistral")
    
    # Use Mistral
    client = Mistral(api_key=api_key)
    response = await client.chat.complete_async(
        model=model,
        messages=all_messages,
        max_tokens=150,
    )
    return response.choices[0].message.content


async def run_voice_agent(
    mistral_key: str,
    openai_key: Optional[str] = None,
    sample_rate: int = 16000,
) -> None:
    """
    Run the voice agent loop.
    
    Flow:
    1. Listen for speech
    2. Transcribe in real-time with Voxtral
    3. Detect end of utterance
    4. Send transcript to LLM
    5. Display/speak response
    6. Repeat
    """
    if not MISTRAL_AVAILABLE:
        raise ImportError("mistralai package required")
    
    client = Mistral(api_key=mistral_key)
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=sample_rate)
    display = VoiceAgentDisplay()
    
    print("Voice Agent Started")
    print("Speak naturally. The agent will respond when you pause.")
    print("Press Ctrl+C to exit.")
    print("-" * 50)
    
    while True:
        try:
            # Reset for new utterance
            display.state.current_transcript = ""
            display.state.is_listening = True
            
            # Collect audio with VAD
            audio_chunks = []
            
            async for chunk, is_speech in iter_microphone_with_vad(
                sample_rate=sample_rate,
                silence_threshold=1.5,
            ):
                audio_chunks.append(chunk)
            
            if not audio_chunks:
                continue
            
            display.state.is_listening = False
            display.state.is_processing = True
            
            # Create audio iterator from collected chunks
            async def audio_iter():
                for chunk in audio_chunks:
                    yield chunk
            
            # Transcribe
            transcript_parts = []
            async for event in client.audio.realtime.transcribe_stream(
                audio_stream=audio_iter(),
                model="voxtral-mini-transcribe-realtime-2602",
                audio_format=audio_format,
            ):
                if isinstance(event, TranscriptionStreamTextDelta):
                    transcript_parts.append(event.text)
                    display.state.current_transcript = "".join(transcript_parts)
                elif isinstance(event, TranscriptionStreamDone):
                    break
            
            transcript = "".join(transcript_parts).strip()
            
            if not transcript:
                display.state.is_processing = False
                continue
            
            # Add to history and get response
            display.state.add_user_message(transcript)
            print(f"\nYou: {transcript}")
            
            # Get LLM response
            response = await get_llm_response(
                messages=display.state.get_context(),
                api_key=openai_key or mistral_key,
                use_openai=bool(openai_key),
            )
            
            display.state.add_assistant_message(response)
            display.state.is_processing = False
            
            print(f"Agent: {response}\n")
            
            # Small delay before listening again
            await asyncio.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\nExiting voice agent...")
            break
        except Exception as e:
            print(f"Error: {e}")
            display.state.is_processing = False
            await asyncio.sleep(1)


def demo_architecture():
    """Show the voice agent architecture without running."""
    print("=" * 60)
    print("Voice Agent Architecture with Voxtral Realtime")
    print("=" * 60)
    print()
    print("Pipeline:")
    print()
    print("  1. Audio Capture")
    print("     [Microphone] -> 16kHz mono PCM -> 10ms chunks")
    print()
    print("  2. Voice Activity Detection (VAD)")
    print("     Detects speech start/end to segment utterances")
    print("     Options: Energy-based, WebRTC VAD, Silero VAD")
    print()
    print("  3. Streaming Transcription (Voxtral Realtime)")
    print("     Sub-200ms latency, transcribes as audio arrives")
    print("     No waiting for complete utterance")
    print()
    print("  4. LLM Response Generation")
    print("     Process transcript, generate conversational response")
    print("     Can start generating before transcription completes")
    print()
    print("  5. Output (Text-to-Speech optional)")
    print("     Display response or synthesize with TTS")
    print()
    print("Latency Breakdown (typical):")
    print("  - Audio capture:      ~10ms")
    print("  - Network:            ~50ms")
    print("  - Voxtral processing: ~150ms")
    print("  - LLM response:       ~300ms")
    print("  - Total:              ~500ms end-to-end")
    print()
    print("This is significantly faster than traditional pipelines that")
    print("wait for complete utterances before starting transcription.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Voice Agent with Voxtral Realtime STT"
    )
    parser.add_argument(
        "--mistral-key",
        default=os.environ.get("MISTRAL_API_KEY"),
        help="Mistral API key"
    )
    parser.add_argument(
        "--openai-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (optional, for GPT responses)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show architecture demo without running"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_architecture()
        return 0
    
    if not args.mistral_key:
        print("No Mistral API key. Running architecture demo...")
        print()
        demo_architecture()
        return 0
    
    if not MISTRAL_AVAILABLE:
        print("Error: mistralai package required")
        print("Install: pip install mistralai")
        return 1
    
    try:
        asyncio.run(run_voice_agent(
            mistral_key=args.mistral_key,
            openai_key=args.openai_key,
            sample_rate=args.sample_rate,
        ))
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
