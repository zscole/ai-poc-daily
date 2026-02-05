#!/usr/bin/env python3
"""
Streaming vs Chunked Transcription Comparison

This demo illustrates the architectural difference between:
1. Traditional chunked transcription (wait for audio, then transcribe)
2. Streaming transcription (transcribe as audio arrives)

The key insight from Voxtral Realtime is that true streaming enables
sub-200ms first-token latency vs 1-2+ seconds for chunked approaches.

Run: python streaming_vs_chunked.py --demo
"""

import time
import sys


def simulate_chunked_transcription(audio_duration_sec: float = 5.0):
    """
    Simulate traditional chunked transcription.
    
    Pattern: Record complete utterance -> Send to API -> Get transcript
    
    Latency = recording_time + network + processing
    """
    print("\n" + "=" * 60)
    print("CHUNKED TRANSCRIPTION (Traditional)")
    print("=" * 60)
    print()
    
    # Simulate recording
    print("Phase 1: Recording complete utterance...")
    print("  [", end="")
    for i in range(int(audio_duration_sec * 2)):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("]")
    print(f"  Recorded {audio_duration_sec}s of audio")
    print()
    
    # Simulate API call
    print("Phase 2: Sending to transcription API...")
    time.sleep(0.3)  # Network latency
    print("  Uploading audio...")
    time.sleep(0.5)  # Upload time
    print()
    
    # Simulate processing
    print("Phase 3: Server processing...")
    time.sleep(1.0)  # Typical Whisper processing
    print("  Processing complete")
    print()
    
    # Result
    total_latency = audio_duration_sec + 0.3 + 0.5 + 1.0
    print(f"Phase 4: Transcript ready!")
    print(f"  'Hello, this is a test of the transcription system.'")
    print()
    print(f"TOTAL TIME TO FIRST TOKEN: {total_latency:.1f} seconds")
    print(f"  - Recording wait: {audio_duration_sec:.1f}s")
    print(f"  - Network: 0.3s")
    print(f"  - Upload: 0.5s")
    print(f"  - Processing: 1.0s")
    
    return total_latency


def simulate_streaming_transcription(audio_duration_sec: float = 5.0):
    """
    Simulate streaming transcription (Voxtral Realtime approach).
    
    Pattern: Stream audio chunks -> Get tokens as they're ready
    
    First token latency = chunk_duration + network + ~150ms processing
    """
    print("\n" + "=" * 60)
    print("STREAMING TRANSCRIPTION (Voxtral Realtime)")
    print("=" * 60)
    print()
    
    chunk_ms = 10
    first_token_latency = 0.150  # 150ms typical
    
    print("Streaming audio and transcribing simultaneously...")
    print()
    
    transcript_words = [
        "Hello,", " this", " is", " a", " test", " of", " the",
        " transcription", " system."
    ]
    
    # First chunk and first token
    print(f"[{chunk_ms}ms] First audio chunk sent")
    time.sleep(first_token_latency)
    first_token_time = chunk_ms / 1000 + first_token_latency
    print(f"[{first_token_time*1000:.0f}ms] FIRST TOKEN: {transcript_words[0]}")
    
    # Simulate streaming rest of transcript
    current_transcript = transcript_words[0]
    for i, word in enumerate(transcript_words[1:], 1):
        time.sleep(0.3)  # Simulated word spacing
        current_transcript += word
        elapsed = first_token_time + (i * 0.3)
        print(f"[{elapsed*1000:.0f}ms] Stream: {word.strip()}")
    
    print()
    print(f"Final transcript: '{current_transcript}'")
    print()
    print(f"FIRST TOKEN LATENCY: {first_token_time*1000:.0f}ms")
    print(f"  - Chunk duration: {chunk_ms}ms")
    print(f"  - Network: ~50ms")
    print(f"  - Processing: ~100ms")
    print()
    print("KEY INSIGHT: Transcript tokens arrive WHILE you're still speaking!")
    
    return first_token_time


def show_comparison():
    """Show side-by-side comparison of approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: CHUNKED vs STREAMING")
    print("=" * 60)
    print()
    print("                    CHUNKED          STREAMING")
    print("                    --------         ---------")
    print("First token         5-10+ sec        150-200ms")
    print("latency")
    print()
    print("Processes audio     After complete   As it arrives")
    print("                    utterance")
    print()
    print("Use case            Batch processing Voice agents")
    print("                    Podcasts         Live captions")
    print("                    Meeting notes    Real-time apps")
    print()
    print("Architecture        Request/Response WebSocket stream")
    print()
    print("When to use         - Accuracy is    - Latency critical")
    print("                      paramount      - Interactive apps")
    print("                    - Batch jobs     - Live feedback")
    print()
    print("-" * 60)
    print()
    print("Voxtral Realtime achieves BOTH:")
    print("  - Streaming latency (sub-200ms)")
    print("  - Near-offline quality (within 1-2% WER at 480ms)")
    print()
    print("This is the architectural breakthrough that enables")
    print("truly responsive voice agents and real-time applications.")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare streaming vs chunked transcription"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison table"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        show_comparison()
        return 0
    
    if args.demo or not (args.compare):
        print("STREAMING vs CHUNKED TRANSCRIPTION DEMO")
        print("This illustrates why streaming architecture matters")
        print()
        
        # Run chunked simulation
        chunked_time = simulate_chunked_transcription(3.0)
        
        print("\n" + "~" * 60 + "\n")
        input("Press Enter to see streaming approach...")
        
        # Run streaming simulation
        streaming_time = simulate_streaming_transcription(3.0)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Chunked first token:   {chunked_time*1000:.0f}ms")
        print(f"Streaming first token: {streaming_time*1000:.0f}ms")
        print(f"Improvement:           {chunked_time/streaming_time:.0f}x faster")
        print()
        
        show_comparison()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
