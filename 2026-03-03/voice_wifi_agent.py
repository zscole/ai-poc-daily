#!/usr/bin/env python3

"""
Ultra-Fast Voice Agent with WiFi Presence Detection
Combines sub-500ms voice response with WiFi-based human presence sensing

This POC demonstrates the cutting-edge convergence of:
1. Ultra-low latency voice AI (targeting <500ms response)
2. WiFi signal processing for presence detection (no cameras)
3. Real-time agent coordination based on environmental context

Based on trending research:
- Sub-500ms voice agents (ntik.me/posts/voice-agent)
- WiFi DensePose sensing (github.com/ruvnet/RuView)
- Agent orchestration platforms (OpenSandbox, Claude swarms)
"""

import asyncio
import json
import time
import numpy as np
import websockets
import subprocess
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PresenceData:
    """WiFi-based presence detection data"""
    presence_detected: bool
    signal_strength: float
    motion_level: float
    timestamp: float
    confidence: float

@dataclass
class VoiceResponse:
    """Voice agent response with timing metrics"""
    text: str
    audio_url: Optional[str]
    response_time_ms: float
    first_token_time_ms: float
    presence_context: bool

class WiFiPresenceDetector:
    """
    Simulates WiFi-based presence detection
    In production, this would interface with ESP32 CSI data or WiFi signal analysis
    """
    
    def __init__(self):
        self.baseline_rssi = {}
        self.signal_history = deque(maxlen=20)
        self.is_scanning = False
        
    async def start_scanning(self):
        """Start WiFi presence detection"""
        self.is_scanning = True
        logger.info("WiFi presence detection started")
        
    async def stop_scanning(self):
        """Stop WiFi presence detection"""
        self.is_scanning = False
        logger.info("WiFi presence detection stopped")
        
    async def get_presence_data(self) -> PresenceData:
        """
        Analyze WiFi signals for human presence
        Uses RSSI variance and signal patterns to detect movement/presence
        """
        try:
            # Get WiFi scan data
            result = subprocess.run(['iwlist', 'scan'], capture_output=True, text=True, timeout=2)
            wifi_data = result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to simulated data for demo
            wifi_data = self._simulate_wifi_data()
            
        presence_score = self._analyze_wifi_signals(wifi_data)
        
        presence_data = PresenceData(
            presence_detected=presence_score > 0.3,
            signal_strength=presence_score,
            motion_level=min(presence_score * 2, 1.0),
            timestamp=time.time(),
            confidence=0.85 if presence_score > 0.3 else 0.65
        )
        
        self.signal_history.append(presence_score)
        return presence_data
    
    def _simulate_wifi_data(self) -> str:
        """Simulate WiFi scan data for demonstration"""
        return f"""Cell 01 - Address: 00:1B:2F:BD:04:07
                  ESSID:"TestNetwork"
                  Quality=70/70  Signal level=-{30 + np.random.randint(0, 20)} dBm
        """
    
    def _analyze_wifi_signals(self, wifi_data: str) -> float:
        """
        Analyze WiFi signal patterns for presence detection
        Real implementation would use CSI phase/amplitude analysis
        """
        import re
        
        # Extract signal levels
        signal_pattern = r'Signal level=(-?\d+) dBm'
        signals = [int(match) for match in re.findall(signal_pattern, wifi_data)]
        
        if not signals:
            # Simulate presence based on time variation
            base_presence = 0.4 + 0.3 * np.sin(time.time() * 0.1)
            noise = np.random.normal(0, 0.1)
            return max(0, min(1, base_presence + noise))
        
        # Calculate variance-based presence score
        if len(self.signal_history) >= 5:
            recent_signals = list(self.signal_history)[-5:]
            variance = np.var(recent_signals)
            presence_score = min(variance / 10.0, 1.0)
        else:
            presence_score = 0.5
            
        return presence_score

class UltraFastVoiceAgent:
    """
    Ultra-low latency voice agent targeting <500ms response time
    Implements streaming pipeline: STT -> LLM -> TTS with minimal buffering
    """
    
    def __init__(self):
        self.conversation_history = []
        self.response_times = deque(maxlen=100)
        self.target_latency_ms = 500
        
    async def process_voice_input(self, audio_data: bytes, presence_context: PresenceData) -> VoiceResponse:
        """
        Process voice input with ultra-fast pipeline
        Returns response in <500ms target
        """
        start_time = time.time()
        
        # Simulated STT (in production: use streaming STT like Deepgram)
        transcript = await self._simulate_stt(audio_data)
        stt_time = time.time()
        
        # Context-aware prompt with presence information
        context_prompt = self._build_context_prompt(transcript, presence_context)
        
        # Streaming LLM inference (target: <100ms TTFT)
        llm_start = time.time()
        response_text = await self._fast_llm_inference(context_prompt)
        first_token_time = time.time()
        
        # Stream to TTS immediately (no buffering)
        audio_url = await self._stream_to_tts(response_text)
        
        total_time = (time.time() - start_time) * 1000
        first_token_time_ms = (first_token_time - llm_start) * 1000
        
        response = VoiceResponse(
            text=response_text,
            audio_url=audio_url,
            response_time_ms=total_time,
            first_token_time_ms=first_token_time_ms,
            presence_context=presence_context.presence_detected
        )
        
        self.response_times.append(total_time)
        self.conversation_history.append((transcript, response_text))
        
        logger.info(f"Response generated in {total_time:.1f}ms (TTFT: {first_token_time_ms:.1f}ms)")
        
        return response
    
    async def _simulate_stt(self, audio_data: bytes) -> str:
        """Simulate speech-to-text conversion"""
        await asyncio.sleep(0.05)  # Simulate 50ms STT latency
        
        # Simulate realistic voice queries
        queries = [
            "What's the weather like?",
            "Turn on the lights",
            "Is anyone in the room?",
            "Set a timer for 5 minutes",
            "Play some music",
            "What time is it?",
            "Tell me a joke"
        ]
        return np.random.choice(queries)
    
    def _build_context_prompt(self, transcript: str, presence: PresenceData) -> str:
        """Build context-aware prompt including presence information"""
        presence_info = "someone is present" if presence.presence_detected else "no one detected"
        motion_info = f"motion level: {presence.motion_level:.1f}"
        
        prompt = f"""User said: "{transcript}"
        
Room context: {presence_info}, {motion_info}
Confidence: {presence.confidence:.1f}

Respond naturally and briefly. If the user asks about presence/occupancy, use the room context.
If motion is detected, acknowledge environmental awareness."""
        
        return prompt
    
    async def _fast_llm_inference(self, prompt: str) -> str:
        """
        Simulate ultra-fast LLM inference
        Target: <100ms first token (like Groq's llama-3.3-70b)
        """
        # Simulate TTFT latency (Groq-level performance)
        await asyncio.sleep(0.08)  # 80ms TTFT simulation
        
        # Context-aware responses
        if "weather" in prompt.lower():
            return "It's a lovely 72°F with clear skies today."
        elif "lights" in prompt.lower():
            return "I've turned on the lights in your current room."
        elif "anyone" in prompt.lower() or "present" in prompt.lower():
            if "someone is present" in prompt:
                return "Yes, I detect someone is currently in the room based on WiFi signal patterns."
            else:
                return "I don't detect anyone in the room right now based on the WiFi sensors."
        elif "timer" in prompt.lower():
            return "Timer set for 5 minutes. I'll alert you when it's done."
        elif "music" in prompt.lower():
            return "Starting your favorite playlist now."
        elif "time" in prompt.lower():
            current_time = datetime.now().strftime("%I:%M %p")
            return f"It's currently {current_time}."
        elif "joke" in prompt.lower():
            return "Why don't scientists trust atoms? Because they make up everything!"
        else:
            motion_context = ""
            if "motion level: 0.7" in prompt or "motion level: 0.8" in prompt or "motion level: 0.9" in prompt:
                motion_context = " I notice some movement in the room."
            return f"I understand.{motion_context} How else can I help?"
    
    async def _stream_to_tts(self, text: str) -> Optional[str]:
        """
        Stream text to TTS for immediate audio generation
        In production: use streaming TTS like ElevenLabs WebSocket
        """
        await asyncio.sleep(0.1)  # Simulate TTS processing
        return f"audio://generated/{hash(text)}.wav"
    
    def get_performance_stats(self) -> Dict:
        """Get agent performance statistics"""
        if not self.response_times:
            return {"status": "no_data"}
            
        times = list(self.response_times)
        return {
            "avg_response_time_ms": np.mean(times),
            "p95_response_time_ms": np.percentile(times, 95),
            "p99_response_time_ms": np.percentile(times, 99),
            "target_latency_ms": self.target_latency_ms,
            "target_met_percentage": sum(1 for t in times if t <= self.target_latency_ms) / len(times) * 100,
            "total_interactions": len(times)
        }

class VoiceWiFiOrchestrator:
    """
    Main orchestrator combining voice AI with WiFi presence detection
    Demonstrates the convergence of ultra-fast voice and ambient sensing
    """
    
    def __init__(self):
        self.wifi_detector = WiFiPresenceDetector()
        self.voice_agent = UltraFastVoiceAgent()
        self.is_running = False
        
    async def start(self):
        """Start the combined voice+WiFi system"""
        logger.info("Starting Voice+WiFi Agent System")
        self.is_running = True
        
        await self.wifi_detector.start_scanning()
        
        # Start background presence monitoring
        asyncio.create_task(self._background_presence_monitoring())
        
    async def stop(self):
        """Stop the system"""
        logger.info("Stopping Voice+WiFi Agent System")
        self.is_running = False
        await self.wifi_detector.stop_scanning()
        
    async def _background_presence_monitoring(self):
        """Continuously monitor WiFi presence in background"""
        while self.is_running:
            try:
                presence = await self.wifi_detector.get_presence_data()
                
                # Log significant presence changes
                if hasattr(self, '_last_presence'):
                    if presence.presence_detected != self._last_presence.presence_detected:
                        status = "entered" if presence.presence_detected else "left"
                        logger.info(f"Presence change detected: someone {status} the room")
                
                self._last_presence = presence
                
            except Exception as e:
                logger.error(f"Error in presence monitoring: {e}")
                
            await asyncio.sleep(1.0)  # Check every second
    
    async def handle_voice_interaction(self, audio_data: bytes) -> VoiceResponse:
        """
        Handle a complete voice interaction with environmental context
        """
        # Get current presence data
        presence = await self.wifi_detector.get_presence_data()
        
        # Process voice with presence context
        response = await self.voice_agent.process_voice_input(audio_data, presence)
        
        # Log the interaction
        logger.info(f"Voice interaction: {response.response_time_ms:.1f}ms response time")
        logger.info(f"Presence context: {presence.presence_detected} (confidence: {presence.confidence:.2f})")
        
        return response
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        return {
            "voice_agent": self.voice_agent.get_performance_stats(),
            "presence_detection": {
                "active": self.is_running,
                "last_reading": getattr(self, '_last_presence', None).__dict__ if hasattr(self, '_last_presence') else None
            },
            "system": {
                "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
                "target_latency_ms": 500
            }
        }

# WebSocket Server for Real-time Demo
class DemoWebSocketServer:
    """WebSocket server for real-time demonstration"""
    
    def __init__(self, orchestrator: VoiceWiFiOrchestrator):
        self.orchestrator = orchestrator
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'voice_input':
                    # Simulate audio data
                    audio_data = b"simulated_audio_data"
                    response = await self.orchestrator.handle_voice_interaction(audio_data)
                    
                    await websocket.send(json.dumps({
                        'type': 'voice_response',
                        'data': response.__dict__
                    }))
                    
                elif data['type'] == 'get_status':
                    status = self.orchestrator.get_system_status()
                    await websocket.send(json.dumps({
                        'type': 'system_status',
                        'data': status
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")

async def main():
    """Main demonstration function"""
    print("=" * 60)
    print("Ultra-Fast Voice Agent with WiFi Presence Detection")
    print("Real-time demonstration of <500ms voice AI + ambient sensing")
    print("=" * 60)
    
    # Initialize system
    orchestrator = VoiceWiFiOrchestrator()
    await orchestrator.start()
    
    try:
        # Run demo interactions
        print("\nRunning sample interactions...")
        
        for i in range(5):
            print(f"\n--- Interaction {i+1} ---")
            
            # Simulate voice input
            audio_data = b"simulated_audio_data"
            response = await orchestrator.handle_voice_interaction(audio_data)
            
            print(f"Response: {response.text}")
            print(f"Latency: {response.response_time_ms:.1f}ms")
            print(f"TTFT: {response.first_token_time_ms:.1f}ms")
            print(f"Presence detected: {response.presence_context}")
            
            # Wait between interactions
            await asyncio.sleep(2)
        
        # Show final statistics
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        stats = orchestrator.get_system_status()
        voice_stats = stats['voice_agent']
        
        print(f"Average Response Time: {voice_stats['avg_response_time_ms']:.1f}ms")
        print(f"95th Percentile: {voice_stats['p95_response_time_ms']:.1f}ms")
        print(f"Target Met: {voice_stats['target_met_percentage']:.1f}% of interactions")
        print(f"Total Interactions: {voice_stats['total_interactions']}")
        
        print(f"\nWiFi Presence Detection: {'Active' if stats['presence_detection']['active'] else 'Inactive'}")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("This POC demonstrates the convergence of ultra-fast voice AI")
        print("with WiFi-based ambient sensing for next-generation interfaces.")
        print("=" * 60)
        
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")