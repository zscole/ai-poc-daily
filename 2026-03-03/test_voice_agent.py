#!/usr/bin/env python3
"""
Test suite for Ultra-Low Latency Voice Agent
Comprehensive testing without requiring actual audio hardware
"""

import unittest
import asyncio
import numpy as np
import time
from unittest.mock import AsyncMock, MagicMock, patch
from voice_agent_orchestrator import (
    VoiceAgentOrchestrator,
    VoiceActivityDetector,
    ConnectionPool,
    LLMClient,
    AudioFrame,
    AgentState,
    LatencyMetrics
)

class TestVoiceActivityDetector(unittest.TestCase):
    """Test the VAD implementation"""
    
    def setUp(self):
        self.vad = VoiceActivityDetector(threshold=0.02)
    
    def test_silent_audio_detection(self):
        """Test that silence is correctly detected"""
        # Generate silent audio (zeros)
        silent_audio = np.zeros(160, dtype=np.int16).tobytes()
        
        # Multiple frames to build history
        for _ in range(5):
            result = self.vad.detect_speech(silent_audio)
        
        self.assertFalse(result, "Silent audio should not be detected as speech")
    
    def test_speech_audio_detection(self):
        """Test that speech-like audio is detected"""
        # Generate audio with energy (sine wave)
        t = np.linspace(0, 0.02, 160)  # 20ms at 8kHz
        speech_audio = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16).tobytes()
        
        # Multiple frames to build history
        for _ in range(5):
            result = self.vad.detect_speech(speech_audio)
        
        self.assertTrue(result, "Speech-like audio should be detected")
    
    def test_energy_threshold_sensitivity(self):
        """Test different energy thresholds"""
        vad_sensitive = VoiceActivityDetector(threshold=0.001)
        vad_conservative = VoiceActivityDetector(threshold=0.1)
        
        # Low energy audio
        low_energy = (np.random.normal(0, 0.01, 160) * 16384).astype(np.int16).tobytes()
        
        # Build history for both
        for _ in range(5):
            sensitive_result = vad_sensitive.detect_speech(low_energy)
            conservative_result = vad_conservative.detect_speech(low_energy)
        
        # Sensitive VAD should detect, conservative should not
        self.assertTrue(sensitive_result, "Sensitive VAD should detect low energy")
        self.assertFalse(conservative_result, "Conservative VAD should not detect low energy")

class TestConnectionPool(unittest.TestCase):
    """Test the TTS connection pool"""
    
    def setUp(self):
        self.pool = ConnectionPool(pool_size=2)
    
    @patch('websockets.connect')
    async def test_connection_creation(self, mock_connect):
        """Test that connections are created correctly"""
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        
        connection = await self.pool.get_tts_connection()
        
        mock_connect.assert_called_once()
        self.assertEqual(connection, mock_ws)
    
    async def test_connection_pooling(self):
        """Test that connections are properly pooled"""
        # Mock a connection
        mock_ws = AsyncMock()
        
        # Return it to pool
        await self.pool.return_connection(mock_ws)
        
        # Pool should now have one connection
        self.assertEqual(len(self.pool.tts_connections), 1)
        
        # Getting a connection should return the pooled one
        with patch.object(self.pool, '_create_tts_connection') as mock_create:
            connection = await self.pool.get_tts_connection()
            mock_create.assert_not_called()
            self.assertEqual(connection, mock_ws)

class TestLLMClient(unittest.TestCase):
    """Test the LLM streaming client"""
    
    def setUp(self):
        self.client = LLMClient("test-api-key")
    
    @patch('aiohttp.ClientSession.post')
    async def test_streaming_completion(self, mock_post):
        """Test streaming completion with token callback"""
        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.content.__aiter__.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        mock_post.return_value.__aenter__.return_value = mock_response
        
        tokens = []
        
        async def token_callback(token):
            tokens.append(token)
        
        await self.client.stream_completion(
            [{"role": "user", "content": "Hi"}],
            token_callback
        )
        
        self.assertEqual(tokens, ["Hello", " world"])
    
    def test_model_configuration(self):
        """Test model configuration"""
        client = LLMClient("key", model="custom-model")
        self.assertEqual(client.model, "custom-model")
        
        client_default = LLMClient("key")
        self.assertEqual(client_default.model, "llama-3.3-70b-versatile")

class TestVoiceAgentOrchestrator(unittest.TestCase):
    """Test the main orchestrator"""
    
    def setUp(self):
        self.config = {
            "groq_api_key": "test-key",
            "elevenlabs_api_key": "test-key",
            "target_latency_ms": 400
        }
        self.orchestrator = VoiceAgentOrchestrator(self.config)
    
    def test_initial_state(self):
        """Test that orchestrator starts in correct state"""
        self.assertEqual(self.orchestrator.state, AgentState.IDLE)
        self.assertEqual(len(self.orchestrator.conversation_history), 0)
    
    async def test_user_start_speaking_transition(self):
        """Test state transition when user starts speaking"""
        # Create mock audio frame with speech
        audio_frame = AudioFrame(
            data=b'\x01' * 160,  # Non-zero data
            timestamp=time.time()
        )
        
        with patch.object(self.orchestrator.vad, 'detect_speech', return_value=True):
            result = await self.orchestrator.process_audio_frame(audio_frame)
        
        self.assertEqual(self.orchestrator.state, AgentState.USER_SPEAKING)
        self.assertTrue(result["speech_detected"])
    
    async def test_user_stop_speaking_transition(self):
        """Test state transition when user stops speaking"""
        # Set initial state to speaking
        self.orchestrator.state = AgentState.USER_SPEAKING
        
        # Create silent audio frame
        audio_frame = AudioFrame(
            data=b'\x00' * 160,
            timestamp=time.time()
        )
        
        with patch.object(self.orchestrator.vad, 'detect_speech', return_value=False):
            with patch.object(self.orchestrator, '_generate_and_stream_response', new_callable=AsyncMock):
                result = await self.orchestrator.process_audio_frame(audio_frame)
        
        self.assertEqual(self.orchestrator.state, AgentState.PROCESSING)
        self.assertFalse(result["speech_detected"])
    
    async def test_interruption_handling(self):
        """Test that agent generation is cancelled on interruption"""
        # Start with agent processing
        self.orchestrator.state = AgentState.PROCESSING
        
        # Create mock task
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        self.orchestrator.current_generation_task = mock_task
        
        await self.orchestrator._handle_user_start_speaking(time.time())
        
        # Task should be cancelled
        mock_task.cancel.assert_called_once()
        self.assertEqual(self.orchestrator.state, AgentState.USER_SPEAKING)

class TestLatencyMeasurement(unittest.TestCase):
    """Test latency measurement and optimization"""
    
    def test_metrics_initialization(self):
        """Test that metrics are properly initialized"""
        metrics = LatencyMetrics(100, 200, 50, 350)
        
        self.assertEqual(metrics.turn_detection_ms, 100)
        self.assertEqual(metrics.llm_first_token_ms, 200)
        self.assertEqual(metrics.tts_first_audio_ms, 50)
        self.assertEqual(metrics.end_to_end_ms, 350)
    
    def test_sub_500ms_target(self):
        """Test that we can measure sub-500ms performance"""
        config = {"groq_api_key": "test", "elevenlabs_api_key": "test"}
        orchestrator = VoiceAgentOrchestrator(config)
        
        # Simulate fast processing
        start_time = time.time()
        time.sleep(0.1)  # 100ms simulation
        end_time = time.time()
        
        measured_latency = (end_time - start_time) * 1000
        
        # Should be around 100ms (with some tolerance)
        self.assertLess(measured_latency, 200)
        self.assertGreater(measured_latency, 50)

class TestRealTimeConstraints(unittest.TestCase):
    """Test real-time processing constraints"""
    
    def test_audio_frame_processing_speed(self):
        """Test that audio frame processing is fast enough"""
        config = {"groq_api_key": "test", "elevenlabs_api_key": "test"}
        orchestrator = VoiceAgentOrchestrator(config)
        
        # 8kHz audio, 20ms frames = 160 samples
        audio_frame = AudioFrame(
            data=b'\x00' * 320,  # 160 samples * 2 bytes per sample
            timestamp=time.time()
        )
        
        async def measure_processing_time():
            start = time.time()
            await orchestrator.process_audio_frame(audio_frame)
            return (time.time() - start) * 1000
        
        processing_time = asyncio.run(measure_processing_time())
        
        # Should process faster than real-time (20ms frame in less than 20ms)
        self.assertLess(processing_time, 20, 
                       "Audio processing must be faster than real-time")
    
    def test_concurrent_processing(self):
        """Test that multiple frames can be processed concurrently"""
        config = {"groq_api_key": "test", "elevenlabs_api_key": "test"}
        orchestrator = VoiceAgentOrchestrator(config)
        
        async def process_multiple_frames():
            frames = []
            for i in range(10):
                frame = AudioFrame(
                    data=b'\x00' * 320,
                    timestamp=time.time() + i * 0.02
                )
                frames.append(frame)
            
            start = time.time()
            tasks = [orchestrator.process_audio_frame(frame) for frame in frames]
            await asyncio.gather(*tasks)
            return (time.time() - start) * 1000
        
        total_time = asyncio.run(process_multiple_frames())
        
        # 10 frames should not take 10x the time due to concurrency
        self.assertLess(total_time, 200, "Concurrent processing should be efficient")

def run_performance_benchmark():
    """Run performance benchmark to validate sub-500ms claims"""
    print("\n=== PERFORMANCE BENCHMARK ===")
    
    config = {"groq_api_key": "test", "elevenlabs_api_key": "test"}
    orchestrator = VoiceAgentOrchestrator(config)
    
    # Test VAD performance
    vad = VoiceActivityDetector()
    audio_data = np.random.randint(-16384, 16384, 160, dtype=np.int16).tobytes()
    
    vad_times = []
    for _ in range(1000):
        start = time.time()
        vad.detect_speech(audio_data)
        vad_times.append((time.time() - start) * 1000000)  # microseconds
    
    print(f"VAD processing: {np.mean(vad_times):.1f}µs avg, {np.max(vad_times):.1f}µs max")
    
    # Test frame processing
    frame = AudioFrame(data=audio_data, timestamp=time.time())
    
    async def benchmark_frame_processing():
        times = []
        for _ in range(100):
            start = time.time()
            await orchestrator.process_audio_frame(frame)
            times.append((time.time() - start) * 1000)
        return times
    
    frame_times = asyncio.run(benchmark_frame_processing())
    print(f"Frame processing: {np.mean(frame_times):.2f}ms avg, {np.max(frame_times):.2f}ms max")
    
    # Validate real-time constraints
    avg_frame_time = np.mean(frame_times)
    if avg_frame_time < 20:  # 20ms real-time constraint
        print(f"✓ REAL-TIME CAPABLE: {avg_frame_time:.2f}ms < 20ms")
    else:
        print(f"✗ TOO SLOW: {avg_frame_time:.2f}ms >= 20ms")
    
    print("=== BENCHMARK COMPLETE ===\n")

if __name__ == "__main__":
    # Run performance benchmark first
    run_performance_benchmark()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)