#!/usr/bin/env python3
"""
WiFi-AI Sensing POC
Demonstration of WiFi signal analysis for human presence detection and basic pose estimation
Inspired by RuView and emerging WiFi-based computer vision research

This POC simulates CSI (Channel State Information) processing for:
- Human presence detection
- Basic pose classification (standing, sitting, lying)
- Privacy-preserving vital sign estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fft import fft, fftfreq
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class WiFiAISensor:
    """WiFi-based AI sensing system using CSI analysis"""
    
    def __init__(self, sampling_rate=1000, num_antennas=3, num_subcarriers=30):
        self.sampling_rate = sampling_rate
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        
        # Pose classification model (simplified neural network weights)
        self.pose_weights = {
            'standing': np.array([0.7, 0.2, 0.1, 0.8, 0.3]),
            'sitting': np.array([0.3, 0.8, 0.4, 0.2, 0.7]),
            'lying': np.array([0.1, 0.3, 0.9, 0.1, 0.2])
        }
        
    def generate_csi_data(self, duration=10, scenario='standing'):
        """Generate synthetic CSI data for different scenarios"""
        samples = int(duration * self.sampling_rate)
        time = np.linspace(0, duration, samples)
        
        # Base WiFi signal (2.4 GHz simulation)
        base_freq = 2.4e9
        
        csi_data = np.zeros((samples, self.num_antennas, self.num_subcarriers), dtype=complex)
        
        for ant in range(self.num_antennas):
            for sc in range(self.num_subcarriers):
                # Base signal
                signal = np.exp(1j * 2 * np.pi * base_freq * time)
                
                # Add scenario-specific modulations
                if scenario == 'standing':
                    # Small breathing variations
                    breathing = 0.1 * np.sin(2 * np.pi * 0.3 * time)  # 18 bpm
                    # Micro-movements
                    micro_motion = 0.05 * np.sin(2 * np.pi * 2 * time) * np.random.random()
                    
                elif scenario == 'sitting':
                    # Stronger breathing, less body sway
                    breathing = 0.15 * np.sin(2 * np.pi * 0.25 * time)  # 15 bpm
                    micro_motion = 0.02 * np.sin(2 * np.pi * 0.5 * time)
                    
                elif scenario == 'lying':
                    # Minimal movement, primary breathing
                    breathing = 0.2 * np.sin(2 * np.pi * 0.2 * time)  # 12 bpm
                    micro_motion = 0.01 * np.random.normal(0, 0.1, samples)
                    
                else:  # empty room
                    breathing = 0
                    micro_motion = np.random.normal(0, 0.02, samples)
                
                # Combine modulations
                modulation = breathing + micro_motion
                signal *= np.exp(1j * modulation)
                
                # Add noise
                noise = 0.1 * (np.random.normal(0, 1, samples) + 1j * np.random.normal(0, 1, samples))
                csi_data[:, ant, sc] = signal + noise
                
        return csi_data, time
    
    def extract_features(self, csi_data):
        """Extract features from CSI data for AI processing"""
        # Amplitude and phase features
        amplitude = np.abs(csi_data)
        phase = np.angle(csi_data)
        
        # Statistical features
        features = []
        
        # Amplitude statistics
        features.extend([
            np.mean(amplitude),
            np.std(amplitude),
            np.var(amplitude),
            np.max(amplitude) - np.min(amplitude)
        ])
        
        # Phase statistics
        phase_diff = np.diff(phase, axis=0)
        features.extend([
            np.mean(phase_diff),
            np.std(phase_diff)
        ])
        
        # Cross-antenna correlation
        for ant1 in range(self.num_antennas):
            for ant2 in range(ant1 + 1, self.num_antennas):
                corr = np.corrcoef(amplitude[:, ant1, 0], amplitude[:, ant2, 0])[0, 1]
                features.append(corr if not np.isnan(corr) else 0)
        
        # Frequency domain features
        fft_amp = np.abs(fft(amplitude[:, 0, 0]))
        features.extend([
            np.sum(fft_amp[:10]),  # Low frequency energy
            np.sum(fft_amp[10:50])  # Mid frequency energy
        ])
        
        return np.array(features[:10])  # Normalize to fixed size
    
    def detect_presence(self, csi_data):
        """Detect human presence using variance-based threshold"""
        amplitude = np.abs(csi_data)
        variance = np.var(amplitude)
        
        # Threshold determined empirically
        threshold = 0.015
        presence = variance > threshold
        confidence = min(variance / threshold, 1.0)
        
        return presence, confidence
    
    def classify_pose(self, features):
        """Classify human pose using simplified neural network"""
        if len(features) < 5:
            features = np.pad(features, (0, 5 - len(features)), 'constant')
        
        scores = {}
        for pose, weights in self.pose_weights.items():
            # Simplified dot product similarity
            score = np.dot(features[:5], weights)
            scores[pose] = max(0, score)  # ReLU activation
        
        # Softmax normalization
        total = sum(scores.values()) + 1e-10
        probabilities = {pose: score/total for pose, score in scores.items()}
        
        best_pose = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_pose]
        
        return best_pose, probabilities, confidence
    
    def estimate_vitals(self, csi_data, time):
        """Estimate breathing rate from CSI variations"""
        # Use amplitude variations for breathing detection
        amplitude = np.abs(csi_data[:, 0, 0])  # First antenna, first subcarrier
        
        # Bandpass filter for breathing (0.1-0.5 Hz, 6-30 bpm)
        nyquist = self.sampling_rate / 2
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, amplitude)
        
        # Peak detection for breathing rate
        fft_filtered = np.abs(fft(filtered))
        freqs = fftfreq(len(filtered), 1/self.sampling_rate)
        
        # Find dominant frequency in breathing range
        breathing_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if np.any(breathing_mask):
            breathing_idx = np.argmax(fft_filtered[breathing_mask])
            breathing_freq = freqs[breathing_mask][breathing_idx]
            breathing_rate = breathing_freq * 60  # Convert to BPM
        else:
            breathing_rate = 0
        
        return {
            'breathing_rate_bpm': max(0, breathing_rate),
            'signal_quality': np.std(filtered)
        }
    
    def analyze_scene(self, duration=10, scenario='standing'):
        """Complete scene analysis pipeline"""
        print(f"🔍 Analyzing WiFi signals for {duration}s ({scenario} scenario)...")
        
        # Generate/collect CSI data
        csi_data, time = self.generate_csi_data(duration, scenario)
        
        # Extract features
        features = self.extract_features(csi_data)
        
        # Detect presence
        presence, presence_conf = self.detect_presence(csi_data)
        
        # Classify pose (if present)
        if presence:
            pose, pose_probs, pose_conf = self.classify_pose(features)
            vitals = self.estimate_vitals(csi_data, time)
        else:
            pose, pose_probs, pose_conf = "none", {}, 0
            vitals = {'breathing_rate_bpm': 0, 'signal_quality': 0}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'presence': {
                'detected': presence,
                'confidence': presence_conf
            },
            'pose': {
                'classification': pose,
                'confidence': pose_conf,
                'probabilities': pose_probs
            },
            'vitals': vitals,
            'features': features.tolist()
        }


def demo_wifi_ai_sensing():
    """Demonstrate WiFi-AI sensing capabilities"""
    print("🌊 WiFi-AI Sensing POC - Privacy-Preserving Computer Vision")
    print("=" * 60)
    
    sensor = WiFiAISensor()
    
    scenarios = ['empty', 'standing', 'sitting', 'lying']
    results = []
    
    for scenario in scenarios:
        print(f"\n📡 Scenario: {scenario}")
        result = sensor.analyze_scene(duration=5, scenario=scenario)
        results.append(result)
        
        # Display results
        presence = result['presence']
        pose = result['pose']
        vitals = result['vitals']
        
        print(f"   Presence: {'✅ Detected' if presence['detected'] else '❌ Empty'} "
              f"(confidence: {presence['confidence']:.2f})")
        
        if presence['detected']:
            print(f"   Pose: {pose['classification'].title()} "
                  f"(confidence: {pose['confidence']:.2f})")
            print(f"   Breathing: {vitals['breathing_rate_bpm']:.1f} BPM")
            
            # Show pose probabilities
            print("   Pose Probabilities:")
            for p, prob in pose['probabilities'].items():
                print(f"     {p.title()}: {prob:.2f}")
    
    # Save results
    with open('/Users/zak/.openclaw/workspace/ai-poc-daily/2026-03-03/sensing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to sensing_results.json")
    
    # Create visualization
    create_visualization(sensor)
    
    return results


def create_visualization(sensor):
    """Create visualization of WiFi sensing data"""
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('WiFi-AI Sensing Analysis', fontsize=16)
    
    scenarios = ['empty', 'standing', 'sitting', 'lying']
    colors = ['gray', 'blue', 'green', 'red']
    
    # Plot 1: CSI amplitude over time
    for i, scenario in enumerate(scenarios):
        csi_data, time = sensor.generate_csi_data(duration=3, scenario=scenario)
        amplitude = np.abs(csi_data[:, 0, 0])
        ax1.plot(time, amplitude, label=scenario.title(), color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CSI Amplitude')
    ax1.set_title('CSI Signal Variations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature comparison
    features_data = []
    for scenario in scenarios:
        csi_data, _ = sensor.generate_csi_data(duration=2, scenario=scenario)
        features = sensor.extract_features(csi_data)
        features_data.append(features[:5])  # First 5 features
    
    features_array = np.array(features_data)
    im = ax2.imshow(features_array.T, aspect='auto', cmap='viridis')
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.title() for s in scenarios])
    ax2.set_ylabel('Feature Index')
    ax2.set_title('Feature Profiles')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Breathing rate estimation
    breathing_rates = []
    for scenario in scenarios:
        csi_data, time = sensor.generate_csi_data(duration=5, scenario=scenario)
        vitals = sensor.estimate_vitals(csi_data, time)
        breathing_rates.append(vitals['breathing_rate_bpm'])
    
    ax3.bar(scenarios, breathing_rates, color=colors)
    ax3.set_ylabel('Breathing Rate (BPM)')
    ax3.set_title('Estimated Breathing Rates')
    ax3.set_xticklabels([s.title() for s in scenarios], rotation=45)
    
    # Plot 4: Pose classification confidence
    pose_confidences = {'standing': [], 'sitting': [], 'lying': []}
    
    for scenario in scenarios:
        csi_data, _ = sensor.generate_csi_data(duration=2, scenario=scenario)
        features = sensor.extract_features(csi_data)
        pose, probs, conf = sensor.classify_pose(features)
        
        for pose_type in pose_confidences.keys():
            pose_confidences[pose_type].append(probs.get(pose_type, 0))
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, (pose_type, confidences) in enumerate(pose_confidences.items()):
        ax4.bar(x + i * width, confidences, width, label=pose_type.title())
    
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Classification Confidence')
    ax4.set_title('Pose Classification Results')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([s.title() for s in scenarios])
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/zak/.openclaw/workspace/ai-poc-daily/2026-03-03/wifi_sensing_analysis.png', 
                dpi=150, bbox_inches='tight')
    print("📊 Visualization saved to wifi_sensing_analysis.png")


if __name__ == "__main__":
    demo_wifi_ai_sensing()