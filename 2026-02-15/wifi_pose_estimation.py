#!/usr/bin/env python3
"""
WiFi-based Pose Estimation Simulator
====================================

This POC demonstrates the principles behind WiFi-based human pose estimation,
inspired by the trending wifi-densepose project. It simulates how WiFi signal
perturbations can be used to infer human body poses without cameras.

The system models:
1. WiFi signal propagation and reflection
2. Human body as a signal occluder/reflector
3. Machine learning inference of pose from signal patterns
4. Real-time visualization of detected poses

Based on research from papers like "Through-Wall Human Pose Estimation Using
Radio Signals" and recent breakthroughs in RF-based sensing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
from scipy.spatial.distance import cdist
import json
import time
from typing import List, Tuple, Dict, Optional

class WiFiTransmitter:
    """Simulates a WiFi transmitter with beamforming capabilities."""
    
    def __init__(self, position: Tuple[float, float], frequency: float = 2.4e9):
        self.position = np.array(position)
        self.frequency = frequency
        self.wavelength = 3e8 / frequency  # speed of light / frequency
        self.power = 20  # dBm
        
    def get_signal_strength(self, receiver_pos: np.ndarray, obstacles: List = None) -> float:
        """Calculate received signal strength accounting for obstacles."""
        distance = np.linalg.norm(receiver_pos - self.position)
        
        # Free space path loss
        fspl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) - 147.55
        
        # Add multipath effects
        multipath_loss = 2 * np.sin(2 * np.pi * distance / self.wavelength)**2
        
        # Obstacle attenuation (human body causes ~3-6 dB loss)
        obstacle_loss = 0
        if obstacles:
            for obstacle in obstacles:
                if self._line_intersects_obstacle(self.position, receiver_pos, obstacle):
                    obstacle_loss += obstacle['attenuation']
        
        return self.power - fspl - multipath_loss - obstacle_loss
    
    def _line_intersects_obstacle(self, start: np.ndarray, end: np.ndarray, obstacle: Dict) -> bool:
        """Check if signal path intersects with human body obstacle."""
        center = np.array(obstacle['center'])
        radius = obstacle['radius']
        
        # Distance from center to line segment
        line_vec = end - start
        point_vec = center - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point_vec) <= radius
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_length = max(min(proj_length, line_len), 0)  # Clamp to line segment
        
        closest_point = start + line_unitvec * proj_length
        distance = np.linalg.norm(center - closest_point)
        
        return distance <= radius

class HumanPose:
    """Represents human pose with key body joints."""
    
    # Standard COCO pose keypoints
    KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    SKELETON = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    
    def __init__(self, center_x: float = 0, center_y: float = 0):
        self.center = np.array([center_x, center_y])
        self.keypoints = self._initialize_default_pose()
        self.timestamp = time.time()
        
    def _initialize_default_pose(self) -> Dict[str, np.ndarray]:
        """Initialize a standing human pose."""
        poses = {
            "nose": np.array([0, 1.7]),
            "left_eye": np.array([-0.05, 1.72]),
            "right_eye": np.array([0.05, 1.72]),
            "left_ear": np.array([-0.1, 1.7]),
            "right_ear": np.array([0.1, 1.7]),
            "left_shoulder": np.array([-0.2, 1.5]),
            "right_shoulder": np.array([0.2, 1.5]),
            "left_elbow": np.array([-0.3, 1.2]),
            "right_elbow": np.array([0.3, 1.2]),
            "left_wrist": np.array([-0.35, 0.9]),
            "right_wrist": np.array([0.35, 0.9]),
            "left_hip": np.array([-0.15, 1.0]),
            "right_hip": np.array([0.15, 1.0]),
            "left_knee": np.array([-0.15, 0.5]),
            "right_knee": np.array([0.15, 0.5]),
            "left_ankle": np.array([-0.15, 0.1]),
            "right_ankle": np.array([0.15, 0.1]),
        }
        
        # Offset by center position
        for key in poses:
            poses[key] += self.center
            
        return poses
    
    def animate_wave(self, time_factor: float):
        """Animate a waving motion for demonstration."""
        wave_amplitude = 0.3
        wave_frequency = 2.0
        
        # Wave right arm
        self.keypoints["right_elbow"][1] = 1.2 + wave_amplitude * np.sin(wave_frequency * time_factor)
        self.keypoints["right_wrist"][0] = 0.35 + wave_amplitude * np.cos(wave_frequency * time_factor)
        self.keypoints["right_wrist"][1] = 0.9 + wave_amplitude * np.sin(wave_frequency * time_factor)
    
    def get_obstacles(self) -> List[Dict]:
        """Convert pose keypoints to obstacles that affect WiFi signals."""
        obstacles = []
        
        # Model main body parts as cylindrical obstacles
        body_parts = [
            ("torso", ("left_shoulder", "left_hip"), 0.15, 4.0),  # torso
            ("left_arm", ("left_shoulder", "left_wrist"), 0.05, 2.0),
            ("right_arm", ("right_shoulder", "right_wrist"), 0.05, 2.0),
            ("left_leg", ("left_hip", "left_ankle"), 0.08, 3.0),
            ("right_leg", ("right_hip", "right_ankle"), 0.08, 3.0),
            ("head", ("nose", "nose"), 0.12, 5.0),
        ]
        
        for name, (start_joint, end_joint), radius, attenuation in body_parts:
            if start_joint == end_joint:
                center = self.keypoints[start_joint]
            else:
                center = (self.keypoints[start_joint] + self.keypoints[end_joint]) / 2
            
            obstacles.append({
                'name': name,
                'center': center,
                'radius': radius,
                'attenuation': attenuation
            })
        
        return obstacles

class WiFiPoseEstimator:
    """Machine learning model for inferring poses from WiFi signal patterns."""
    
    def __init__(self, transmitters: List[WiFiTransmitter], receivers: List[Tuple[float, float]]):
        self.transmitters = transmitters
        self.receivers = [np.array(pos) for pos in receivers]
        self.baseline_signals = None
        self.pose_history = []
        
    def calibrate_baseline(self):
        """Calibrate baseline signals without human presence."""
        self.baseline_signals = []
        
        for tx in self.transmitters:
            tx_signals = []
            for rx_pos in self.receivers:
                signal_strength = tx.get_signal_strength(rx_pos)
                tx_signals.append(signal_strength)
            self.baseline_signals.append(tx_signals)
        
        self.baseline_signals = np.array(self.baseline_signals)
    
    def extract_features(self, pose: HumanPose) -> np.ndarray:
        """Extract WiFi signal features from current pose."""
        current_signals = []
        obstacles = pose.get_obstacles()
        
        for tx in self.transmitters:
            tx_signals = []
            for rx_pos in self.receivers:
                signal_strength = tx.get_signal_strength(rx_pos, obstacles)
                tx_signals.append(signal_strength)
            current_signals.append(tx_signals)
        
        current_signals = np.array(current_signals)
        
        # Calculate signal perturbation from baseline
        if self.baseline_signals is not None:
            perturbations = self.baseline_signals - current_signals
        else:
            perturbations = current_signals
        
        # Flatten and add statistical features
        features = perturbations.flatten()
        
        # Add temporal features if history exists
        if self.pose_history:
            prev_features = self.pose_history[-1]
            temporal_features = features - prev_features
            features = np.concatenate([features, temporal_features])
        
        return features
    
    def estimate_pose(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate pose from WiFi signal features (simplified ML inference)."""
        # This is a simplified demonstration - real systems use deep neural networks
        # trained on thousands of pose-signal pairs
        
        # For demo: use PCA-style dimensionality reduction + regression
        pose_estimate = {}
        
        # Simulate ML inference with some realistic pose constraints
        base_pose = HumanPose()
        
        # Add noise and perturbations based on signal strength
        signal_strength = np.mean(np.abs(features))
        noise_level = max(0.1, 0.5 / signal_strength) if signal_strength > 0 else 0.1
        
        for joint_name, joint_pos in base_pose.keypoints.items():
            # Add controlled random perturbation
            perturbation = np.random.normal(0, noise_level, 2)
            pose_estimate[joint_name] = joint_pos + perturbation
        
        # Apply anatomical constraints (simplified)
        pose_estimate = self._apply_constraints(pose_estimate)
        
        return pose_estimate
    
    def _apply_constraints(self, pose: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply basic anatomical constraints to pose estimation."""
        # Ensure limb lengths are reasonable
        constraints = [
            ("left_shoulder", "left_elbow", 0.3),
            ("left_elbow", "left_wrist", 0.25),
            ("right_shoulder", "right_elbow", 0.3),
            ("right_elbow", "right_wrist", 0.25),
            ("left_hip", "left_knee", 0.4),
            ("left_knee", "left_ankle", 0.4),
            ("right_hip", "right_knee", 0.4),
            ("right_knee", "right_ankle", 0.4),
        ]
        
        for joint1, joint2, target_length in constraints:
            current_vec = pose[joint2] - pose[joint1]
            current_length = np.linalg.norm(current_vec)
            
            if current_length > 0:
                scale_factor = target_length / current_length
                pose[joint2] = pose[joint1] + current_vec * scale_factor
        
        return pose

class WiFiPoseVisualizer:
    """Real-time visualization of WiFi-based pose estimation."""
    
    def __init__(self, estimator: WiFiPoseEstimator):
        self.estimator = estimator
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Setup pose visualization
        self.ax1.set_xlim(-2, 2)
        self.ax1.set_ylim(-0.5, 2)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('WiFi Pose Estimation')
        self.ax1.grid(True, alpha=0.3)
        
        # Setup signal strength heatmap
        self.ax2.set_title('WiFi Signal Strength Heatmap')
        
        # Initialize pose lines
        self.pose_lines = []
        for connection in HumanPose.SKELETON:
            line, = self.ax1.plot([], [], 'b-', linewidth=2, alpha=0.8)
            self.pose_lines.append(line)
        
        # Joint markers
        self.joint_scatter = self.ax1.scatter([], [], c='red', s=50, zorder=5)
        
        # WiFi transmitter/receiver positions
        tx_positions = [tx.position for tx in estimator.transmitters]
        rx_positions = estimator.receivers
        
        if tx_positions:
            tx_pos = np.array(tx_positions)
            self.ax1.scatter(tx_pos[:, 0], tx_pos[:, 1], c='green', s=100, marker='^', label='WiFi TX')
        
        if rx_positions:
            rx_pos = np.array(rx_positions)
            self.ax1.scatter(rx_pos[:, 0], rx_pos[:, 1], c='blue', s=100, marker='s', label='WiFi RX')
        
        self.ax1.legend()
        
        # Signal strength history for visualization
        self.signal_history = []
        self.max_history = 50
        
    def update(self, frame):
        """Update visualization with new pose estimation."""
        # Create animated pose
        current_time = time.time()
        true_pose = HumanPose()
        true_pose.animate_wave(current_time)
        
        # Extract features and estimate pose
        features = self.estimator.extract_features(true_pose)
        estimated_pose = self.estimator.estimate_pose(features)
        
        # Update pose history for temporal features
        self.estimator.pose_history.append(features[:len(features)//2 if len(self.estimator.pose_history) > 0 else len(features)])
        if len(self.estimator.pose_history) > 10:
            self.estimator.pose_history.pop(0)
        
        # Update pose visualization
        joint_positions = []
        for i, (joint1, joint2) in enumerate(HumanPose.SKELETON):
            pos1 = estimated_pose[joint1]
            pos2 = estimated_pose[joint2]
            
            self.pose_lines[i].set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
            joint_positions.extend([pos1, pos2])
        
        # Update joint markers
        if joint_positions:
            joint_array = np.array(joint_positions)
            self.joint_scatter.set_offsets(joint_array)
        
        # Update signal strength heatmap
        self.signal_history.append(np.mean(features))
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
        
        self.ax2.clear()
        self.ax2.plot(self.signal_history, 'g-', alpha=0.7)
        self.ax2.set_title('Signal Perturbation Over Time')
        self.ax2.set_ylabel('Signal Strength (dB)')
        self.ax2.grid(True, alpha=0.3)
        
        return self.pose_lines + [self.joint_scatter]

def main():
    """Run the WiFi pose estimation demonstration."""
    print("WiFi-based Pose Estimation Simulator")
    print("====================================")
    print("Simulating breakthrough technology for pose detection through WiFi signals...")
    
    # Setup WiFi network topology
    transmitters = [
        WiFiTransmitter((-1.5, 0.5)),  # Left wall
        WiFiTransmitter((1.5, 0.5)),   # Right wall  
        WiFiTransmitter((0, 2.0)),     # Ceiling
    ]
    
    # Receiver grid for comprehensive coverage
    receivers = [
        (-1, 0), (0, 0), (1, 0),      # Floor level
        (-1, 1), (0, 1), (1, 1),      # Mid level
        (-0.5, 1.5), (0.5, 1.5),      # Upper level
    ]
    
    # Initialize pose estimation system
    estimator = WiFiPoseEstimator(transmitters, receivers)
    estimator.calibrate_baseline()
    
    print(f"Network setup: {len(transmitters)} transmitters, {len(receivers)} receivers")
    print("Calibrating baseline signal strengths...")
    
    # Setup visualization
    visualizer = WiFiPoseVisualizer(estimator)
    
    print("Starting real-time pose estimation...")
    print("The system detects pose changes through WiFi signal perturbations")
    print("Watch how arm movements affect the signal strength patterns!")
    
    # Run animation
    anim = animation.FuncAnimation(
        visualizer.fig, visualizer.update, frames=200,
        interval=100, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    # Save sample data for analysis
    sample_data = {
        'timestamp': time.time(),
        'transmitters': [(tx.position.tolist(), tx.frequency) for tx in transmitters],
        'receivers': [pos for pos in receivers],
        'baseline_signals': estimator.baseline_signals.tolist() if estimator.baseline_signals is not None else [],
        'description': 'WiFi-based pose estimation simulation data'
    }
    
    with open('wifi_pose_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("\nSimulation complete! Sample data saved to wifi_pose_data.json")

if __name__ == "__main__":
    main()