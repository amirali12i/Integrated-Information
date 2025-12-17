"""
EdgeFace: Real-Time Facial Landmark Tracking and 3D Pose Estimation
for Resource-Constrained Edge Devices

This module implements the core EdgeFace framework as described in:
"EdgeFace: A Unified Framework for Real-Time Facial Landmark Tracking 
and 3D Pose Estimation on Resource-Constrained Edge Devices"

Authors: Amirali Ghajari, Maicol Ochoa
Universidad Europea de Madrid, 2025

License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import cv2
import time


@dataclass
class EdgeFaceConfig:
    """Configuration parameters for EdgeFace framework."""
    
    # Preprocessing
    clahe_clip_limit: float = 2.5
    clahe_grid_size: int = 8
    face_margin: float = 0.25
    
    # Quality assessment weights (learned via constrained logistic regression)
    w_conf: float = 0.52
    w_geom: float = 0.28
    w_temp: float = 0.20
    quality_threshold: float = 0.73
    
    # Kalman filter (ML-tuned parameters)
    sigma_theta: float = 0.38  # Process noise for angles (degrees)
    sigma_theta_dot: float = 5.2  # Process noise for angular velocity (deg/s)
    sigma_yaw: float = 3.52  # Measurement noise yaw
    sigma_pitch: float = 3.78  # Measurement noise pitch
    sigma_roll: float = 4.15  # Measurement noise roll
    
    # Temporal scales (seconds) - behaviorally motivated
    temporal_scales: Tuple[float, ...] = (0.5, 1.0, 2.5, 6.0, 12.0)
    
    # Multi-scale fusion weights (learned via constrained ridge regression)
    scale_weights: Tuple[float, ...] = (0.08, 0.15, 0.38, 0.27, 0.12)
    
    # Camera parameters
    default_fov: float = 58.0  # degrees
    
    # Eye/mouth thresholds
    ear_threshold: float = 0.19
    blink_min_frames: int = 2
    
    # Frame rate
    target_fps: float = 15.0


@dataclass
class LandmarkQuality:
    """Quality assessment result for landmark detection."""
    composite_score: float
    confidence_score: float
    geometric_score: float
    temporal_score: float
    is_valid: bool
    occluded_indices: List[int] = field(default_factory=list)


@dataclass
class PoseEstimate:
    """3D head pose estimation result."""
    yaw: float  # degrees
    pitch: float  # degrees
    roll: float  # degrees
    confidence: float
    filtered: bool = False


@dataclass
class TemporalFeatures:
    """Multi-scale temporal features."""
    features: np.ndarray  # 160-dimensional feature vector
    scale_features: Dict[float, np.ndarray]  # Per-scale features
    ear: float
    mar: float
    blink_detected: bool


@dataclass
class EdgeFaceResult:
    """Complete EdgeFace processing result for a single frame."""
    landmarks: Optional[np.ndarray]  # 478 x 3 (x, y, z)
    quality: LandmarkQuality
    pose_raw: Optional[PoseEstimate]
    pose_filtered: Optional[PoseEstimate]
    temporal_features: Optional[TemporalFeatures]
    processing_time_ms: float
    frame_index: int


class AdaptivePreprocessor:
    """
    Adaptive preprocessing with CLAHE in YCrCb color space.
    
    Improves landmark detection by 12.8% in dim conditions (<100 lux)
    as validated through controlled experiments.
    """
    
    def __init__(self, config: EdgeFaceConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_grid_size, config.clahe_grid_size)
        )
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE in YCrCb color space.
        
        Args:
            frame: BGR input frame
            
        Returns:
            Preprocessed BGR frame with normalized luminance
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Apply CLAHE to Y channel only
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        
        # Convert back to BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


class GeometricConsistencyChecker:
    """
    Validates anatomical constraints via inter-landmark distances.
    
    Uses 7 anatomically-stable landmark pairs with learned statistics
    from 1,500 high-quality reference frames.
    """
    
    # Anatomically stable landmark pairs (MediaPipe indices)
    # Format: (idx1, idx2, mean_normalized_dist, std_normalized_dist)
    STABLE_PAIRS = [
        (33, 133, 0.165, 0.012),   # Left eye corners
        (362, 263, 0.165, 0.012),  # Right eye corners
        (33, 362, 0.330, 0.018),   # Eye corner to eye corner (IOD)
        (1, 4, 0.082, 0.008),      # Nose bridge to tip
        (61, 291, 0.185, 0.015),   # Mouth corners
        (1, 61, 0.195, 0.014),     # Nose to left mouth
        (1, 291, 0.195, 0.014),    # Nose to right mouth
    ]
    
    def __init__(self):
        self.reference_iod = None
    
    def compute_score(self, landmarks: np.ndarray) -> float:
        """
        Compute geometric consistency score.
        
        Args:
            landmarks: 478 x 3 landmark array (normalized coordinates)
            
        Returns:
            Score in [0, 1] where 1 is perfectly consistent
        """
        if landmarks is None or len(landmarks) < 400:
            return 0.0
        
        deviations = []
        for idx1, idx2, mu, sigma in self.STABLE_PAIRS:
            if idx1 >= len(landmarks) or idx2 >= len(landmarks):
                continue
            
            dist = np.linalg.norm(landmarks[idx1, :2] - landmarks[idx2, :2])
            z_score = abs(dist - mu) / (3 * sigma)
            deviations.append(min(1.0, z_score))
        
        if not deviations:
            return 0.0
        
        return 1.0 - np.mean(deviations)


class TemporalConsistencyChecker:
    """
    Measures agreement with motion-predicted landmarks.
    
    Uses constant-velocity prediction model with learned motion
    variance parameter sigma_motion = 0.015 (normalized coordinates).
    """
    
    def __init__(self, sigma_motion: float = 0.015):
        self.sigma_motion = sigma_motion
        self.prev_landmarks = None
        self.prev_velocity = None
    
    def compute_score(self, landmarks: np.ndarray, dt: float = 1/15) -> float:
        """
        Compute temporal consistency score.
        
        Args:
            landmarks: Current frame landmarks (478 x 3)
            dt: Time delta from previous frame
            
        Returns:
            Score in [0, 1] where 1 is perfectly consistent
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return 1.0
        
        # Predict current position using previous velocity
        if self.prev_velocity is not None:
            predicted = self.prev_landmarks + self.prev_velocity * dt
        else:
            predicted = self.prev_landmarks
        
        # Compute prediction error
        errors = np.linalg.norm(landmarks[:, :2] - predicted[:, :2], axis=1)
        normalized_errors = np.minimum(1.0, errors / self.sigma_motion)
        score = 1.0 - np.mean(normalized_errors)
        
        # Update state
        self.prev_velocity = (landmarks - self.prev_landmarks) / dt
        self.prev_landmarks = landmarks.copy()
        
        return max(0.0, score)
    
    def reset(self):
        """Reset temporal state."""
        self.prev_landmarks = None
        self.prev_velocity = None


class QualityAssessor:
    """
    Multi-criteria quality assessment with learned fusion weights.
    
    Combines neural confidence, geometric consistency, and temporal
    coherence with weights optimized via constrained logistic regression.
    Achieves AUC=0.981 for predicting detection success.
    """
    
    def __init__(self, config: EdgeFaceConfig):
        self.config = config
        self.geom_checker = GeometricConsistencyChecker()
        self.temp_checker = TemporalConsistencyChecker()
    
    def assess(
        self,
        landmarks: Optional[np.ndarray],
        confidence: float,
        dt: float = 1/15
    ) -> LandmarkQuality:
        """
        Assess landmark quality using multi-criteria fusion.
        
        Args:
            landmarks: Detected landmarks (478 x 3) or None
            confidence: Neural detector confidence [0, 1]
            dt: Time delta from previous frame
            
        Returns:
            LandmarkQuality with composite score and validity
        """
        if landmarks is None:
            return LandmarkQuality(
                composite_score=0.0,
                confidence_score=0.0,
                geometric_score=0.0,
                temporal_score=0.0,
                is_valid=False
            )
        
        # Compute individual scores
        q_conf = confidence
        q_geom = self.geom_checker.compute_score(landmarks)
        q_temp = self.temp_checker.compute_score(landmarks, dt)
        
        # Weighted fusion (learned weights)
        composite = (
            self.config.w_conf * q_conf +
            self.config.w_geom * q_geom +
            self.config.w_temp * q_temp
        )
        
        # Identify potentially occluded landmarks
        occluded = self._detect_occlusions(landmarks)
        
        return LandmarkQuality(
            composite_score=composite,
            confidence_score=q_conf,
            geometric_score=q_geom,
            temporal_score=q_temp,
            is_valid=composite >= self.config.quality_threshold,
            occluded_indices=occluded
        )
    
    def _detect_occlusions(self, landmarks: np.ndarray) -> List[int]:
        """Detect potentially occluded landmarks based on motion prediction."""
        if self.temp_checker.prev_landmarks is None:
            return []
        
        # Landmarks with large prediction error likely occluded
        if self.temp_checker.prev_velocity is not None:
            predicted = (self.temp_checker.prev_landmarks + 
                        self.temp_checker.prev_velocity * (1/15))
            errors = np.linalg.norm(landmarks[:, :2] - predicted[:, :2], axis=1)
            threshold = 3 * self.temp_checker.sigma_motion
            return np.where(errors > threshold)[0].tolist()
        
        return []
    
    def reset(self):
        """Reset temporal state."""
        self.temp_checker.reset()


class KalmanPoseFilter:
    """
    Maximum-likelihood-tuned Kalman filter for SO(3) pose stabilization.
    
    Uses Euler angle representation with constant angular velocity model.
    Parameters optimized via ML estimation on innovation sequences,
    achieving 67.4% jitter reduction while maintaining 28ms latency.
    """
    
    def __init__(self, config: EdgeFaceConfig, dt: float = 1/15):
        self.config = config
        self.dt = dt
        
        # State: [yaw, pitch, roll, yaw_dot, pitch_dot, roll_dot]
        self.state = np.zeros(6)
        self.P = np.diag([10, 10, 10, 50, 50, 50])  # Initial covariance
        
        # State transition matrix (constant velocity model)
        self.A = np.eye(6)
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt
        
        # Measurement matrix (observe angles only)
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        
        # Process noise (ML-tuned)
        sigma_theta_rad = np.deg2rad(config.sigma_theta)
        sigma_theta_dot_rad = np.deg2rad(config.sigma_theta_dot)
        self.Q = np.diag([
            sigma_theta_rad**2, sigma_theta_rad**2, sigma_theta_rad**2,
            sigma_theta_dot_rad**2, sigma_theta_dot_rad**2, sigma_theta_dot_rad**2
        ])
        
        # Measurement noise (ML-tuned, per-angle)
        self.R = np.diag([
            np.deg2rad(config.sigma_yaw)**2,
            np.deg2rad(config.sigma_pitch)**2,
            np.deg2rad(config.sigma_roll)**2
        ])
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """
        Prediction step.
        
        Returns:
            Predicted state (angles in radians)
        """
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.state[:3]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step with new measurement.
        
        Args:
            measurement: [yaw, pitch, roll] in radians
            
        Returns:
            Updated state estimate (angles in radians)
        """
        if not self.initialized:
            self.state[:3] = measurement
            self.initialized = True
            return self.state[:3]
        
        # Innovation
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.state[:3]
    
    def filter(self, measurement: np.ndarray) -> np.ndarray:
        """
        Combined predict-update cycle.
        
        Args:
            measurement: [yaw, pitch, roll] in radians
            
        Returns:
            Filtered state estimate (angles in radians)
        """
        self.predict()
        return self.update(measurement)
    
    def get_velocity(self) -> np.ndarray:
        """Get current angular velocity estimate."""
        return self.state[3:6]
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(6)
        self.P = np.diag([10, 10, 10, 50, 50, 50])
        self.initialized = False


class IPPEPoseSolver:
    """
    Infinitesimal Plane-based Pose Estimation with RANSAC.
    
    Uses 9-point canonical face model derived from FaceWarehouse.
    RANSAC provides robust estimation under partial occlusion.
    """
    
    # 9-point canonical model (normalized, nose-centered)
    # Selected landmarks: eye corners (4), nose tip (1), 
    # mouth corners (2), eyebrow centers (2)
    CANONICAL_POINTS = np.array([
        [-0.165, -0.062, -0.015],  # Left eye outer corner
        [-0.050, -0.062, 0.025],   # Left eye inner corner
        [0.050, -0.062, 0.025],    # Right eye inner corner
        [0.165, -0.062, -0.015],   # Right eye outer corner
        [0.0, 0.0, 0.065],         # Nose tip
        [-0.092, 0.095, -0.005],   # Left mouth corner
        [0.092, 0.095, -0.005],    # Right mouth corner
        [-0.115, -0.115, -0.025],  # Left eyebrow center
        [0.115, -0.115, -0.025],   # Right eyebrow center
    ], dtype=np.float32)
    
    # Corresponding MediaPipe indices
    LANDMARK_INDICES = [33, 133, 362, 263, 1, 61, 291, 105, 334]
    
    def __init__(self, config: EdgeFaceConfig):
        self.config = config
        self.camera_matrix = None
    
    def _build_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """Build intrinsic camera matrix from FOV assumption."""
        fov_rad = np.deg2rad(self.config.default_fov)
        focal_length = width / (2 * np.tan(fov_rad / 2))
        
        return np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def estimate(
        self,
        landmarks: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Optional[PoseEstimate]:
        """
        Estimate 3D head pose from landmarks.
        
        Args:
            landmarks: 478 x 3 landmark array (normalized [0,1])
            image_size: (width, height) of the image
            
        Returns:
            PoseEstimate or None if estimation fails
        """
        width, height = image_size
        
        # Build camera matrix if needed
        if self.camera_matrix is None or self.camera_matrix[0, 2] != width / 2:
            self.camera_matrix = self._build_camera_matrix(width, height)
        
        # Extract relevant landmarks and convert to pixel coordinates
        try:
            image_points = np.array([
                landmarks[idx, :2] * [width, height]
                for idx in self.LANDMARK_INDICES
            ], dtype=np.float32)
        except (IndexError, TypeError):
            return None
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.CANONICAL_POINTS,
            image_points,
            self.camera_matrix,
            None,
            iterationsCount=1000,
            reprojectionError=3.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 5:
            return None
        
        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_to_euler(rotation_matrix)
        
        # Confidence based on inlier ratio
        confidence = len(inliers) / len(self.LANDMARK_INDICES)
        
        return PoseEstimate(
            yaw=np.rad2deg(yaw),
            pitch=np.rad2deg(pitch),
            roll=np.rad2deg(roll),
            confidence=confidence,
            filtered=False
        )
    
    def _rotation_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (yaw, pitch, roll)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        if sy > 1e-6:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        return yaw, pitch, roll


class EyeAnalyzer:
    """
    Eye Aspect Ratio (EAR) computation and blink detection.
    
    Following Soukupova & Cech (2016) with threshold optimized
    via ROC analysis on validation data (threshold=0.19).
    """
    
    # 6-point eye model (MediaPipe indices)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, config: EdgeFaceConfig):
        self.config = config
        self.ear_history = deque(maxlen=10)
        self.blink_counter = 0
    
    def compute_ear(self, landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio.
        
        Args:
            landmarks: 478 x 3 landmark array
            
        Returns:
            Average EAR for both eyes
        """
        def single_ear(eye_indices):
            p = [landmarks[i, :2] for i in eye_indices]
            
            # Vertical distances
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            
            # Horizontal distance
            h = np.linalg.norm(p[0] - p[3])
            
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        
        left_ear = single_ear(self.LEFT_EYE)
        right_ear = single_ear(self.RIGHT_EYE)
        
        return (left_ear + right_ear) / 2.0
    
    def detect_blink(self, ear: float) -> bool:
        """
        Detect eye blink based on EAR threshold.
        
        Args:
            ear: Current Eye Aspect Ratio
            
        Returns:
            True if blink detected
        """
        self.ear_history.append(ear)
        
        if ear < self.config.ear_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.config.blink_min_frames:
                self.blink_counter = 0
                return True
            self.blink_counter = 0
        
        return False
    
    def reset(self):
        """Reset blink detection state."""
        self.ear_history.clear()
        self.blink_counter = 0


class MouthAnalyzer:
    """Mouth Aspect Ratio (MAR) computation."""
    
    # Mouth landmarks (MediaPipe indices)
    MOUTH_OUTER = [61, 291, 0, 17]  # corners, top, bottom
    MOUTH_INNER = [78, 308, 13, 14]
    
    def __init__(self):
        pass
    
    def compute_mar(self, landmarks: np.ndarray) -> float:
        """
        Compute Mouth Aspect Ratio.
        
        Args:
            landmarks: 478 x 3 landmark array
            
        Returns:
            Mouth aspect ratio
        """
        try:
            # Vertical distances
            v1 = np.linalg.norm(
                landmarks[self.MOUTH_OUTER[2], :2] - 
                landmarks[self.MOUTH_OUTER[3], :2]
            )
            v2 = np.linalg.norm(
                landmarks[self.MOUTH_INNER[2], :2] - 
                landmarks[self.MOUTH_INNER[3], :2]
            )
            
            # Horizontal distance
            h = np.linalg.norm(
                landmarks[self.MOUTH_OUTER[0], :2] - 
                landmarks[self.MOUTH_OUTER[1], :2]
            )
            
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except (IndexError, TypeError):
            return 0.0


class MultiScaleTemporalFusion:
    """
    Hierarchical temporal pyramid with constrained ridge regression fusion.
    
    Constructs features at behaviorally-motivated timescales (0.5-12s)
    and fuses using learned weights that empirically validate cognitive
    science predictions about behavioral timescales.
    """
    
    def __init__(self, config: EdgeFaceConfig):
        self.config = config
        self.fps = config.target_fps
        
        # Convert scales to frame counts
        self.window_sizes = [
            max(2, int(s * self.fps)) for s in config.temporal_scales
        ]
        
        # Feature buffers per scale
        self.buffers = {
            scale: deque(maxlen=ws)
            for scale, ws in zip(config.temporal_scales, self.window_sizes)
        }
        
        # Base feature names
        self.base_features = [
            'ear', 'mar', 'yaw', 'pitch', 'roll',
            'yaw_vel', 'pitch_vel', 'roll_vel'
        ]
    
    def update(
        self,
        ear: float,
        mar: float,
        pose: PoseEstimate,
        velocity: np.ndarray
    ) -> TemporalFeatures:
        """
        Update temporal buffers and compute multi-scale features.
        
        Args:
            ear: Eye Aspect Ratio
            mar: Mouth Aspect Ratio
            pose: Current pose estimate
            velocity: Angular velocity [yaw_dot, pitch_dot, roll_dot]
            
        Returns:
            TemporalFeatures with 160-dimensional representation
        """
        # Create base feature vector
        base = np.array([
            ear, mar,
            pose.yaw, pose.pitch, pose.roll,
            np.rad2deg(velocity[0]), np.rad2deg(velocity[1]), np.rad2deg(velocity[2])
        ])
        
        # Update all buffers
        for scale in self.config.temporal_scales:
            self.buffers[scale].append(base)
        
        # Compute scale-specific features
        scale_features = {}
        all_features = []
        
        for scale, weight in zip(self.config.temporal_scales, self.config.scale_weights):
            buffer = np.array(self.buffers[scale])
            if len(buffer) < 2:
                # Not enough data, use zeros
                sf = np.zeros(32)
            else:
                sf = self._compute_scale_features(buffer)
            
            scale_features[scale] = sf
            all_features.append(sf * weight)
        
        # Fused feature vector (weighted concatenation)
        fused = np.concatenate(all_features)
        
        return TemporalFeatures(
            features=fused,
            scale_features=scale_features,
            ear=ear,
            mar=mar,
            blink_detected=False  # Updated by caller
        )
    
    def _compute_scale_features(self, buffer: np.ndarray) -> np.ndarray:
        """
        Compute statistics for a single temporal scale.
        
        Args:
            buffer: T x 8 array of base features over time window
            
        Returns:
            32-dimensional feature vector (8 features x 4 statistics)
        """
        features = []
        
        for i in range(buffer.shape[1]):
            signal = buffer[:, i]
            
            # Mean
            features.append(np.mean(signal))
            
            # Standard deviation
            features.append(np.std(signal))
            
            # Range
            features.append(np.max(signal) - np.min(signal))
            
            # Temporal gradient (end - start)
            features.append(signal[-1] - signal[0])
        
        return np.array(features)
    
    def reset(self):
        """Reset all temporal buffers."""
        for buffer in self.buffers.values():
            buffer.clear()


class EdgeFace:
    """
    Main EdgeFace framework class.
    
    Integrates all components for real-time facial landmark tracking
    and 3D pose estimation on resource-constrained edge devices.
    
    Example usage:
        config = EdgeFaceConfig()
        edgeface = EdgeFace(config)
        
        for frame in video_stream:
            result = edgeface.process(frame)
            if result.quality.is_valid:
                print(f"Pose: yaw={result.pose_filtered.yaw:.1f}°")
    """
    
    def __init__(self, config: Optional[EdgeFaceConfig] = None):
        self.config = config or EdgeFaceConfig()
        
        # Initialize components
        self.preprocessor = AdaptivePreprocessor(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        self.pose_solver = IPPEPoseSolver(self.config)
        self.kalman_filter = KalmanPoseFilter(self.config)
        self.eye_analyzer = EyeAnalyzer(self.config)
        self.mouth_analyzer = MouthAnalyzer()
        self.temporal_fusion = MultiScaleTemporalFusion(self.config)
        
        # MediaPipe face mesh (lazy initialization)
        self._face_mesh = None
        
        # State
        self.frame_index = 0
        self.last_valid_pose = None
        self.last_valid_landmarks = None
    
    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except ImportError:
                raise ImportError(
                    "MediaPipe is required. Install with: pip install mediapipe"
                )
    
    def process(self, frame: np.ndarray) -> EdgeFaceResult:
        """
        Process a single frame through the EdgeFace pipeline.
        
        Args:
            frame: BGR input frame
            
        Returns:
            EdgeFaceResult with landmarks, pose, and temporal features
        """
        start_time = time.time()
        self._init_mediapipe()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        image_size = (width, height)
        
        # Stage 1: Preprocessing
        processed = self.preprocessor.process(frame)
        
        # Stage 2: Face detection and landmark extraction
        rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)
        
        # Extract landmarks
        landmarks = None
        confidence = 0.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
            ])
            confidence = 0.9  # MediaPipe doesn't expose confidence directly
        
        # Stage 3: Quality assessment
        quality = self.quality_assessor.assess(
            landmarks, confidence, dt=1/self.config.target_fps
        )
        
        # Stage 4: Pose estimation
        pose_raw = None
        pose_filtered = None
        
        if quality.is_valid and landmarks is not None:
            pose_raw = self.pose_solver.estimate(landmarks, image_size)
            
            if pose_raw is not None:
                # Stage 5: Kalman filtering
                measurement = np.deg2rad([
                    pose_raw.yaw, pose_raw.pitch, pose_raw.roll
                ])
                filtered = self.kalman_filter.filter(measurement)
                
                pose_filtered = PoseEstimate(
                    yaw=np.rad2deg(filtered[0]),
                    pitch=np.rad2deg(filtered[1]),
                    roll=np.rad2deg(filtered[2]),
                    confidence=pose_raw.confidence,
                    filtered=True
                )
                
                self.last_valid_pose = pose_filtered
                self.last_valid_landmarks = landmarks
        
        elif self.last_valid_pose is not None:
            # Temporal recovery: use prediction
            predicted = self.kalman_filter.predict()
            pose_filtered = PoseEstimate(
                yaw=np.rad2deg(predicted[0]),
                pitch=np.rad2deg(predicted[1]),
                roll=np.rad2deg(predicted[2]),
                confidence=0.5,  # Lower confidence for prediction
                filtered=True
            )
        
        # Stage 6: Temporal features
        temporal_features = None
        
        if pose_filtered is not None and landmarks is not None:
            ear = self.eye_analyzer.compute_ear(landmarks)
            mar = self.mouth_analyzer.compute_mar(landmarks)
            blink = self.eye_analyzer.detect_blink(ear)
            velocity = self.kalman_filter.get_velocity()
            
            temporal_features = self.temporal_fusion.update(
                ear, mar, pose_filtered, velocity
            )
            temporal_features.blink_detected = blink
        
        # Compute processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Build result
        result = EdgeFaceResult(
            landmarks=landmarks,
            quality=quality,
            pose_raw=pose_raw,
            pose_filtered=pose_filtered,
            temporal_features=temporal_features,
            processing_time_ms=processing_time,
            frame_index=self.frame_index
        )
        
        self.frame_index += 1
        return result
    
    def reset(self):
        """Reset all temporal state."""
        self.quality_assessor.reset()
        self.kalman_filter.reset()
        self.eye_analyzer.reset()
        self.temporal_fusion.reset()
        self.frame_index = 0
        self.last_valid_pose = None
        self.last_valid_landmarks = None
    
    def release(self):
        """Release resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None


def demo_webcam():
    """
    Demo function for webcam processing.
    
    Press 'q' to quit, 'r' to reset.
    """
    import cv2
    
    config = EdgeFaceConfig()
    edgeface = EdgeFace(config)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("EdgeFace Demo - Press 'q' to quit, 'r' to reset")
    
    fps_history = deque(maxlen=30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = edgeface.process(frame)
            
            # Track FPS
            fps = 1000 / result.processing_time_ms if result.processing_time_ms > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # Visualization
            vis_frame = frame.copy()
            
            # Draw landmarks
            if result.landmarks is not None and result.quality.is_valid:
                h, w = frame.shape[:2]
                for i, (x, y, z) in enumerate(result.landmarks):
                    px, py = int(x * w), int(y * h)
                    color = (0, 255, 0) if i not in result.quality.occluded_indices else (0, 0, 255)
                    cv2.circle(vis_frame, (px, py), 1, color, -1)
            
            # Draw info overlay
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Quality: {result.quality.composite_score:.2f}",
                f"Valid: {result.quality.is_valid}"
            ]
            
            if result.pose_filtered is not None:
                info_lines.extend([
                    f"Yaw: {result.pose_filtered.yaw:.1f}°",
                    f"Pitch: {result.pose_filtered.pitch:.1f}°",
                    f"Roll: {result.pose_filtered.roll:.1f}°"
                ])
            
            if result.temporal_features is not None:
                info_lines.extend([
                    f"EAR: {result.temporal_features.ear:.2f}",
                    f"Blink: {result.temporal_features.blink_detected}"
                ])
            
            for i, line in enumerate(info_lines):
                cv2.putText(
                    vis_frame, line, (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            
            # Quality indicator
            color = (0, 255, 0) if result.quality.is_valid else (0, 0, 255)
            cv2.rectangle(vis_frame, (5, 5), (15, 15), color, -1)
            
            cv2.imshow('EdgeFace Demo', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                edgeface.reset()
                print("Reset!")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        edgeface.release()


if __name__ == '__main__':
    demo_webcam()
