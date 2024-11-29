import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from ultralytics import YOLO

class GoalDetector:
    def __init__(self, frame_width, frame_height):
        # Improved goal regions with more realistic proportions
        self.left_goal_region = {
            'x1': 0,
            'y1': int(frame_height * 0.35),  # Adjusted for better goal area coverage
            'x2': int(frame_width * 0.08),   # Slightly narrower for precision
            'y2': int(frame_height * 0.65)   # Adjusted for better goal area coverage
        }
        self.right_goal_region = {
            'x1': int(frame_width * 0.92),   # Slightly narrower for precision
            'y1': int(frame_height * 0.35),  # Adjusted for better goal area coverage
            'x2': frame_width,
            'y2': int(frame_height * 0.65)   # Adjusted for better goal area coverage
        }
        
        # Enhanced tracking parameters
        self.ball_history = deque(maxlen=15)  # Increased history for better trajectory analysis
        self.goal_cooldown = 45  # Increased cooldown to prevent false positives
        self.cooldown_counter = 0
        self.goal_threshold_speed = 12  # Reduced speed threshold for better sensitivity
        self.consecutive_frames_threshold = 2  # Reduced for faster detection
        self.frames_in_goal_region = 0
        
        # New parameters for improved detection
        self.trajectory_points = deque(maxlen=10)  # Store recent trajectory
        self.confidence_threshold = 0.6  # Minimum confidence for goal detection
        self.last_goal_time = time.time()
        self.min_goal_interval = 3.0  # Minimum seconds between goals
        
    def calculate_ball_trajectory(self, ball_pos):
        """Calculate ball trajectory and predict if it's heading towards goal"""
        if len(self.trajectory_points) < 3:
            self.trajectory_points.append(ball_pos)
            return None, 0, 0
            
        # Calculate trajectory using last few points
        points = list(self.trajectory_points)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Linear regression to predict trajectory
        try:
            coeffs = np.polyfit(x_coords, y_coords, 1)
            slope = coeffs[0]
            
            # Calculate trajectory confidence based on linearity
            y_pred = np.polyval(coeffs, x_coords)
            residuals = np.array(y_coords) - y_pred
            trajectory_confidence = 1 - (np.std(residuals) / np.std(y_coords))
            
            # Calculate direction and speed
            dx = ball_pos[0] - points[-1][0]
            dy = ball_pos[1] - points[-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            direction = 'right' if dx > 0 else 'left'
            
            self.trajectory_points.append(ball_pos)
            return direction, speed, trajectory_confidence
            
        except np.linalg.LinAlgError:
            return None, 0, 0
            
    def is_in_goal_region(self, ball_pos, side='both'):
        """Enhanced goal region detection with proximity checking"""
        if ball_pos is None:
            return None
            
        x, y = ball_pos
        
        # Add proximity zones around goal regions
        proximity_margin = 20  # pixels
        
        if side == 'left' or side == 'both':
            # Check actual goal region
            if (self.left_goal_region['x1'] <= x <= self.left_goal_region['x2'] and
                self.left_goal_region['y1'] <= y <= self.left_goal_region['y2']):
                return 'left'
            # Check proximity zone
            elif (self.left_goal_region['x1'] <= x <= self.left_goal_region['x2'] + proximity_margin and
                  self.left_goal_region['y1'] - proximity_margin <= y <= self.left_goal_region['y2'] + proximity_margin):
                return 'left_proximity'
                
        if side == 'right' or side == 'both':
            # Check actual goal region
            if (self.right_goal_region['x1'] <= x <= self.right_goal_region['x2'] and
                self.right_goal_region['y1'] <= y <= self.right_goal_region['y2']):
                return 'right'
            # Check proximity zone
            elif (self.right_goal_region['x1'] - proximity_margin <= x <= self.right_goal_region['x2'] and
                  self.right_goal_region['y1'] - proximity_margin <= y <= self.right_goal_region['y2'] + proximity_margin):
                return 'right_proximity'
                
        return None

    def detect_goal(self, ball_pos):
        """Enhanced goal detection with trajectory prediction"""
        current_time = time.time()
        
        # Check cooldown and minimum goal interval
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None
            
        if current_time - self.last_goal_time < self.min_goal_interval:
            return None
            
        if ball_pos is None:
            self.frames_in_goal_region = 0
            return None
            
        # Calculate trajectory and movement
        direction, speed, trajectory_confidence = self.calculate_ball_trajectory(ball_pos)
        goal_region = self.is_in_goal_region(ball_pos)
        
        # Initialize goal confidence
        goal_confidence = 0.0
        
        if goal_region in ['left', 'right']:
            self.frames_in_goal_region += 1
            
            # Calculate goal confidence based on multiple factors
            position_confidence = min(1.0, self.frames_in_goal_region / self.consecutive_frames_threshold)
            speed_confidence = min(1.0, speed / self.goal_threshold_speed)
            
            # Combine confidences
            goal_confidence = (position_confidence * 0.4 + 
                             speed_confidence * 0.3 + 
                             trajectory_confidence * 0.3)
            
            if (goal_confidence >= self.confidence_threshold and 
                self.frames_in_goal_region >= self.consecutive_frames_threshold and
                speed >= self.goal_threshold_speed):
                
                # Verify direction matches goal side
                if ((goal_region == 'left' and direction == 'left') or
                    (goal_region == 'right' and direction == 'right')):
                    self.cooldown_counter = self.goal_cooldown
                    self.frames_in_goal_region = 0
                    self.last_goal_time = current_time
                    return goal_region
                    
        elif goal_region in ['left_proximity', 'right_proximity']:
            # Keep tracking but don't reset frames counter
            pass
        else:
            self.frames_in_goal_region = 0
            
        return None

    def draw_goal_regions(self, frame):
        """Enhanced visualization with proximity zones"""
        # Draw main goal regions
        cv2.rectangle(frame,
                     (self.left_goal_region['x1'], self.left_goal_region['y1']),
                     (self.left_goal_region['x2'], self.left_goal_region['y2']),
                     (0, 255, 0), 2)
        
        cv2.rectangle(frame,
                     (self.right_goal_region['x1'], self.right_goal_region['y1']),
                     (self.right_goal_region['x2'], self.right_goal_region['y2']),
                     (0, 255, 0), 2)
        
        # Draw proximity zones (optional - for debugging)
        proximity_margin = 20
        cv2.rectangle(frame,
                     (self.left_goal_region['x1'], self.left_goal_region['y1'] - proximity_margin),
                     (self.left_goal_region['x2'] + proximity_margin, self.left_goal_region['y2'] + proximity_margin),
                     (0, 255, 0), 1)
        
        cv2.rectangle(frame,
                     (self.right_goal_region['x1'] - proximity_margin, self.right_goal_region['y1'] - proximity_margin),
                     (self.right_goal_region['x2'], self.right_goal_region['y2'] + proximity_margin),
                     (0, 255, 0), 1)

class PlayerDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.model.to("mps")
        
    def detect_players(self, frame: np.ndarray):
        results = self.model.predict(frame)
        boxes, confidences = [], []
        for r in results:
            for *box, conf, cls in r.boxes.data.tolist():
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = [int(x) for x in box]
                    boxes.append((x1, y1, x2, y2))
                    confidences.append(conf)
        return boxes, confidences

class BallDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.model.to("mps")
        self.ball_history = deque(maxlen=5)
        
    def detect_ball(self, frame: np.ndarray):
        results = self.model.predict(frame)
        best_ball = None
        max_conf = 0
        
        for r in results:
            for *box, conf, cls in r.boxes.data.tolist():
                if int(cls) == 32:  # ball class in COCO dataset
                    if conf > max_conf:
                        x1, y1, x2, y2 = [int(x) for x in box]
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        best_ball = (center, conf)
                        max_conf = conf
        
        if best_ball:
            self.ball_history.append(best_ball[0])
            return best_ball[0]
        elif self.ball_history:
            return self.ball_history[-1]
        return None

class EnhancedGameAnalysis:
    def __init__(self, video_path, model_paths):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize detectors
        self.player_detector = PlayerDetector(model_paths['player'])
        self.ball_detector = BallDetector(model_paths['ball'])
        self.goal_detector = GoalDetector(self.frame_width, self.frame_height)
        
        # Tracking and motion analysis
        self.player_tracks = {}
        self.player_speeds = {}
        self.next_player_id = 0
        self.max_track_length = 30
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.prev_positions = {}
        self.pixels_per_meter = 30
        
        # Score tracking
        self.left_score = 0
        self.right_score = 0
        
        # Metrics tracking
        self.frame_times = []
        self.detection_counts = []

    def update_player_tracks(self, current_boxes):
        """
        Update player tracking information based on current detections
        Args:
            current_boxes: List of (x1, y1, x2, y2) bounding boxes for detected players
        """
        # If no previous tracks, initialize with current detections
        if not self.prev_positions:
            for i, box in enumerate(current_boxes):
                player_id = self.next_player_id
                self.next_player_id += 1
                center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                self.player_tracks[player_id] = deque(maxlen=self.max_track_length)
                self.player_tracks[player_id].append(center)
                self.prev_positions[player_id] = center
                self.player_speeds[player_id] = 0
            return

        # Calculate centers for current detections
        current_centers = [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in current_boxes]
        
        # Match current detections with existing tracks using simple distance matching
        matched_tracks = {}
        matched_detections = set()
        
        for player_id, prev_pos in self.prev_positions.items():
            min_dist = float('inf')
            best_match = None
            
            for i, center in enumerate(current_centers):
                if i in matched_detections:
                    continue
                    
                dist = np.sqrt((prev_pos[0] - center[0])**2 + (prev_pos[1] - center[1])**2)
                if dist < min_dist and dist < 100:  # Maximum distance threshold
                    min_dist = dist
                    best_match = (i, center)
            
            if best_match is not None:
                idx, center = best_match
                matched_detections.add(idx)
                matched_tracks[player_id] = center
                
                # Update track and calculate speed
                self.player_tracks[player_id].append(center)
                speed = min_dist / (1/30)  # Assuming 30 fps
                self.player_speeds[player_id] = speed / self.pixels_per_meter  # Convert to m/s
        
        # Add new tracks for unmatched detections
        for i, center in enumerate(current_centers):
            if i not in matched_detections:
                player_id = self.next_player_id
                self.next_player_id += 1
                self.player_tracks[player_id] = deque(maxlen=self.max_track_length)
                self.player_tracks[player_id].append(center)
                matched_tracks[player_id] = center
                self.player_speeds[player_id] = 0
        
        # Remove tracks that weren't matched
        self.player_tracks = {k: v for k, v in self.player_tracks.items() if k in matched_tracks}
        self.player_speeds = {k: v for k, v in self.player_speeds.items() if k in matched_tracks}
        
        # Update previous positions
        self.prev_positions = matched_tracks

    def create_motion_frame(self, frame):
        """
        Create a visualization of motion analysis
        """
        motion_frame = frame.copy()
        
        # Draw player tracks
        for player_id, track in self.player_tracks.items():
            points = list(track)
            if len(points) > 1:
                # Draw track line
                for i in range(len(points) - 1):
                    cv2.line(motion_frame, points[i], points[i + 1], (0, 255, 0), 2)
                
                # Draw current position and speed
                current_pos = points[-1]
                speed = self.player_speeds.get(player_id, 0)
                cv2.putText(motion_frame, f"ID:{player_id} {speed:.1f}m/s", 
                           (current_pos[0], current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return motion_frame

    def update_score(self, goal_side):
        if goal_side == 'left':
            self.right_score += 1
        elif goal_side == 'right':
            self.left_score += 1

    def draw_score(self, frame):
        score_text = f"Score: {self.left_score} - {self.right_score}"
        cv2.putText(frame, score_text, (self.frame_width//2 - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def run(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_main = cv2.VideoWriter('enhanced_output_main.mp4', fourcc, 30, (self.frame_width, self.frame_height))
        out_motion = cv2.VideoWriter('enhanced_output_motion.mp4', fourcc, 30, (self.frame_width, self.frame_height))
        start_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret: 
                break

            frame_start_time = time.time()
            
            # Detect players and ball
            player_boxes, confidences = self.player_detector.detect_players(frame)
            ball_pos = self.ball_detector.detect_ball(frame)
            
            # Update tracking
            self.update_player_tracks(player_boxes)
            
            # Check for goals
            goal_scored = self.goal_detector.detect_goal(ball_pos)
            if goal_scored:
                self.update_score(goal_scored)

            # Create visualization
            main_frame = frame.copy()
            
            # Draw goal regions
            self.goal_detector.draw_goal_regions(main_frame)
            
            # Draw players
            for (x1, y1, x2, y2), conf in zip(player_boxes, confidences):
                cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(main_frame, f"{conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw ball
            if ball_pos:
                cv2.circle(main_frame, ball_pos, 5, (0, 0, 255), -1)
            
            # Draw score
            self.draw_score(main_frame)
            
            # Draw player count
            cv2.putText(main_frame, f"Players: {len(player_boxes)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Create motion analysis
            motion_frame = self.create_motion_frame(frame)

            # Write and display frames
            out_main.write(main_frame)
            out_motion.write(motion_frame)
            cv2.imshow('Main Analysis', main_frame)
            cv2.imshow('Motion Analysis', motion_frame)

            # Update metrics
            frame_time = time.time() - frame_start_time
            self.frame_times.append(time.time() - start_time)
            self.detection_counts.append(len(player_boxes))
            
            # Update plots
            ax1.clear()
            ax2.clear()
            ax1.plot(self.frame_times, [1 / ft for ft in np.diff(self.frame_times, prepend=self.frame_times[0])], label="FPS")
            ax2.plot(self.frame_times, self.detection_counts, label="Detections")
            ax1.set_ylabel("FPS")
            ax2.set_ylabel("Detections")
            ax2.set_xlabel("Time (s)")
            plt.pause(0.001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        out_main.release()
        out_motion.release()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == "__main__":
    model_paths = {
        'player': 'yolov8n.pt',
        'ball': 'yolov8n.pt'  
    }
    analysis_system = EnhancedGameAnalysis('football2.mp4', model_paths)
    analysis_system.run()