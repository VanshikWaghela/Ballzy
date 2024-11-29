import cv2
import numpy as np
from collections import deque
from typing import Tuple, List
import torch
from ultralytics import YOLO
import time

class PlayerDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.model.to("mps")

    def detect_players(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        results = self.model.predict(frame, stream=True)
        boxes, confidences = [], []
        for r in results:
            for *box, conf, cls in r.boxes.data.tolist():
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = [int(x) for x in box]
                    boxes.append((x1, y1, x2, y2))
                    confidences.append(conf)
        return boxes, confidences

class GameAnalysisSystem:
    def __init__(self, video_path, model_paths):
        self.cap = cv2.VideoCapture(video_path)
        self.player_detector = PlayerDetector(model_paths['player_detector'])
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tracking parameters
        self.player_tracks = {}
        self.player_speeds = {}
        self.next_player_id = 0
        self.max_track_length = 30
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        self.prev_positions = {}
        self.pixels_per_meter = 30

    def calculate_speed(self, player_id, current_pos):
        if player_id in self.prev_positions:
            prev_pos, prev_time = self.prev_positions[player_id]
            distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos)) / self.pixels_per_meter
            time_diff = time.time() - prev_time
            speed = distance / time_diff if time_diff > 0 else 0
            self.prev_positions[player_id] = (current_pos, time.time())
            return speed
        self.prev_positions[player_id] = (current_pos, time.time())
        return 0

    def update_player_tracks(self, current_boxes):
        current_centers = [
            ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            for box in current_boxes
        ]

        matched_tracks = {}
        active_ids = set()

        # Match existing tracks to new detections
        for center in current_centers:
            min_dist, best_id = float('inf'), None
            for player_id in self.player_tracks:
                if player_id in matched_tracks:
                    continue
                last_pos = self.player_tracks[player_id][-1] if self.player_tracks[player_id] else None
                if last_pos:
                    dist = np.linalg.norm(np.array(center) - np.array(last_pos))
                    if dist < 50 and dist < min_dist:
                        min_dist, best_id = dist, player_id

            if best_id is not None:
                matched_tracks[best_id] = center
                active_ids.add(best_id)
            else:
                # Create new track for unmatched detection
                self.player_tracks[self.next_player_id] = deque(maxlen=self.max_track_length)
                self.player_speeds[self.next_player_id] = 0.0
                matched_tracks[self.next_player_id] = center
                active_ids.add(self.next_player_id)
                self.next_player_id += 1

        # Update tracks
        for player_id, center in matched_tracks.items():
            self.player_tracks[player_id].append(center)
            self.player_speeds[player_id] = self.calculate_speed(player_id, center)

        # Remove stale tracks
        stale_ids = set(self.player_tracks.keys()) - active_ids
        for stale_id in stale_ids:
            del self.player_tracks[stale_id]
            del self.player_speeds[stale_id]

    def draw_motion_trail(self, frame, player_id, track, color=(0, 255, 255)):
        if len(track) < 2:
            return

        # Draw only the most recent segment of the trail
        for i in range(1, len(track)):
            alpha = i / len(track)  # Fade older trail segments
            thickness = max(1, int(3 * alpha))  # Thinner lines for older segments
            cv2.line(
                frame,
                tuple(map(int, track[i - 1])),
                tuple(map(int, track[i])),
                (int(255 * alpha), int(255 * (1 - alpha)), 0),
                thickness
            )

        # Display current speed
        if track and self.player_speeds[player_id] > 0:
            speed = self.player_speeds[player_id]
            cv2.putText(
                frame,
                f"{speed:.1f} m/s",
                (int(track[-1][0]), int(track[-1][1] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    def create_motion_frame(self, frame):
        # Apply background subtraction
        fgmask = self.background_subtractor.apply(frame)
        
        # Create a fresh overlay for each frame
        motion_frame = frame.copy()
        motion_overlay = np.zeros_like(motion_frame)
        
        # Add subtle motion detection overlay
        motion_overlay[fgmask > 0] = [0, 0, 64]  # Reduced blue intensity
        cv2.addWeighted(motion_overlay, 0.3, motion_frame, 1, 0, motion_frame)
        
        # Draw current motion trails
        for player_id, track in self.player_tracks.items():
            self.draw_motion_trail(motion_frame, player_id, track)
            
        return motion_frame

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_main = cv2.VideoWriter('output_main.mp4', fourcc, 30, (self.frame_width, self.frame_height))
        out_motion = cv2.VideoWriter('output_motion.mp4', fourcc, 30, (self.frame_width, self.frame_height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect players
            player_boxes, confidences = self.player_detector.detect_players(frame)
            self.update_player_tracks(player_boxes)

            # Create main visualization
            main_frame = frame.copy()
            for (x1, y1, x2, y2), conf in zip(player_boxes, confidences):
                cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(main_frame, f"{conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add player count
            cv2.putText(main_frame, f"Players: {len(player_boxes)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Create motion analysis frame
            motion_frame = self.create_motion_frame(frame)

            # Write and display frames
            out_main.write(main_frame)
            out_motion.write(motion_frame)
            cv2.imshow('Main Analysis', main_frame)
            cv2.imshow('Motion Analysis', motion_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        out_main.release()
        out_motion.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_paths = {
        'player_detector': 'yolov8n.pt'
    }
    analysis_system = GameAnalysisSystem('football.mp4', model_paths)
    analysis_system.run()