import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.model.to("mps")
        
    def detect_players(self, frame: np.ndarray):
        results = self.model.predict(frame)
        boxes, confidences = [], []
        for r in results:
            for *box, conf, cls in r.boxes.data.tolist():
                if int(cls) == 0:
                    x1, y1, x2, y2 = [int(x) for x in box]
                    boxes.append((x1, y1, x2, y2))
                    confidences.append(conf)
        return boxes, confidences


class EnhancedGameAnalysis:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(video_path)
        self.player_detector = PlayerDetector(model_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tracking and motion analysis
        self.player_tracks = {}
        self.player_speeds = {}
        self.next_player_id = 0
        self.max_track_length = 30
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.prev_positions = {}
        self.pixels_per_meter = 30
        
        # Metrics tracking
        self.frame_times = []
        self.detection_counts = []

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

        for center in current_centers:
            min_dist, best_id = float('inf'), None
            for player_id in self.player_tracks:
                if player_id in matched_tracks: 
                    continue
                last_pos = self.player_tracks[player_id][-1] if self.player_tracks[player_id] else None
                dist = np.linalg.norm(np.array(center) - np.array(last_pos)) if last_pos else float('inf')
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

        # Create overlay for smoother blending
        overlay = frame.copy()

        # Draw the motion trail
        for i in range(1, len(track)):
            alpha = i / len(track)  # Calculate gradient transparency
            thickness = max(2, int(4 * alpha))  # Adaptive thickness
            cv2.line(
                overlay, 
                tuple(track[i - 1]), 
                tuple(track[i]), 
                (int(255 * alpha), int(255 * (1 - alpha)), 0), 
                thickness
            )
        
        # Blend overlay into the frame for smooth visuals
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Display the speed only if non-zero
        if track and self.player_speeds[player_id] > 0:
            speed = self.player_speeds[player_id]
            cv2.putText(
                frame, 
                f"{speed:.1f} m/s", 
                (track[-1][0], track[-1][1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )

    def create_motion_frame(self, frame):
        fgmask = self.background_subtractor.apply(frame)
        motion_frame = frame.copy()
        motion_overlay = np.zeros_like(motion_frame)
        motion_overlay[fgmask > 0] = [0, 0, 128]
        cv2.addWeighted(motion_overlay, 0.3, motion_frame, 1, 0, motion_frame)
        for player_id, track in self.player_tracks.items():
            self.draw_motion_trail(motion_frame, player_id, track)
        return motion_frame

    def run(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_main = cv2.VideoWriter('football_output_main.mp4', fourcc, 30, (self.frame_width, self.frame_height))
        out_motion = cv2.VideoWriter('football_output_motion.mp4', fourcc, 30, (self.frame_width, self.frame_height))
        start_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret: 
                break

            frame_start_time = time.time()
            player_boxes, confidences = self.player_detector.detect_players(frame)
            self.update_player_tracks(player_boxes)

            main_frame = frame.copy()
            for (x1, y1, x2, y2), conf in zip(player_boxes, confidences):
                cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(main_frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(main_frame, f"Players: {len(player_boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            motion_frame = self.create_motion_frame(frame)

            out_main.write(main_frame)
            out_motion.write(motion_frame)
            cv2.imshow('Main Analysis', main_frame)
            cv2.imshow('Motion Analysis', motion_frame)

            frame_time = time.time() - frame_start_time
            self.frame_times.append(time.time() - start_time)
            self.detection_counts.append(len(player_boxes))
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
    model_path = 'yolov8n.pt'
    analysis_system = EnhancedGameAnalysis('football.mp4', model_path)
    analysis_system.run()
