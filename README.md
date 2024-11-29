

# Ballzy ⚽⚽️⚽️

Welcome to Ballzy! This system is designed to track players' movements and analyze their performance during football matches. Using the power of computer vision and deep learning, this tool allows you to detect players, track their movements, calculate their speeds, and even analyze goals scored during the game.

## Key Features 🚀

- **Player Detection**: Leverages YOLOv8 for real-time player detection.
- **Player Tracking**: Tracks players throughout the game and calculates their speed based on movement.
- **Motion Analysis**: Displays motion trails and overlays them on the video, providing a detailed view of player movements.
- **Goal Checking** (Enhanced Version): The enhanced system goes a step further by analyzing goals, adding another layer of game analysis.
- **Real-Time Visualization**: Displays FPS and detection counts to monitor the system's performance during runtime.
- **Multi-output**: Generates multiple video outputs — one showing the main analysis and another for motion tracking.

## How to Use 🏃‍♂️

1. **Clone the Repo**
   
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/ballzy.git
   cd football-player-tracking
   ```

2. **Install Dependencies**
   
   Make sure you have Python 3.8+ installed. Create a virtual environment and install the required dependencies:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # For Windows: myenv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Place Input Files**
   
   Add your video files in the `inputs/` directory (e.g., `football.mp4`).

4. **Run the Analysis**

   To run the analysis, simply execute the `football_final.py` script:
   ```bash
   python football_final.py
   ```

   The script will process the video, detect players, track them, and generate two output videos: one for main analysis and one for motion tracking. You can modify the input video path in the script to suit your needs.

5. **View the Results**
   
   Check the `outputs/` directory for the processed videos:
   - `football_output_main.mp4` — Main analysis video.
   - `football_output_motion.mp4` — Motion tracking video.

6. **Goal Checking** (Enhanced Version)

   For goal detection and analysis, use the enhanced version of the code. The logic includes additional processing steps to check for goals and analyze the game further.


## Requirements 📋

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib

Install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Contributing 🤝

Feel free to open issues or create pull requests if you want to contribute to the project. Your feedback and improvements are always welcome!
