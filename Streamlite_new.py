import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import pygame

########################Video Capture###################
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
count_right = 0
count_left = 0
incorrect_reps_right = 0
incorrect_reps_left = 0

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Custom VideoTransformer class
class PoseVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.level_right = None
        self.level_left = None

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Rest of the code...
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Making detection
        res = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ########################################
        # extract landmarks
        try:
            landmarks = res.pose_landmarks.landmark
            # Get coordinates for right arm
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Get coordinates for left arm
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angles for both arms
            angle_right = calculate_angle(shoulder_r, elbow_R, wrist_R)
            angle_left = calculate_angle(shoulder_l, elbow_L, wrist_L)

            # Visualize angles for both arms
            cv2.putText(image, f"Right Arm Angle: {angle_right:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Left Arm Angle: {angle_left:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic for right arm
            if angle_right > 160:
                self.level_right = "DOWN"
            if angle_right < 45 and self.level_right == 'DOWN':
                self.level_right = "UP"
                global count_right, incorrect_reps_right
                count_right += 1
                st.write(f"Right Arm Count: {count_right}")
                if count_right % 2 == 0:
                    incorrect_reps_right += 1
                    if incorrect_reps_right >= 3:
                        pygame.mixer.init()
                        pygame.mixer.music.load("alert_sound.wav")
                        pygame.mixer.music.play()

                    st.write(f"Right Arm Incorrect Reps: {incorrect_reps_right}")

            # Curl counter logic for left arm
            if angle_left > 160:
                self.level_left = "DOWN"
            if angle_left < 45 and self.level_left == 'DOWN':
                self.level_left = "UP"
                global count_left, incorrect_reps_left
                count_left += 1
                st.write(f"Left Arm Count: {count_left}")
                if count_left % 2 == 0:
                    incorrect_reps_left += 1
                    if incorrect_reps_left >= 3:
                        pygame.mixer.init()
                        pygame.mixer.music.load("alert_sound.wav")
                        pygame.mixer.music.play()

                    st.write(f"Left Arm Incorrect Reps: {incorrect_reps_left}")

            # Draw landmarks
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except:
            pass

        cv2.putText(image, 'RIGHT ARM COUNT', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(count_right),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'LEFT ARM COUNT', (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(count_left),
                    (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'RIGHT ARM LEVEL', (10, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.level_right,
                    (50, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'LEFT ARM LEVEL', (10, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.level_left,
                    (50, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def main():
    # Streamlit application code...
    st.title("Fitness Tracker")
    st.write("Real-time Video Feed")

    webrtc_streamer(
        key="pose",
        video_transformer_factory=PoseVideoTransformer,
        async_transform=True,
    )

if __name__ == '__main__':
    main()
