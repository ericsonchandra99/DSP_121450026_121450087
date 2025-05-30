# src/video_processor.py

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
import time
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage

# --- Parameter Global ---
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_DEFAULT = 30

DEFAULT_LOWCUT_RESP = 0.1
DEFAULT_HIGHCUT_RESP = 0.5
DEFAULT_ORDER_RESP = 2

DEFAULT_LOWCUT_PPG = 0.8
DEFAULT_HIGHCUT_PPG = 2.5
DEFAULT_ORDER_PPG = 3

BUFFER_SIZE_SEC = 10


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen:
        return data  # fallback ke data asli kalau belum cukup panjang
    try:
        y = filtfilt(b, a, data)
    except Exception as e:
        print(f"Filter error: {e}")
        return data
    return y


class VideoProcessor(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_data_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, float)
    update_stable_time_signal = pyqtSignal(float)
    update_amplitude_signal = pyqtSignal(float, float)
    update_face_in_roi_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.fps = FPS_DEFAULT
        self.buffer_length = int(BUFFER_SIZE_SEC * self.fps)
        self.time_buffer = deque(maxlen=self.buffer_length)
        self.respiration_signal_buffer = deque(maxlen=self.buffer_length)
        self.r_channel_buffer = deque(maxlen=self.buffer_length)
        self.g_channel_buffer = deque(maxlen=self.buffer_length)
        self.b_channel_buffer = deque(maxlen=self.buffer_length)

        self.lowcut_resp = DEFAULT_LOWCUT_RESP
        self.highcut_resp = DEFAULT_HIGHCUT_RESP
        self.order_resp = DEFAULT_ORDER_RESP

        self.lowcut_ppg = DEFAULT_LOWCUT_PPG
        self.highcut_ppg = DEFAULT_HIGHCUT_PPG
        self.order_ppg = DEFAULT_ORDER_PPG

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.prev_chest_y = None
        self.roi_size = 200

        self.face_in_roi_start_time = None
        self.stable_time = 0.0

    def set_filter_params(self, resp_low, resp_high, resp_order, ppg_low, ppg_high, ppg_order):
        self.lowcut_resp = resp_low
        self.highcut_resp = resp_high
        self.order_resp = resp_order
        self.lowcut_ppg = ppg_low
        self.highcut_ppg = ppg_high
        self.order_ppg = ppg_order

    def chrom_rppg(self, R, G, B):
        min_len = min(len(R), len(G), len(B))
        R_arr = np.array(R)[-min_len:]
        G_arr = np.array(G)[-min_len:]
        B_arr = np.array(B)[-min_len:]
        epsilon = 1e-6
        sum_rgb = R_arr + G_arr + B_arr + epsilon
        R_norm = R_arr / sum_rgb
        G_norm = G_arr / sum_rgb
        B_norm = B_arr / sum_rgb
        Y_chrom = 1.5 * R_norm + G_norm - 1.5 * B_norm
        return Y_chrom

    def run(self):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            print("Error: Tidak dapat membuka webcam.")
            self._run_flag = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = actual_fps
        else:
            self.fps = FPS_DEFAULT

        self.buffer_length = int(BUFFER_SIZE_SEC * self.fps)
        self.time_buffer = deque(maxlen=self.buffer_length)
        self.respiration_signal_buffer = deque(maxlen=self.buffer_length)
        self.r_channel_buffer = deque(maxlen=self.buffer_length)
        self.g_channel_buffer = deque(maxlen=self.buffer_length)
        self.b_channel_buffer = deque(maxlen=self.buffer_length)

        self.prev_chest_y = None
        start_time = time.time()

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time() - start_time
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
            half = self.roi_size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            pose_results = self.pose.process(frame_rgb)

            face_detected = False
            chest_movement = 0.0

            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    current_chest_y = (left_shoulder.y + right_shoulder.y) / 2 * FRAME_HEIGHT

                    if self.prev_chest_y is not None:
                        chest_movement = current_chest_y - self.prev_chest_y
                    self.prev_chest_y = current_chest_y
                    self.respiration_signal_buffer.append(chest_movement)
                    face_detected = True
                else:
                    self.prev_chest_y = None
                    self.respiration_signal_buffer.append(0.0)
                    face_detected = False
            else:
                self.prev_chest_y = None
                self.respiration_signal_buffer.append(0.0)
                face_detected = False

            self.update_face_in_roi_signal.emit(face_detected)

            if face_detected:
                if self.face_in_roi_start_time is None:
                    self.face_in_roi_start_time = time.time()
                self.stable_time = time.time() - self.face_in_roi_start_time
            else:
                self.face_in_roi_start_time = None
                self.stable_time = 0.0

            self.update_stable_time_signal.emit(self.stable_time)

            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size > 0:
                avg_color = np.mean(np.mean(roi, axis=0), axis=0)
                self.r_channel_buffer.append(avg_color[0])
                self.g_channel_buffer.append(avg_color[1])
                self.b_channel_buffer.append(avg_color[2])
            else:
                self.r_channel_buffer.append(0.0)
                self.g_channel_buffer.append(0.0)
                self.b_channel_buffer.append(0.0)

            self.time_buffer.append(current_time)

            current_resp_signal = np.array(self.respiration_signal_buffer)
            current_rppg_signal = self.chrom_rppg(self.r_channel_buffer, self.g_channel_buffer, self.b_channel_buffer)

            if len(current_resp_signal) >= 5:
                filtered_resp_signal = butter_bandpass_filter(
                    current_resp_signal,
                    self.lowcut_resp,
                    self.highcut_resp,
                    self.fps,
                    self.order_resp
                )
                amplitude_resp = np.sqrt(np.mean(filtered_resp_signal ** 2))
            else:
                filtered_resp_signal = current_resp_signal
                amplitude_resp = 0.0

            if len(current_rppg_signal) >= 5:
                filtered_rppg_signal = butter_bandpass_filter(
                    current_rppg_signal,
                    self.lowcut_ppg,
                    self.highcut_ppg,
                    self.fps,
                    self.order_ppg
                )
                amplitude_rppg = np.sqrt(np.mean(filtered_rppg_signal ** 2))
            else:
                filtered_rppg_signal = current_rppg_signal
                amplitude_rppg = 0.0

            self.update_amplitude_signal.emit(amplitude_resp, amplitude_rppg)

            self.update_data_signal.emit(
                np.array(self.time_buffer),
                current_resp_signal,
                current_rppg_signal,
                self.fps
            )

            overlay = frame.copy()
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            qt_img = self.convert_cv_qt(frame)
            self.change_pixmap_signal.emit(qt_img)

        self.cap.release()
        self.face_mesh.close()
        self.pose.close()

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio)
        return p

    def stop(self):
        self._run_flag = False
        self.wait()
