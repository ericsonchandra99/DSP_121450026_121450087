import sys
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFrame, QLineEdit, QFormLayout, QMessageBox, QScrollArea,
    QSplitter, QAction, QTextEdit, QDialog, QDialogButtonBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Parameter Global ---
WEBCAM_INDEX = 0  # indeks webcam default
FRAME_WIDTH = 640  # lebar frame video
FRAME_HEIGHT = 480  # tinggi frame video
FPS_DEFAULT = 30  # frame per detik default

# Parameter filter default untuk sinyal pernapasan dan rPPG (detak jantung)
DEFAULT_LOWCUT_RESP = 0.1
DEFAULT_HIGHCUT_RESP = 0.5
DEFAULT_ORDER_RESP = 2

DEFAULT_LOWCUT_PPG = 0.8
DEFAULT_HIGHCUT_PPG = 2.5
DEFAULT_ORDER_PPG = 3

BUFFER_SIZE_SEC = 10  # ukuran buffer data, contoh 10 detik

# --- Fungsi filter bandpass menggunakan scipy ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # frekuensi nyquist (setengah dari fs)
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')  # hitung koefisien filter
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Fungsi untuk memfilter data sinyal dengan bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    padlen = 3 * (max(len(a), len(b)) - 1)  # panjang padding untuk filtfilt
    if len(data) <= padlen:
        return data  # jika data terlalu pendek, kembalikan data asli
    try:
        y = filtfilt(b, a, data)  # filter data tanpa delay phase
    except Exception as e:
        print(f"Filter error: {e}")
        return data  # jika error saat filtering, kembalikan data asli
    return y

# --- Kelas thread untuk memproses video secara paralel ---
class VideoProcessor(QThread):
    # Signal untuk komunikasi dengan GUI (mengirim data dan gambar)
    change_pixmap_signal = pyqtSignal(QImage)  # untuk update gambar video di GUI
    update_data_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, float)  # waktu, sinyal napas, rPPG, fps
    update_stable_time_signal = pyqtSignal(float)  # waktu wajah stabil di ROI
    update_amplitude_signal = pyqtSignal(float, float)  # amplitudo sinyal napas dan rPPG
    update_face_in_roi_signal = pyqtSignal(bool)  # deteksi wajah di ROI (kotak hijau)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Inisialisasi variabel dan buffer sinyal
        self._run_flag = True  # flag kontrol thread
        self.fps = FPS_DEFAULT
        self.buffer_length = int(BUFFER_SIZE_SEC * self.fps)
        self.time_buffer = deque(maxlen=self.buffer_length)
        self.respiration_signal_buffer = deque(maxlen=self.buffer_length)
        self.r_channel_buffer = deque(maxlen=self.buffer_length)
        self.g_channel_buffer = deque(maxlen=self.buffer_length)
        self.b_channel_buffer = deque(maxlen=self.buffer_length)

        # Parameter filter default
        self.lowcut_resp = DEFAULT_LOWCUT_RESP
        self.highcut_resp = DEFAULT_HIGHCUT_RESP
        self.order_resp = DEFAULT_ORDER_RESP

        self.lowcut_ppg = DEFAULT_LOWCUT_PPG
        self.highcut_ppg = DEFAULT_HIGHCUT_PPG
        self.order_ppg = DEFAULT_ORDER_PPG

        # Inisialisasi MediaPipe FaceMesh dan Pose untuk deteksi wajah dan gerak tubuh
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

        self.prev_chest_y = None  # posisi dada sebelumnya (untuk deteksi pernapasan)
        self.roi_size = 200  # ukuran ROI untuk deteksi warna rPPG (kotak hijau)

        self.face_in_roi_start_time = None  # waktu mulai wajah stabil
        self.stable_time = 0.0  # durasi wajah stabil

    # Method untuk update parameter filter dari GUI
    def set_filter_params(self, resp_low, resp_high, resp_order, ppg_low, ppg_high, ppg_order):
        self.lowcut_resp = resp_low
        self.highcut_resp = resp_high
        self.order_resp = resp_order
        self.lowcut_ppg = ppg_low
        self.highcut_ppg = ppg_high
        self.order_ppg = ppg_order

    # Algoritma CHROM untuk ekstraksi sinyal rPPG dari warna R,G,B
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
        Y_chrom = 1.5 * R_norm + G_norm - 1.5 * B_norm  # kombinasi linier untuk sinyal rPPG
        return Y_chrom

    # Fungsi utama thread untuk pemrosesan video secara terus menerus
    def run(self):
        # Buka webcam
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            print("Error: Tidak dapat membuka webcam.")
            self._run_flag = False
            return

        # Set ukuran frame webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = actual_fps
        else:
            self.fps = FPS_DEFAULT

        # Reinisialisasi buffer sesuai fps aktual
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
                continue  # skip jika gagal baca frame

            current_time = time.time() - start_time
            frame = cv2.flip(frame, 1)  # mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tentukan koordinat ROI (kotak hijau) di tengah frame
            cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
            half = self.roi_size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            # Proses deteksi pose (untuk napas)
            pose_results = self.pose.process(frame_rgb)

            face_detected = False
            chest_movement = 0.0

            if pose_results.pose_landmarks:
                # Gambar landmark pada frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # Cek visibility landmark
                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    current_chest_y = (left_shoulder.y + right_shoulder.y) / 2 * FRAME_HEIGHT

                    # Hitung perubahan posisi dada (napas)
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

            # Kirim info apakah wajah terdeteksi di ROI
            self.update_face_in_roi_signal.emit(face_detected)

            # Hitung waktu stabil wajah di ROI
            if face_detected:
                if self.face_in_roi_start_time is None:
                    self.face_in_roi_start_time = time.time()
                self.stable_time = time.time() - self.face_in_roi_start_time
            else:
                self.face_in_roi_start_time = None
                self.stable_time = 0.0

            self.update_stable_time_signal.emit(self.stable_time)

            # Ambil ROI warna untuk sinyal rPPG
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size > 0:
                avg_color = np.mean(np.mean(roi, axis=0), axis=0)  # rata-rata warna RGB
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

            # Filter sinyal napas dan hitung amplitudo (energi sinyal)
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

            # Filter sinyal rPPG dan hitung amplitudo
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

            # Kirim data amplitudo ke GUI
            self.update_amplitude_signal.emit(amplitude_resp, amplitude_rppg)

            # Kirim data sinyal & waktu ke GUI untuk plot
            self.update_data_signal.emit(
                np.array(self.time_buffer),
                current_resp_signal,
                current_rppg_signal,
                self.fps
            )

            # Gambar overlay kotak hijau di frame
            overlay = frame.copy()
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Konversi frame ke QImage untuk update di GUI
            qt_img = self.convert_cv_qt(frame)
            self.change_pixmap_signal.emit(qt_img)

        # Release sumber daya saat thread dihentikan
        self.cap.release()
        self.face_mesh.close()
        self.pose.close()

    # Konversi frame OpenCV ke QImage agar bisa ditampilkan di PyQt QLabel
    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio)
        return p

    def stop(self):
        self._run_flag = False
        self.wait()  # tunggu thread selesai sebelum lanjut


# --- Dialog Bantuan untuk penjelasan parameter filter ---
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Penjelasan Parameter Filter")
        self.resize(600, 400)
        layout = QVBoxLayout(self)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Segoe UI", 11))
        text.setHtml("""
        <h2>Penjelasan Parameter Filter</h2>
        <p><b>Respirasi Low Cut (Hz):</b> Batas frekuensi paling rendah untuk menangkap sinyal napas. Misalnya 0,1 Hz berarti minimal 6 napas per menit (karena 0,1 Hz = 6/menit).</p>
        <p><b>Respirasi High Cut (Hz):</b> Batas frekuensi paling tinggi yang diambil untuk napas. Misalnya 0,5 Hz berarti maksimal 30 napas per menit.</p>
        <p><b>Respirasi Orde Filter:</b> Semakin tinggi angka ini, filter semakin 'tajam' memisahkan sinyal dari noise, tapi membutuhkan proses lebih lama.</p>
        <p><b>PPG Low Cut (Hz):</b> Batas paling rendah untuk detak jantung. Misal 0,8 Hz sama dengan 48 detak/menit.</p>
        <p><b>PPG High Cut (Hz):</b> Batas paling tinggi untuk detak jantung. Misal 2,5 Hz sama dengan 150 detak/menit.</p>
        <p><b>PPG Orde Filter:</b> Sama seperti orde filter napas, tapi untuk sinyal detak jantung.</p>
        <hr>
        <p><b>Kenapa penting?</b> Parameter ini membantu alat membedakan sinyal asli dari gangguan agar hasil akurat.</p>
        <hr>
        <p><b>Jika wajah Anda tetap berada di kotak hijau selama 30 detik:</b></p>
        <ul>
            <li>Pengukuran detak jantung dan napas kemungkinan besar akurat dan stabil.</li>
            <li>30 detik adalah durasi minimal agar data cukup untuk dianalisis dengan baik (berdasarkan penelitian rPPG).</li>
            <li>Pastikan pencahayaan dan posisi wajah tetap stabil agar hasil optimal.</li>
            <li>Referensi: <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5807456/" target="_blank">Studi rPPG NCBI</a></li>
        </ul>
        <hr>
        <p><i>Selamat menggunakan aplikasi ini! Jaga posisi wajah tetap di dalam kotak untuk hasil terbaik.</i></p>
        """)
        layout.addWidget(text)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)


# --- Main Window untuk GUI utama ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor Real-time Pernapasan & rPPG")
        self.showMaximized()  # tampilkan window fullscreen

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout utama horizontal
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(15)

        splitter_horizontal = QSplitter(Qt.Horizontal)

        # Bagian kiri: video, kontrol, dan parameter filter
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setSpacing(12)

        # Label untuk menampilkan video webcam
        self.video_label = QLabel("Memuat video...")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            background-color: black;
            border: 3px solid #28a745;
            border-radius: 6px;
        """)
        self.left_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Layout tombol kontrol
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        # Tombol mulai
        self.start_button = QPushButton("Mulai")
        self.start_button.setMinimumHeight(42)
        self.start_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #007bff; color: white; border-radius: 6px;")
        self.start_button.clicked.connect(self.start_processing)

        # Tombol berhenti
        self.stop_button = QPushButton("Berhenti")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #dc3545; color: white; border-radius: 6px;")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)

        # Tombol terapkan filter
        self.apply_filter_button = QPushButton("Terapkan Filter")
        self.apply_filter_button.setMinimumHeight(42)
        self.apply_filter_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #28a745; color: white; border-radius: 6px;")
        self.apply_filter_button.clicked.connect(self.apply_filter_changes)

        # Tombol reset pengukuran
        self.reset_button = QPushButton("Ulangi Pengukuran")
        self.reset_button.setMinimumHeight(42)
        self.reset_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #ffc107; color: black; border-radius: 6px;")
        self.reset_button.clicked.connect(self.reset_measurement)
        self.reset_button.setEnabled(False)

        # Tambahkan tombol ke layout kontrol
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.apply_filter_button)
        control_layout.addWidget(self.reset_button)

        self.left_layout.addLayout(control_layout)

        # Label metrik (fps, bpm napas, bpm jantung)
        self.metrics_label = QLabel("FPS: 0 | Pernapasan (BPM): 0,00 | Detak Jantung (BPM): 0,00")
        self.metrics_label.setStyleSheet("font-size: 17pt; font-weight: bold; color: #333; margin-top: 12px;")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.metrics_label)

        # Label instruksi posisi wajah
        self.face_pos_label = QLabel("")
        self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: #b03a2e; margin-top: 8px;")
        self.face_pos_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.face_pos_label)

        # Label amplitudo sinyal napas dan jantung
        self.amplitude_label = QLabel("Amplitudo Pernapasan: 0.00 | Amplitudo Detak Jantung: 0.00")
        self.amplitude_label.setStyleSheet("font-size: 11pt; color: #444; margin-top: 8px;")
        self.amplitude_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.amplitude_label)

        # Parameter filter dengan scroll agar tidak memakan tempat
        self.filter_frame = QFrame()
        self.filter_frame.setFrameShape(QFrame.StyledPanel)
        self.filter_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #6c757d;
                border-radius: 8px;
                background-color: #f8f9fa;
                padding: 12px;
            }
        """)
        self.filter_layout = QFormLayout(self.filter_frame)
        self.filter_layout.setLabelAlignment(Qt.AlignRight)
        self.filter_layout.setFormAlignment(Qt.AlignCenter)
        self.filter_layout.setHorizontalSpacing(30)
        self.filter_layout.setVerticalSpacing(18)

        label_title = QLabel("<b>Parameter Filter</b>")
        label_title.setStyleSheet("font-size: 14pt; color: #495057;")
        self.filter_layout.addRow(label_title)
        self.filter_layout.addRow(QLabel(""))

        def setup_lineedit(le, placeholder):
            le.setFixedWidth(100)
            le.setAlignment(Qt.AlignCenter)
            le.setPlaceholderText(placeholder)
            le.setStyleSheet("font-size: 12pt; padding: 5px;")

        self.resp_low_input = QLineEdit(str(DEFAULT_LOWCUT_RESP))
        setup_lineedit(self.resp_low_input, "Hz (contoh: 0.1)")
        self.resp_high_input = QLineEdit(str(DEFAULT_HIGHCUT_RESP))
        setup_lineedit(self.resp_high_input, "Hz (contoh: 0.5)")
        self.resp_order_input = QLineEdit(str(DEFAULT_ORDER_RESP))
        setup_lineedit(self.resp_order_input, "Orde (contoh: 2)")

        self.ppg_low_input = QLineEdit(str(DEFAULT_LOWCUT_PPG))
        setup_lineedit(self.ppg_low_input, "Hz (contoh: 0.8)")
        self.ppg_high_input = QLineEdit(str(DEFAULT_HIGHCUT_PPG))
        setup_lineedit(self.ppg_high_input, "Hz (contoh: 2.5)")
        self.ppg_order_input = QLineEdit(str(DEFAULT_ORDER_PPG))
        setup_lineedit(self.ppg_order_input, "Orde (contoh: 3)")

        self.filter_layout.addRow("Respirasi Low Cut (Hz):", self.resp_low_input)
        self.filter_layout.addRow("Respirasi High Cut (Hz):", self.resp_high_input)
        self.filter_layout.addRow("Respirasi Orde Filter:", self.resp_order_input)
        self.filter_layout.addRow("PPG Low Cut (Hz):", self.ppg_low_input)
        self.filter_layout.addRow("PPG High Cut (Hz):", self.ppg_high_input)
        self.filter_layout.addRow("PPG Orde Filter:", self.ppg_order_input)

        self.filter_scroll = QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setWidget(self.filter_frame)
        self.filter_scroll.setMaximumHeight(280)
        self.left_layout.addWidget(self.filter_scroll)

        # Label info stabilitas wajah di kotak hijau
        self.stable_info_label = QLabel("Wajah belum stabil di kotak.")
        self.stable_info_label.setStyleSheet("font-size: 11pt; color: #555; font-style: italic; margin-top: 8px;")
        self.left_layout.addWidget(self.stable_info_label)

        # Label status kualitas sinyal setelah 30 detik
        self.signal_quality_label = QLabel("")
        self.signal_quality_label.setWordWrap(True)
        self.signal_quality_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #0056b3; margin-top: 12px;")
        self.left_layout.addWidget(self.signal_quality_label)

        # Label info ilmiah singkat
        self.science_info_label = QLabel()
        self.science_info_label.setWordWrap(True)
        self.science_info_label.setStyleSheet("font-size: 11pt; color: #222; margin-top: 8px;")
        self.left_layout.addWidget(self.science_info_label)

        self.left_layout.addStretch()

        splitter_horizontal.addWidget(self.left_widget)

        # Bagian kanan: grafik sinyal
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setSpacing(20)

        splitter_vertical = QSplitter(Qt.Vertical)

        # Grafik sinyal pernapasan
        self.fig_resp = Figure(figsize=(7, 3), dpi=100)
        self.ax_resp = self.fig_resp.add_subplot(111)
        self.canvas_resp = FigureCanvas(self.fig_resp)
        resp_widget = QWidget()
        resp_layout = QVBoxLayout(resp_widget)
        resp_layout.addWidget(self.canvas_resp)
        splitter_vertical.addWidget(resp_widget)

        self.line_resp, = self.ax_resp.plot([], [], color='#007bff', linewidth=2)
        self.ax_resp.set_title('Sinyal Pernapasan', fontsize=14)
        self.ax_resp.set_xlabel('Waktu (detik)', fontsize=12)
        self.ax_resp.set_ylabel('Amplitudo', fontsize=12)
        self.ax_resp.grid(True)

        # Grafik sinyal rPPG (detak jantung)
        self.fig_rppg = Figure(figsize=(7, 3), dpi=100)
        self.ax_rppg = self.fig_rppg.add_subplot(111)
        self.canvas_rppg = FigureCanvas(self.fig_rppg)
        rppg_widget = QWidget()
        rppg_layout = QVBoxLayout(rppg_widget)
        rppg_layout.addWidget(self.canvas_rppg)
        splitter_vertical.addWidget(rppg_widget)

        self.line_rppg, = self.ax_rppg.plot([], [], color='#dc3545', linewidth=2)
        self.ax_rppg.set_title('Sinyal Detak Jantung (rPPG)', fontsize=14)
        self.ax_rppg.set_xlabel('Waktu (detik)', fontsize=12)
        self.ax_rppg.set_ylabel('Amplitudo', fontsize=12)
        self.ax_rppg.grid(True)

        splitter_vertical.setStretchFactor(0, 1)
        splitter_vertical.setStretchFactor(1, 1)

        self.right_layout.addWidget(splitter_vertical)
        splitter_horizontal.addWidget(self.right_widget)

        splitter_horizontal.setStretchFactor(0, 1)
        splitter_horizontal.setStretchFactor(1, 2)

        self.main_layout.addWidget(splitter_horizontal)

        # Thread video dan sinyal
        self.video_thread = VideoProcessor()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_data_signal.connect(self.store_data_for_plot)
        self.video_thread.update_stable_time_signal.connect(self.update_stable_time_info)
        self.video_thread.update_amplitude_signal.connect(self.update_amplitude_info)
        self.video_thread.update_face_in_roi_signal.connect(self.update_face_position_info)

        # Timer untuk update plot secara berkala
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(150)  # update tiap 150 ms
        self.plot_timer.timeout.connect(self.update_plots)

        # Inisialisasi data kosong untuk plot dan status
        self.time_data = np.array([])
        self.resp_data = np.array([])
        self.rppg_data = np.array([])
        self.current_fps = FPS_DEFAULT

        self.amp_resp_history = []
        self.amp_rppg_history = []
        self.current_stable_time = 0.0

        self.create_menu()

    # Membuat menu 'Bantuan' untuk penjelasan parameter
    def create_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Bantuan")

        penjelasan_action = QAction("Penjelasan Parameter Filter", self)
        penjelasan_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(penjelasan_action)

    def show_help_dialog(self):
        dlg = HelpDialog(self)
        dlg.exec_()

    # Mulai proses video dan pengolahan sinyal
    def start_processing(self):
        self.reset_button.setEnabled(False)
        self.amp_resp_history = []
        self.amp_rppg_history = []
        if not self.video_thread.isRunning():
            self.video_thread._run_flag = True
            self.video_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.apply_filter_changes()
        self.plot_timer.start()

    # Berhenti proses video dan pengolahan sinyal
    def stop_processing(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.plot_timer.stop()
        self.signal_quality_label.setText("")
        self.amp_resp_history = []
        self.amp_rppg_history = []

    # Reset pengukuran dan data
    def reset_measurement(self):
        self.current_stable_time = 0.0
        self.amp_resp_history.clear()
        self.amp_rppg_history.clear()
        self.stable_info_label.setText("Wajah belum stabil di kotak.")
        self.signal_quality_label.setText("")
        self.amplitude_label.setText("Amplitudo Pernapasan: 0.00 | Amplitudo Detak Jantung: 0.00")

        self.reset_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if self.video_thread.isRunning():
            self.video_thread.stop()

    # Terapkan perubahan parameter filter dari GUI
    def apply_filter_changes(self):
        try:
            resp_low = float(self.resp_low_input.text())
            resp_high = float(self.resp_high_input.text())
            resp_order = int(self.resp_order_input.text())
            ppg_low = float(self.ppg_low_input.text())
            ppg_high = float(self.ppg_high_input.text())
            ppg_order = int(self.ppg_order_input.text())

            # Validasi input
            if not (0 < resp_low < resp_high):
                raise ValueError("Filter Low Cut Respirasi harus lebih kecil dari High Cut dan keduanya positif.")
            if not (0 < ppg_low < ppg_high):
                raise ValueError("Filter Low Cut PPG harus lebih kecil dari High Cut dan keduanya positif.")
            if resp_order <= 0 or ppg_order <= 0:
                raise ValueError("Orde filter harus bilangan positif.")

            self.video_thread.set_filter_params(resp_low, resp_high, resp_order, ppg_low, ppg_high, ppg_order)
            QMessageBox.information(self, "Sukses", "Parameter filter berhasil diperbarui.")
        except ValueError as e:
            QMessageBox.warning(self, "Kesalahan Input",
                                f"Terjadi kesalahan saat mengubah parameter filter:\n{e}\nSilakan masukkan angka yang valid.")

    # Update gambar video di GUI
    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    # Simpan data hasil proses video untuk plot
    def store_data_for_plot(self, time_arr, resp_arr, rppg_arr, fps):
        self.time_data = time_arr
        self.resp_data = resp_arr
        self.rppg_data = rppg_arr
        self.current_fps = fps

    # Update plot sinyal pernapasan dan rPPG secara berkala
    def update_plots(self):
        if self.time_data.size == 0:
            return

        # Filter sinyal sebelum plot agar terlihat jelas
        filtered_resp = butter_bandpass_filter(
            self.resp_data,
            self.video_thread.lowcut_resp,
            self.video_thread.highcut_resp,
            self.current_fps,
            self.video_thread.order_resp
        )
        filtered_rppg = butter_bandpass_filter(
            self.rppg_data,
            self.video_thread.lowcut_ppg,
            self.video_thread.highcut_ppg,
            self.current_fps,
            self.video_thread.order_ppg
        )

        # Update garis plot pernapasan
        self.line_resp.set_data(self.time_data, filtered_resp)
        self.ax_resp.relim()
        self.ax_resp.autoscale_view(True, True, True)
        self.canvas_resp.draw_idle()

        # Update garis plot rPPG
        self.line_rppg.set_data(self.time_data, filtered_rppg)
        self.ax_rppg.relim()
        self.ax_rppg.autoscale_view(True, True, True)
        self.canvas_rppg.draw_idle()

        # Deteksi puncak dan hitung BPM pernapasan
        peaks_resp, _ = find_peaks(
            filtered_resp,
            distance=max(1, int(self.current_fps / self.video_thread.highcut_resp * 0.5)),
            height=np.std(filtered_resp) * 0.3
        )
        if len(peaks_resp) > 1:
            avg_interval = np.mean(np.diff(peaks_resp))
            bpm_resp = 60 / (avg_interval / self.current_fps)
        else:
            bpm_resp = 0.0

        # Deteksi puncak dan hitung BPM rPPG
        peaks_ppg, _ = find_peaks(
            filtered_rppg,
            distance=max(1, int(self.current_fps / self.video_thread.highcut_ppg * 0.5)),
            height=np.std(filtered_rppg) * 0.5
        )
        if len(peaks_ppg) > 1:
            avg_interval_ppg = np.mean(np.diff(peaks_ppg))
            bpm_ppg = 60 / (avg_interval_ppg / self.current_fps)
        else:
            bpm_ppg = 0.0

        # Update label metrik di GUI
        self.metrics_label.setText(
            f"FPS: {self.current_fps:.2f} | Pernapasan (BPM): {bpm_resp:.2f} | Detak Jantung (BPM): {bpm_ppg:.2f}"
        )

    # Update info durasi wajah stabil di ROI
    def update_stable_time_info(self, seconds):
        self.current_stable_time = seconds
        if seconds >= 30:
            self.stable_info_label.setText(
                "<b>✅ Wajah stabil di kotak selama 30 detik.<br>"
                "Pengukuran detak jantung dan pernapasan dapat dipercaya.</b>"
            )
            self.stable_info_label.setStyleSheet("font-size: 11pt; color: green; font-weight: bold; margin-top: 8px;")
        elif seconds > 0:
            self.stable_info_label.setText(
                f"Wajah stabil di kotak selama {seconds:.1f} detik. "
                "Harap tetap diam hingga 30 detik untuk hasil optimal."
            )
            self.stable_info_label.setStyleSheet("font-size: 11pt; color: #555; margin-top: 8px; font-style: italic;")
        else:
            self.stable_info_label.setText("Wajah belum stabil di kotak.")
            self.stable_info_label.setStyleSheet("font-size: 11pt; color: #555; font-style: italic; margin-top: 8px;")

    # Update info amplitudo sinyal napas dan jantung
    def update_amplitude_info(self, amp_resp, amp_rppg):
        self.amplitude_label.setText(
            f"Amplitudo Pernapasan: {amp_resp:.3f} | Amplitudo Detak Jantung: {amp_rppg:.3f}"
        )
        self.update_science_info(amp_resp, amp_rppg)

        if self.current_stable_time >= 30:
            if not hasattr(self, 'amp_resp_history'):
                self.amp_resp_history = []
                self.amp_rppg_history = []
            self.amp_resp_history.append(amp_resp)
            self.amp_rppg_history.append(amp_rppg)
            if len(self.amp_resp_history) > 30:
                self.amp_resp_history.pop(0)
            if len(self.amp_rppg_history) > 30:
                self.amp_rppg_history.pop(0)

            mean_resp = np.mean(self.amp_resp_history)
            mean_rppg = np.mean(self.amp_rppg_history)

            resp_status, resp_msg = self.kategorikan_respirasi(mean_resp)
            rppg_status, rppg_msg = self.kategorikan_rppg(mean_rppg)

            info = (
                f"<b>Status Sinyal Setelah 30 Detik:</b><br>"
                f"<u>Pernapasan:</u> {resp_status} - {resp_msg}<br>"
                f"<u>Detak Jantung (rPPG):</u> {rppg_status} - {rppg_msg}"
            )
            self.signal_quality_label.setText(info)

            # Aktifkan tombol ulangi pengukuran
            self.reset_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    # Update info posisi wajah dalam ROI
    def update_face_position_info(self, face_in_roi):
        if face_in_roi:
            self.face_pos_label.setText("Wajah terdeteksi dalam kotak hijau. Silakan diam dan tetap di posisi.")
            self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: green; margin-top: 8px;")
        else:
            self.face_pos_label.setText("📌 Silakan posisikan wajah ke tengah kotak hijau agar pengukuran akurat.")
            self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: #b03a2e; margin-top: 8px;")

    # Update info sains singkat untuk pengguna
    def update_science_info(self, amp_resp, amp_rppg):
        info_text = (
            "<b>Info Sains Singkat:</b><br>"
            "• <b>Sinyal Pernapasan:</b> Amplitudo sinyal menandakan intensitas napas. Semakin besar, semakin dalam napas Anda.<br>"
            "• <b>Sinyal rPPG (Detak Jantung):</b> Amplitudo menunjukkan kekuatan sinyal detak jantung wajah.<br>"
            "• Sinyal stabil dan cukup kuat penting untuk hasil akurat.<br>"
            "• Pastikan pencahayaan dan posisi wajah baik untuk hasil terbaik."
        )
        self.science_info_label.setText(info_text)

    # Kategorisasi kualitas sinyal pernapasan berdasarkan amplitudo rata-rata
    def kategorikan_respirasi(self, mean_amp):
        if mean_amp < 0.02:
            return "Kurang Bagus", "Napas kamu kayak bisikan angin... coba tarik napas dalam-dalam, ya!"
        elif mean_amp > 0.1:
            return "Bagus", "Wah, napasnya dalam dan kuat! Ini sinyal sehat banget, lanjutkan!"
        else:
            return "Normal", "Napas kamu stabil dan nyaman, tanda sistem pernapasan bekerja dengan baik."

    # Kategorisasi kualitas sinyal rPPG berdasarkan amplitudo rata-rata
    def kategorikan_rppg(self, mean_amp):
        if mean_amp < 0.02:
            return "Kurang Bagus", "Detak jantungnya kayak malu-malu, coba perbaiki pencahayaan atau posisikan wajah lebih jelas."
        elif mean_amp > 0.05:
            return "Bagus", "Detak jantung kuat dan jelas terekam! Mantap, kesehatan jantung terpantau baik."
        else:
            return "Normal", "Detak jantung terdeteksi dengan baik, terus jaga gaya hidup sehat ya."

    def closeEvent(self, event):
        self.stop_processing()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
    
