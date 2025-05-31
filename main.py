import sys  # Untuk mengakses fungsi sistem seperti argument dan exit program
import cv2  # OpenCV, untuk pengolahan gambar dan video (kamera)
import numpy as np  # Library untuk operasi numerik dan array
from scipy.signal import butter, filtfilt, find_peaks  # Fungsi filter sinyal dan deteksi puncak
from collections import deque  # Struktur data antrian yang bisa dibatasi ukurannya
import time  # Untuk mengukur waktu dan delay
import mediapipe as mp  # Library untuk deteksi wajah, pose, dan landmark tubuh secara realtime

# Import komponen PyQt5 untuk membuat tampilan GUI (jendela aplikasi)
from PyQt5.QtWidgets import (
    QApplication,  # Objek aplikasi PyQt
    QMainWindow,  # Window utama aplikasi
    QLabel,  # Widget untuk menampilkan teks atau gambar
    QPushButton,  # Tombol klik
    QVBoxLayout, QHBoxLayout,  # Layout untuk atur widget secara vertikal atau horizontal
    QWidget,  # Widget dasar untuk menampung widget lain
    QFrame,  # Widget bingkai untuk grup widget lain
    QLineEdit,  # Input teks satu baris
    QFormLayout,  # Layout khusus untuk form label dan input
    QMessageBox,  # Kotak dialog pesan
    QScrollArea,  # Area dengan scrollbar untuk widget besar
    QSplitter,  # Widget untuk membagi area menjadi beberapa bagian yang bisa diubah ukurannya
    QAction,  # Aksi untuk menu atau toolbar
    QTextEdit,  # Input teks banyak baris (rich text)
    QDialog,  # Dialog popup
    QDialogButtonBox  # Tombol OK/Cancel di dialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont  # Untuk gambar, ikon, dan pengaturan font
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt  # Timer untuk event berkala, thread untuk proses paralel, sinyal komunikasi antar objek, dan konstanta Qt

# Import modul untuk menggambar grafik di GUI menggunakan matplotlib dengan integrasi PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Canvas untuk gambar grafik di PyQt
from matplotlib.figure import Figure  # Objek figure atau gambar grafik


# --- Parameter global untuk pengaturan kamera dan filter ---
WEBCAM_INDEX = 0  # Nomor kamera default, biasanya 0 untuk webcam bawaan
FRAME_WIDTH = 640  # Lebar frame video webcam
FRAME_HEIGHT = 480  # Tinggi frame video webcam
FPS_DEFAULT = 30  # Frame per detik default jika webcam tidak melaporkan fps

# Parameter filter untuk sinyal napas (respirasi)
DEFAULT_LOWCUT_RESP = 0.1  # Frekuensi bawah filter napas (Hz)
DEFAULT_HIGHCUT_RESP = 0.5  # Frekuensi atas filter napas (Hz)
DEFAULT_ORDER_RESP = 2  # Orde filter napas (tingkat kekuatan filter)

# Parameter filter untuk sinyal detak jantung (rPPG)
DEFAULT_LOWCUT_PPG = 0.8  # Frekuensi bawah filter detak jantung (Hz)
DEFAULT_HIGHCUT_PPG = 2.5  # Frekuensi atas filter detak jantung (Hz)
DEFAULT_ORDER_PPG = 3  # Orde filter detak jantung

BUFFER_SIZE_SEC = 10  # Ukuran data buffer selama 10 detik

# --- Fungsi untuk membuat filter bandpass (memilih frekuensi tertentu) ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Frekuensi Nyquist = setengah dari frekuensi sampling
    low = lowcut / nyq  # Normalisasi frekuensi bawah
    high = highcut / nyq  # Normalisasi frekuensi atas
    b, a = butter(order, [low, high], btype='band')  # Hitung koefisien filter
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Memfilter data dengan bandpass filter tanpa menimbulkan delay sinyal
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    padlen = 3 * (max(len(a), len(b)) - 1)  # Panjang padding untuk filtfilt
    if len(data) <= padlen:
        return data  # Kalau data terlalu pendek, tidak difilter
    try:
        y = filtfilt(b, a, data)  # Filter data tanpa delay phase
    except Exception as e:
        print(f"Filter error: {e}")  # Jika error saat filtering, tampilkan pesan
        return data  # Kembalikan data asli kalau error
    return y

# --- Kelas thread untuk proses video dan sinyal secara terpisah agar GUI tetap responsif ---
class VideoProcessor(QThread):
    # Sinyal untuk mengirim data dan gambar dari thread ke GUI
    change_pixmap_signal = pyqtSignal(QImage)  # Untuk update video ke GUI
    update_data_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, float)  # Kirim data waktu, napas, rPPG, fps
    update_stable_time_signal = pyqtSignal(float)  # Kirim durasi wajah stabil di ROI
    update_amplitude_signal = pyqtSignal(float, float)  # Kirim amplitudo sinyal napas dan jantung
    update_face_in_roi_signal = pyqtSignal(bool)  # Kirim info apakah wajah terdeteksi di kotak hijau

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True  # Flag untuk mengontrol thread berjalan atau berhenti
        self.fps = FPS_DEFAULT  # FPS awal
        self.buffer_length = int(BUFFER_SIZE_SEC * self.fps)  # Panjang buffer data sesuai durasi dan fps

        # Buffer data menggunakan deque agar bisa simpan data terbaru dengan batas tertentu
        self.time_buffer = deque(maxlen=self.buffer_length)  # Buffer waktu
        self.respiration_signal_buffer = deque(maxlen=self.buffer_length)  # Buffer sinyal napas
        self.r_channel_buffer = deque(maxlen=self.buffer_length)  # Buffer warna merah
        self.g_channel_buffer = deque(maxlen=self.buffer_length)  # Buffer warna hijau
        self.b_channel_buffer = deque(maxlen=self.buffer_length)  # Buffer warna biru

        # Parameter filter default (frekuensi dan orde)
        self.lowcut_resp = DEFAULT_LOWCUT_RESP
        self.highcut_resp = DEFAULT_HIGHCUT_RESP
        self.order_resp = DEFAULT_ORDER_RESP
        self.lowcut_ppg = DEFAULT_LOWCUT_PPG
        self.highcut_ppg = DEFAULT_HIGHCUT_PPG
        self.order_ppg = DEFAULT_ORDER_PPG

        # Inisialisasi MediaPipe FaceMesh dan Pose untuk deteksi wajah dan tubuh
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

        self.prev_chest_y = None  # Posisi dada sebelumnya, untuk deteksi gerak napas
        self.roi_size = 200  # Ukuran kotak hijau (ROI) untuk ambil warna wajah

        self.face_in_roi_start_time = None  # Waktu mulai wajah stabil di ROI
        self.stable_time = 0.0  # Durasi wajah stabil di ROI

    # Fungsi untuk update parameter filter dari GUI
    def set_filter_params(self, resp_low, resp_high, resp_order, ppg_low, ppg_high, ppg_order):
        self.lowcut_resp = resp_low
        self.highcut_resp = resp_high
        self.order_resp = resp_order
        self.lowcut_ppg = ppg_low
        self.highcut_ppg = ppg_high
        self.order_ppg = ppg_order

    # Algoritma CHROM untuk ekstraksi sinyal detak jantung dari warna RGB wajah
    def chrom_rppg(self, R, G, B):
        min_len = min(len(R), len(G), len(B))  # Panjang data terpendek
        R_arr = np.array(R)[-min_len:]
        G_arr = np.array(G)[-min_len:]
        B_arr = np.array(B)[-min_len:]
        epsilon = 1e-6  # Untuk menghindari pembagian dengan nol
        sum_rgb = R_arr + G_arr + B_arr + epsilon
        R_norm = R_arr / sum_rgb
        G_norm = G_arr / sum_rgb
        B_norm = B_arr / sum_rgb
        # Kombinasi linier dari RGB untuk dapat sinyal rPPG
        Y_chrom = 1.5 * R_norm + G_norm - 1.5 * B_norm
        return Y_chrom

    # Fungsi utama thread untuk memproses video dan sinyal secara terus menerus
    def run(self):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)  # Buka webcam
        if not self.cap.isOpened():
            print("Error: Tidak dapat membuka webcam.")
            self._run_flag = False
            return

        # Set ukuran frame webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Coba dapatkan FPS sebenarnya dari webcam
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = actual_fps
        else:
            self.fps = FPS_DEFAULT

        # Reset buffer sesuai fps
        self.buffer_length = int(BUFFER_SIZE_SEC * self.fps)
        self.time_buffer = deque(maxlen=self.buffer_length)
        self.respiration_signal_buffer = deque(maxlen=self.buffer_length)
        self.r_channel_buffer = deque(maxlen=self.buffer_length)
        self.g_channel_buffer = deque(maxlen=self.buffer_length)
        self.b_channel_buffer = deque(maxlen=self.buffer_length)

        self.prev_chest_y = None  # Reset posisi dada sebelumnya
        start_time = time.time()

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                continue  # Jika gagal baca frame, lanjut ke frame berikutnya

            current_time = time.time() - start_time
            frame = cv2.flip(frame, 1)  # Mirror horizontal agar terasa natural
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB

            # Tentukan kotak hijau (ROI) di tengah frame
            cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
            half = self.roi_size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            # Proses deteksi pose untuk ambil landmark tubuh (terutama bahu)
            pose_results = self.pose.process(frame_rgb)

            face_detected = False
            chest_movement = 0.0

            if pose_results.pose_landmarks:
                # Gambar titik-titik landmark pose di frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # Cek apakah landmark terlihat cukup jelas
                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    current_chest_y = (left_shoulder.y + right_shoulder.y) / 2 * FRAME_HEIGHT

                    # Hitung perubahan posisi dada, indikasi napas
                    if self.prev_chest_y is not None:
                        chest_movement = current_chest_y - self.prev_chest_y
                    self.prev_chest_y = current_chest_y
                    self.respiration_signal_buffer.append(chest_movement)  # Simpan perubahan dada
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

            # Hitung durasi wajah stabil di ROI
            if face_detected:
                if self.face_in_roi_start_time is None:
                    self.face_in_roi_start_time = time.time()
                self.stable_time = time.time() - self.face_in_roi_start_time
            else:
                self.face_in_roi_start_time = None
                self.stable_time = 0.0

            self.update_stable_time_signal.emit(self.stable_time)

            # Ambil area ROI warna untuk sinyal rPPG
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size > 0:
                avg_color = np.mean(np.mean(roi, axis=0), axis=0)  # Rata-rata warna RGB di ROI
                self.r_channel_buffer.append(avg_color[0])
                self.g_channel_buffer.append(avg_color[1])
                self.b_channel_buffer.append(avg_color[2])
            else:
                self.r_channel_buffer.append(0.0)
                self.g_channel_buffer.append(0.0)
                self.b_channel_buffer.append(0.0)

            self.time_buffer.append(current_time)

            # Ambil data buffer sinyal napas dan rPPG
            current_resp_signal = np.array(self.respiration_signal_buffer)
            current_rppg_signal = self.chrom_rppg(self.r_channel_buffer, self.g_channel_buffer, self.b_channel_buffer)

            # Filter sinyal napas dan hitung amplitudo
            if len(current_resp_signal) >= 5:
                filtered_resp_signal = butter_bandpass_filter(
                    current_resp_signal,
                    self.lowcut_resp,
                    self.highcut_resp,
                    self.fps,
                    self.order_resp
                )
                amplitude_resp = np.sqrt(np.mean(filtered_resp_signal ** 2))  # Hitung energi sinyal
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

            # Kirim amplitudo sinyal ke GUI
            self.update_amplitude_signal.emit(amplitude_resp, amplitude_rppg)

            # Kirim data waktu dan sinyal ke GUI untuk plot grafik
            self.update_data_signal.emit(
                np.array(self.time_buffer),
                current_resp_signal,
                current_rppg_signal,
                self.fps
            )

            # Gambar kotak hijau (ROI) di frame untuk tanda area pengambilan data rPPG
            overlay = frame.copy()
            color = (0, 255, 0)  # Warna hijau
            thickness = 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            alpha = 0.3  # Transparansi overlay
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Konversi frame OpenCV jadi QImage untuk ditampilkan di PyQt GUI
            qt_img = self.convert_cv_qt(frame)
            self.change_pixmap_signal.emit(qt_img)

        # Jika thread dihentikan, release sumber daya
        self.cap.release()
        self.face_mesh.close()
        self.pose.close()

    # Fungsi konversi frame OpenCV (BGR) ke QImage untuk PyQt
    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio)
        return p

    # Fungsi untuk stop thread dengan aman
    def stop(self):
        self._run_flag = False
        self.wait()  # Tunggu sampai thread selesai


# --- Dialog bantuan yang muncul ketika user ingin tahu arti parameter filter ---
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
        <!-- Penjelasan singkat dan mudah dipahami tentang fungsi parameter filter -->
        """)
        layout.addWidget(text)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)


# --- Window utama GUI dengan PyQt ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor Real-time Pernapasan & rPPG")
        self.showMaximized()  # Tampilkan jendela fullscreen

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout utama horizontal membagi area menjadi dua: kiri dan kanan
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(15)

        splitter_horizontal = QSplitter(Qt.Horizontal)

        # Bagian kiri: tampilan video, tombol kontrol, dan pengaturan filter
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

        # Layout untuk tombol mulai, berhenti, terapkan filter, dan reset
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        # Tombol mulai proses pengambilan data
        self.start_button = QPushButton("Mulai")
        self.start_button.setMinimumHeight(42)
        self.start_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #007bff; color: white; border-radius: 6px;")
        self.start_button.clicked.connect(self.start_processing)

        # Tombol berhenti proses pengambilan data
        self.stop_button = QPushButton("Berhenti")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #dc3545; color: white; border-radius: 6px;")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)  # Tombol berhenti awalnya nonaktif

        # Tombol untuk menerapkan perubahan filter yang diinput pengguna
        self.apply_filter_button = QPushButton("Terapkan Filter")
        self.apply_filter_button.setMinimumHeight(42)
        self.apply_filter_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #28a745; color: white; border-radius: 6px;")
        self.apply_filter_button.clicked.connect(self.apply_filter_changes)

        # Tombol untuk mengulang pengukuran (reset)
        self.reset_button = QPushButton("Ulangi Pengukuran")
        self.reset_button.setMinimumHeight(42)
        self.reset_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #ffc107; color: black; border-radius: 6px;")
        self.reset_button.clicked.connect(self.reset_measurement)
        self.reset_button.setEnabled(False)  # Tombol reset awalnya nonaktif

        # Tambah tombol ke layout kontrol
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.apply_filter_button)
        control_layout.addWidget(self.reset_button)

        self.left_layout.addLayout(control_layout)

        # Label menampilkan metrik seperti FPS dan BPM napas dan jantung
        self.metrics_label = QLabel("FPS: 0 | Pernapasan (BPM): 0,00 | Detak Jantung (BPM): 0,00")
        self.metrics_label.setStyleSheet("font-size: 17pt; font-weight: bold; color: #333; margin-top: 12px;")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.metrics_label)

        # Label instruksi posisi wajah
        self.face_pos_label = QLabel("")
        self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: #b03a2e; margin-top: 8px;")
        self.face_pos_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.face_pos_label)

        # Label menampilkan amplitudo sinyal napas dan jantung
        self.amplitude_label = QLabel("Amplitudo Pernapasan: 0.00 | Amplitudo Detak Jantung: 0.00")
        self.amplitude_label.setStyleSheet("font-size: 11pt; color: #444; margin-top: 8px;")
        self.amplitude_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.amplitude_label)

        # Frame berisi input parameter filter dengan scroll agar tidak terlalu panjang
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

        # Input parameter filter respirasi
        self.resp_low_input = QLineEdit(str(DEFAULT_LOWCUT_RESP))
        setup_lineedit(self.resp_low_input, "Hz (contoh: 0.1)")
        self.resp_high_input = QLineEdit(str(DEFAULT_HIGHCUT_RESP))
        setup_lineedit(self.resp_high_input, "Hz (contoh: 0.5)")
        self.resp_order_input = QLineEdit(str(DEFAULT_ORDER_RESP))
        setup_lineedit(self.resp_order_input, "Orde (contoh: 2)")

        # Input parameter filter PPG (detak jantung)
        self.ppg_low_input = QLineEdit(str(DEFAULT_LOWCUT_PPG))
        setup_lineedit(self.ppg_low_input, "Hz (contoh: 0.8)")
        self.ppg_high_input = QLineEdit(str(DEFAULT_HIGHCUT_PPG))
        setup_lineedit(self.ppg_high_input, "Hz (contoh: 2.5)")
        self.ppg_order_input = QLineEdit(str(DEFAULT_ORDER_PPG))
        setup_lineedit(self.ppg_order_input, "Orde (contoh: 3)")

        # Tambahkan input ke form layout
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

        # Label status kualitas sinyal setelah 30 detik pengukuran
        self.signal_quality_label = QLabel("")
        self.signal_quality_label.setWordWrap(True)
        self.signal_quality_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #0056b3; margin-top: 12px;")
        self.left_layout.addWidget(self.signal_quality_label)

        # Label info singkat ilmiah
        self.science_info_label = QLabel()
        self.science_info_label.setWordWrap(True)
        self.science_info_label.setStyleSheet("font-size: 11pt; color: #222; margin-top: 8px;")
        self.left_layout.addWidget(self.science_info_label)

        self.left_layout.addStretch()

        splitter_horizontal.addWidget(self.left_widget)

        # Bagian kanan GUI: menampilkan grafik sinyal
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

        # Buat thread video yang akan proses video secara paralel
        self.video_thread = VideoProcessor()
        # Sambungkan sinyal thread ke fungsi update GUI
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_data_signal.connect(self.store_data_for_plot)
        self.video_thread.update_stable_time_signal.connect(self.update_stable_time_info)
        self.video_thread.update_amplitude_signal.connect(self.update_amplitude_info)
        self.video_thread.update_face_in_roi_signal.connect(self.update_face_position_info)

        # Timer untuk update plot grafik setiap 150 ms
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(150)
        self.plot_timer.timeout.connect(self.update_plots)

        # Data kosong untuk plot dan status awal
        self.time_data = np.array([])
        self.resp_data = np.array([])
        self.rppg_data = np.array([])
        self.current_fps = FPS_DEFAULT

        self.amp_resp_history = []
        self.amp_rppg_history = []
        self.current_stable_time = 0.0

        self.create_menu()

    # Membuat menu bantuan di menu bar
    def create_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Bantuan")

        penjelasan_action = QAction("Penjelasan Parameter Filter", self)
        penjelasan_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(penjelasan_action)

    # Tampilkan dialog bantuan parameter filter
    def show_help_dialog(self):
        dlg = HelpDialog(self)
        dlg.exec_()

    # Fungsi mulai pengolahan video dan sinyal
    def start_processing(self):
        self.reset_button.setEnabled(False)
        self.amp_resp_history = []
        self.amp_rppg_history = []
        if not self.video_thread.isRunning():
            self.video_thread._run_flag = True
            self.video_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.apply_filter_changes()  # Terapkan parameter filter terbaru
        self.plot_timer.start()

    # Fungsi berhenti pengolahan video dan sinyal
    def stop_processing(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.plot_timer.stop()
        self.signal_quality_label.setText("")
        self.amp_resp_history = []
        self.amp_rppg_history = []

    # Fungsi reset pengukuran dan data
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

    # Terapkan perubahan parameter filter dari input GUI ke thread video
    def apply_filter_changes(self):
        try:
            resp_low = float(self.resp_low_input.text())
            resp_high = float(self.resp_high_input.text())
            resp_order = int(self.resp_order_input.text())
            ppg_low = float(self.ppg_low_input.text())
            ppg_high = float(self.ppg_high_input.text())
            ppg_order = int(self.ppg_order_input.text())

            # Validasi agar input parameter masuk akal
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

    # Update gambar video di label GUI
    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    # Simpan data hasil proses video untuk plot grafik
    def store_data_for_plot(self, time_arr, resp_arr, rppg_arr, fps):
        self.time_data = time_arr
        self.resp_data = resp_arr
        self.rppg_data = rppg_arr
        self.current_fps = fps

    # Update grafik sinyal pernapasan dan rPPG secara berkala
    def update_plots(self):
        if self.time_data.size == 0:
            return  # Jika belum ada data, tidak perlu plot

        # Filter sinyal napas dan rPPG untuk hasil yang lebih halus dan jelas
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

        # Update garis grafik napas
        self.line_resp.set_data(self.time_data, filtered_resp)
        self.ax_resp.relim()
        self.ax_resp.autoscale_view(True, True, True)
        self.canvas_resp.draw_idle()

        # Update garis grafik detak jantung
        self.line_rppg.set_data(self.time_data, filtered_rppg)
        self.ax_rppg.relim()
        self.ax_rppg.autoscale_view(True, True, True)
        self.canvas_rppg.draw_idle()

        # Deteksi puncak sinyal napas untuk hitung BPM
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

        # Deteksi puncak sinyal rPPG untuk hitung BPM
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

        # Update label metrik FPS dan BPM di GUI
        self.metrics_label.setText(
            f"FPS: {self.current_fps:.2f} | Pernapasan (BPM): {bpm_resp:.2f} | Detak Jantung (BPM): {bpm_ppg:.2f}"
        )

    # Update info durasi wajah stabil di kotak hijau
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

    # Update info amplitudo sinyal napas dan detak jantung
    def update_amplitude_info(self, amp_resp, amp_rppg):
        self.amplitude_label.setText(
            f"Amplitudo Pernapasan: {amp_resp:.3f} | Amplitudo Detak Jantung: {amp_rppg:.3f}"
        )
        self.update_science_info(amp_resp, amp_rppg)

        # Setelah 30 detik wajah stabil, cek kualitas sinyal berdasarkan amplitudo rata-rata
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

            # Aktifkan tombol reset pengukuran, matikan tombol mulai dan berhenti
            self.reset_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    # Update info posisi wajah dalam ROI (kotak hijau)
    def update_face_position_info(self, face_in_roi):
        if face_in_roi:
            self.face_pos_label.setText("Wajah terdeteksi dalam kotak hijau. Silakan diam dan tetap di posisi.")
            self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: green; margin-top: 8px;")
        else:
            self.face_pos_label.setText("📌 Silakan posisikan wajah ke tengah kotak hijau agar pengukuran akurat.")
            self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: #b03a2e; margin-top: 8px;")

    # Update info ilmiah singkat untuk pengguna tentang sinyal
    def update_science_info(self, amp_resp, amp_rppg):
        info_text = (
            "<b>Info Sains Singkat:</b><br>"
            "• <b>Sinyal Pernapasan:</b> Amplitudo sinyal menandakan intensitas napas. Semakin besar, semakin dalam napas Anda.<br>"
            "• <b>Sinyal rPPG (Detak Jantung):</b> Amplitudo menunjukkan kekuatan sinyal detak jantung wajah.<br>"
            "• Sinyal stabil dan cukup kuat penting untuk hasil akurat.<br>"
            "• Pastikan pencahayaan dan posisi wajah baik untuk hasil terbaik."
        )
        self.science_info_label.setText(info_text)

    # Kategorikan kualitas sinyal napas berdasarkan amplitudo
    def kategorikan_respirasi(self, mean_amp):
        if mean_amp < 0.02:
            return "Kurang Bagus", "Napas kamu kayak bisikan angin... coba tarik napas dalam-dalam, ya!"
        elif mean_amp > 0.1:
            return "Bagus", "Wah, napasnya dalam dan kuat! Ini sinyal sehat banget, lanjutkan!"
        else:
            return "Normal", "Napas kamu stabil dan nyaman, tanda sistem pernapasan bekerja dengan baik."

    # Kategorikan kualitas sinyal rPPG berdasarkan amplitudo
    def kategorikan_rppg(self, mean_amp):
        if mean_amp < 0.02:
            return "Kurang Bagus", "Detak jantungnya kayak malu-malu, coba perbaiki pencahayaan atau posisikan wajah lebih jelas."
        elif mean_amp > 0.05:
            return "Bagus", "Detak jantung kuat dan jelas terekam! Mantap, kesehatan jantung terpantau baik."
        else:
            return "Normal", "Detak jantung terdeteksi dengan baik, terus jaga gaya hidup sehat ya."

    # Saat window ditutup, pastikan thread video dihentikan
    def closeEvent(self, event):
        self.stop_processing()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Set tema GUI
    window = MainWindow()  # Buat window utama
    window.show()
    sys.exit(app.exec_())  # Mulai event loop PyQt
