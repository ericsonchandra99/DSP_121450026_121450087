# src/main_window.py

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFrame, QLineEdit, QFormLayout, QMessageBox, QScrollArea,
    QSplitter, QAction, QTextEdit, QDialog, QDialogButtonBox, QApplication
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from scipy.signal import find_peaks

from src.video_processor import VideoProcessor, butter_bandpass_filter, DEFAULT_LOWCUT_RESP, DEFAULT_HIGHCUT_RESP, DEFAULT_ORDER_RESP, DEFAULT_LOWCUT_PPG, DEFAULT_HIGHCUT_PPG, DEFAULT_ORDER_PPG
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from src.video_processor import FPS_DEFAULT




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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor Real-time Pernapasan & rPPG")
        self.showMaximized()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(15)

        splitter_horizontal = QSplitter(Qt.Horizontal)

        # Panel kiri
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setSpacing(12)

        self.video_label = QLabel("Memuat video...")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            background-color: black;
            border: 3px solid #28a745;
            border-radius: 6px;
        """)
        self.left_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Tombol kontrol
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.start_button = QPushButton("Mulai")
        self.start_button.setMinimumHeight(42)
        self.start_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #007bff; color: white; border-radius: 6px;")
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QPushButton("Berhenti")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #dc3545; color: white; border-radius: 6px;")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)

        self.apply_filter_button = QPushButton("Terapkan Filter")
        self.apply_filter_button.setMinimumHeight(42)
        self.apply_filter_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #28a745; color: white; border-radius: 6px;")
        self.apply_filter_button.clicked.connect(self.apply_filter_changes)

        self.reset_button = QPushButton("Ulangi Pengukuran")
        self.reset_button.setMinimumHeight(42)
        self.reset_button.setStyleSheet(
            "font-weight: bold; font-size: 15pt; background-color: #ffc107; color: black; border-radius: 6px;")
        self.reset_button.clicked.connect(self.reset_measurement)
        self.reset_button.setEnabled(False)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.apply_filter_button)
        control_layout.addWidget(self.reset_button)

        self.left_layout.addLayout(control_layout)

        # Label metrik
        self.metrics_label = QLabel("FPS: 0 | Pernapasan (BPM): 0,00 | Detak Jantung (BPM): 0,00")
        self.metrics_label.setStyleSheet("font-size: 17pt; font-weight: bold; color: #333; margin-top: 12px;")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.metrics_label)

        # Label instruksi posisi wajah
        self.face_pos_label = QLabel("")
        self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: #b03a2e; margin-top: 8px;")
        self.face_pos_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.face_pos_label)

        # Label amplitudo sinyal
        self.amplitude_label = QLabel("Amplitudo Pernapasan: 0.00 | Amplitudo Detak Jantung: 0.00")
        self.amplitude_label.setStyleSheet("font-size: 11pt; color: #444; margin-top: 8px;")
        self.amplitude_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.amplitude_label)

        # Parameter filter dengan scroll
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

        # Label informasi durasi stabil di bawah parameter filter
        self.stable_info_label = QLabel("Wajah belum stabil di kotak.")
        self.stable_info_label.setStyleSheet("font-size: 11pt; color: #555; font-style: italic; margin-top: 8px;")
        self.left_layout.addWidget(self.stable_info_label)

        # Label kategori kualitas sinyal setelah 30 detik
        self.signal_quality_label = QLabel("")
        self.signal_quality_label.setWordWrap(True)
        self.signal_quality_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #0056b3; margin-top: 12px;")
        self.left_layout.addWidget(self.signal_quality_label)

        # Label info sains singkat
        self.science_info_label = QLabel()
        self.science_info_label.setWordWrap(True)
        self.science_info_label.setStyleSheet("font-size: 11pt; color: #222; margin-top: 8px;")
        self.left_layout.addWidget(self.science_info_label)

        self.left_layout.addStretch()

        splitter_horizontal.addWidget(self.left_widget)

        # Panel kanan dengan splitter vertikal (grafik pernapasan & rPPG)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setSpacing(20)

        splitter_vertical = QSplitter(Qt.Vertical)

        # Grafik Pernapasan
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

        # Grafik rPPG
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

        self.plot_timer = QTimer()
        self.plot_timer.setInterval(150)
        self.plot_timer.timeout.connect(self.update_plots)

        self.time_data = np.array([])
        self.resp_data = np.array([])
        self.rppg_data = np.array([])
        self.current_fps = FPS_DEFAULT

        self.amp_resp_history = []
        self.amp_rppg_history = []
        self.current_stable_time = 0.0

        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Bantuan")

        penjelasan_action = QAction("Penjelasan Parameter Filter", self)
        penjelasan_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(penjelasan_action)

    def show_help_dialog(self):
        dlg = HelpDialog(self)
        dlg.exec_()

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

    def stop_processing(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.plot_timer.stop()
        self.signal_quality_label.setText("")
        self.amp_resp_history = []
        self.amp_rppg_history = []

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

    def apply_filter_changes(self):
        try:
            resp_low = float(self.resp_low_input.text())
            resp_high = float(self.resp_high_input.text())
            resp_order = int(self.resp_order_input.text())
            ppg_low = float(self.ppg_low_input.text())
            ppg_high = float(self.ppg_high_input.text())
            ppg_order = int(self.ppg_order_input.text())

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

    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def store_data_for_plot(self, time_arr, resp_arr, rppg_arr, fps):
        self.time_data = time_arr
        self.resp_data = resp_arr
        self.rppg_data = rppg_arr
        self.current_fps = fps

    def update_plots(self):
        if self.time_data.size == 0:
            return

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

        self.line_resp.set_data(self.time_data, filtered_resp)
        self.ax_resp.relim()
        self.ax_resp.autoscale_view(True, True, True)
        self.canvas_resp.draw_idle()

        self.line_rppg.set_data(self.time_data, filtered_rppg)
        self.ax_rppg.relim()
        self.ax_rppg.autoscale_view(True, True, True)
        self.canvas_rppg.draw_idle()

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

        self.metrics_label.setText(
            f"FPS: {self.current_fps:.2f} | Pernapasan (BPM): {bpm_resp:.2f} | Detak Jantung (BPM): {bpm_ppg:.2f}"
        )

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

            # Aktifkan tombol ulangi
            self.reset_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def update_face_position_info(self, face_in_roi):
        if face_in_roi:
            self.face_pos_label.setText("Wajah terdeteksi dalam kotak hijau. Silakan diam dan tetap di posisi.")
            self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: green; margin-top: 8px;")
        else:
            self.face_pos_label.setText("📌 Silakan posisikan wajah ke tengah kotak hijau agar pengukuran akurat.")
            self.face_pos_label.setStyleSheet("font-size: 11pt; font-weight: normal; color: #b03a2e; margin-top: 8px;")

    def update_science_info(self, amp_resp, amp_rppg):
        info_text = (
            "<b>Info Sains Singkat:</b><br>"
            "• <b>Sinyal Pernapasan:</b> Amplitudo sinyal menandakan intensitas napas. Semakin besar, semakin dalam napas Anda.<br>"
            "• <b>Sinyal rPPG (Detak Jantung):</b> Amplitudo menunjukkan kekuatan sinyal detak jantung wajah.<br>"
            "• Sinyal stabil dan cukup kuat penting untuk hasil akurat.<br>"
            "• Pastikan pencahayaan dan posisi wajah baik untuk hasil terbaik."
        )
        self.science_info_label.setText(info_text)

    def kategorikan_respirasi(self, mean_amp):
        if mean_amp < 0.02:
            return "Kurang Bagus", "Napas kamu kayak bisikan angin... coba tarik napas dalam-dalam, ya!"
        elif mean_amp > 0.1:
            return "Bagus", "Wah, napasnya dalam dan kuat! Ini sinyal sehat banget, lanjutkan!"
        else:
            return "Normal", "Napas kamu stabil dan nyaman, tanda sistem pernapasan bekerja dengan baik."

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
