# 🎯🎯project Real-time Respiration & rPPG Monitoring: 🎯🎯

---

# DSP\_121450026\_121450087 🚀

## 🎯 Deskripsi Singkat

DSP\_121450026\_121450087 adalah aplikasi desktop berbasis Python yang mengintegrasikan pengolahan sinyal digital (Digital Signal Processing - DSP) dengan antarmuka grafis modern menggunakan PyQt5.
Dirancang khusus untuk memantau sinyal pernapasan dan detak jantung secara real-time melalui webcam dengan teknologi rPPG dan analisis posisi tubuh.

---

## ✨ Fitur Utama

* **Antarmuka GUI Elegan & User-Friendly:** Menggunakan PyQt5 dengan style Fusion yang sleek dan modern, cocok untuk pengguna awam maupun profesional.
* **Pemrosesan Sinyal Real-Time:** Memanfaatkan algoritma DSP untuk ekstraksi dan visualisasi sinyal pernapasan dan detak jantung secara simultan.
* **Teknologi MediaPipe:** Deteksi wajah dan pose tubuh untuk meningkatkan akurasi pengukuran.
* **Modular & Mudah Dikembangkan:** Kode terstruktur rapi memudahkan pemeliharaan dan pengembangan fitur baru.
* **Cross-Platform:** Mendukung Windows, Linux, dan macOS dengan sedikit konfigurasi.
* **Deploy Mudah:** Sudah teruji menggunakan PyInstaller untuk menghasilkan executable standalone.
* **Feedback Interaktif:** Informasi kualitas sinyal dan status pengukuran disajikan dengan bahasa yang mudah dimengerti dan sedikit humor agar lebih menyenangkan.
* **Fitur Restart Pengukuran:** Bisa mengulang pengukuran setelah durasi tertentu dengan mudah.

---

## 🛠️ Teknologi & Tools

* Python 3.12
* PyQt5 (Graphical User Interface)
* Numpy, Scipy (Numerical dan Scientific Computing)
* Matplotlib (Visualisasi Data)
* MediaPipe (Deteksi Wajah & Pose)
* PyInstaller (Packaging aplikasi)

---

## 🚀 Instalasi & Cara Menjalankan

### 1. Clone repository ini

```bash
git clone https://github.com/ericsonchandra99/Real-time-Respiration-and-rPPG-Monitoring-with-Python-PyQt5.git
```
```
cd Real-time-Respiration-and-rPPG-Monitoring-with-Python-PyQt5

```

### 2. (Opsional) Buat virtual environment agar instalasi bersih

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/macOS
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ *Catatan:* Instalasi paket seperti numpy, scipy, dan lainnya bisa memakan waktu tergantung koneksi internet dan spesifikasi komputer.

---

### 4. Jalankan aplikasi

```bash
python main.py
```

---

## 📦 Cara Membuat Executable (Windows)

1. Install PyInstaller jika belum ada:

```bash
pip install pyinstaller
```

2. Build aplikasi jadi `.exe`:

```bash
pyinstaller --onefile --windowed --paths=src main.py
```

3. File `main.exe` akan tersedia di folder `dist/`. Jalankan untuk membuka aplikasi.

---

## 📂 Struktur Proyek

```
DSP_121450026_121450087/
├── main.py                 # Entry point aplikasi
├── src/                    # Folder source code utama
│   ├── main_window.py      # UI utama aplikasi
│   └── video_processor.py  # Modul pengolahan video dan sinyal
├── .gitignore              # File pengaturan git ignore
├── requirements.txt        # Daftar dependency Python
└── README.md               # Dokumentasi proyek
```

---

## 🤝 Kontribusi

Kami sangat menghargai kontribusimu! Untuk berkontribusi:

1. Fork repository ini
2. Buat branch fitur baru:

```bash
git checkout -b fitur-baru
```

3. Commit perubahanmu:

```bash
git commit -m "Tambah fitur baru"
```

4. Push branch ke remote:

```bash
git push origin fitur-baru
```

5. Buat Pull Request di GitHub.

---

## 📜 Lisensi

MIT License © 2025 ericsonchandra99
MIT License © 2025 shulatalihta

---

## 📬 Kontak

Email: [sihombingericson@gmail.com](mailto:sihombingericson@gmail.com)

---

✨ Terima kasih telah mengunjungi proyek ini! Selamat mencoba dan semoga bermanfaat! ✨

---


