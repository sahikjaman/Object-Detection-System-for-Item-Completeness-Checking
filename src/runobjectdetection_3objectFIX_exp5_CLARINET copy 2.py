import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import queue
import threading
from ultralytics import YOLO
import os
import csv
import pandas as pd
from datetime import datetime
import logging
import sys
import numpy as np

class AdvancedObjectDetectionSystem:
    def setup_logging(self):
        """Konfigurasi logging profesional"""
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f'object_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized successfully")
        logger.info(f"Log file created at: {log_filename}")
        return logger

    def log_message(self, message, level="info"):
        """Metode umum untuk logging dengan berbagai level"""
        try:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message)
            if hasattr(self, 'log_textbox'):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] {message}\n"
                if hasattr(self, 'root'):
                    self.root.after(0, self._update_log_textbox, log_entry)
        except Exception as e:
            print(f"Logging error: {e}")
            print(f"Original message: {message}")

    def _update_log_textbox(self, message):
        """Update log textbox dengan pesan baru"""
        try:
            if hasattr(self, 'log_textbox'):
                self.log_textbox.insert("end", message)
                self.log_textbox.see("end")
        except Exception as e:
            print(f"Error updating log textbox: {e}")

    def __init__(self, root):
        self.logger = self.setup_logging()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("Sistem Pengecekan Kelengkapan Barang")
        self.root.geometry("1200x800")
        # Mengizinkan resize
        self.root.resizable(True, True)
        self.root.minsize(1200, 800)  # Ukuran minimum agar sesuai gambar

        self.base_path = r"D:\DATA\FOR_DL\objectdetection\magang2025"
        self.model_path = os.path.join(self.base_path, "trainedclarinettes_yolov8n.pt")
        self.photo_dir = os.path.join(self.base_path, "clarinet_captured_photos")
        self.csv_dir = os.path.join(self.base_path, "clarinet_detection_logs")
        self.csv_path = os.path.join(self.csv_dir, "clarinet_detection_log.csv")
        self.report_dir = os.path.join(self.base_path, "reports")

        self.prepare_directories()
        self.configure_detection_settings()

        self.frame_queue = queue.Queue(maxsize=128)
        self.running = True
        self.detection_lock = threading.Lock()
        self.is_fullscreen = False

        self.detected_objects = {}
        self.current_frame = None
        self.detection_history = []
        self.detection_stats = {
            "total_detections": 0,
            "object_counts": {},
            "confidence_levels": {}
        }

        self.required_objects = {"Accessories Set", "Barcode", "Silica", "Strap", "Lower", "Mouthpiece", "Barrel", "Bell", "Upper"}
        self.max_allowed_objects = 9

        self.create_modern_ui()
        self.model = self.load_model()
        self.cap = self.init_video_capture()
        self.start_video_thread()
        self.add_control_features()

    def toggle_fullscreen(self):
        """Toggle antara mode fullscreen dan windowed"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)
        if not self.is_fullscreen:
            self.root.geometry("1200x800")  # Kembali ke ukuran default saat keluar fullscreen

    def show_captured_image(self, image_path, for_preview=False):
        """Menampilkan gambar yang telah dikapture, dengan opsi preview atau final"""
        if for_preview:
            self.preview_window = tk.Toplevel(self.root)
            self.preview_window.title("Preview Gambar Sebelum Submit")
            self.preview_window.geometry("700x500")
            self.preview_window.transient(self.root)
            self.preview_window.grab_set()

            img = Image.open(image_path)
            img = img.resize((600, 400), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            label = tk.Label(self.preview_window, image=img_tk)
            label.image = img_tk
            label.pack(pady=10)

            button_frame = ctk.CTkFrame(self.preview_window)
            button_frame.pack(pady=10)

            submit_btn = ctk.CTkButton(button_frame, text="Submit", command=lambda: self.submit_preview(image_path), 
                                      fg_color="#2ecc71", hover_color="#27ae60")
            submit_btn.pack(side="left", padx=5)

            cancel_btn = ctk.CTkButton(button_frame, text="Batal", command=self.preview_window.destroy, 
                                      fg_color="#e74c3c", hover_color="#c0392b")
            cancel_btn.pack(side="left", padx=5)
        else:
            self.image_window = tk.Toplevel(self.root)
            self.image_window.title("Gambar yang Ditangkap")
            img = Image.open(image_path)
            img = img.resize((600, 400), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            label = tk.Label(self.image_window, image=img_tk)
            label.image = img_tk
            label.pack()

            back_button = tk.Button(self.image_window, text="Kembali ke Deteksi Objek", command=self.image_window.destroy)
            back_button.pack(pady=10)

    def submit_preview(self, image_path):
        """Proses submit setelah preview dikonfirmasi"""
        with self.detection_lock:
            detected_objects = set(self.detected_objects.keys())
        if detected_objects == self.required_objects:
            timestamp = datetime.now()
            filename = os.path.join(self.photo_dir, f"detected_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
            os.rename(image_path, filename)
            self.save_detection_data(timestamp, detected_objects, filename)
            success_msg = f"Foto dan data berhasil disimpan: {filename}"
            messagebox.showinfo("Sukses", success_msg)
            self.log_message(success_msg)
            self.preview_window.destroy()
            self.update_capture_display(filename)
        else:
            messagebox.showwarning("Peringatan", "Objek belum lengkap untuk submit.")

    def update_capture_display(self, image_path):
        """Perbarui tampilan hasil tangkapan gambar di kolom kanan (640x360)"""
        # Resize gambar ke 640x360 untuk ukuran normal
        img = Image.open(image_path)
        img_resized = img.resize((640, 360), Image.LANCZOS)  # Ukuran normal 16:9
        img_tk = ImageTk.PhotoImage(img_resized)
        self.capture_label.configure(image=img_tk)
        self.capture_label.image = img_tk

    def open_csv_file(self):
        """Membuka file CSV dengan aplikasi default sistem"""
        try:
            if os.path.exists(self.csv_path):
                os.startfile(self.csv_path)
                self.log_message(f"File CSV dibuka: {self.csv_path}")
            else:
                self.log_message(f"File CSV tidak ditemukan: {self.csv_path}", level="error")
                messagebox.showerror("File Tidak Ditemukan", "File CSV tidak ditemukan.")
        except Exception as e:
            self.log_message(f"Gagal membuka file CSV: {e}", level="error")
            messagebox.showerror("Error", f"Gagal membuka file CSV: {e}")

    def log_detection(self, timestamp, object_names, filename=None):
        """Logging deteksi ke CSV dengan format khusus (hanya saat submit)"""
        try:
            detection_data = [
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f'"{object_names}"'
            ]
            with open(self.csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(detection_data)
            self.log_message(f"Data berhasil dicatat di {self.csv_path}")
        except Exception as e:
            error_msg = f"Gagal mencatat data CSV: {e}"
            self.log_message(error_msg, level="error")

    def save_detection_data(self, timestamp, detected_objects, image_path):
        """Simpan data kelengkapan ke file CSV dengan format 1/0 dan gambar sebagai .jpg (640x360)"""
        # Resize gambar ke 640x360 sebelum menyimpan
        img = cv2.imread(image_path)
        if img is not None:
            resized_img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, resized_img)  # Timpa file dengan resolusi baru
            self.log_message(f"Foto disimpan dengan resolusi 640x360: {image_path}")

        # Buat nama set berdasarkan timestamp
        set_name = f"Set_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        # Buat data kelengkapan (1 untuk ada, 0 untuk tidak ada)
        data = {'Set': [set_name]}
        for obj in self.required_objects:
            data[obj] = [1 if obj in detected_objects else 0]

        # Simpan ke CSV
        df = pd.DataFrame(data)
        os.makedirs(self.csv_dir, exist_ok=True)
        csv_filename = os.path.join(self.csv_dir, f"{set_name}.csv")
        df.to_csv(csv_filename, index=False)
        self.log_message(f"Data kelengkapan disimpan ke {csv_filename}")

        # Log ke CSV utama hanya saat submit
        self.log_detection(timestamp, ", ".join(detected_objects), image_path)

    def prepare_directories(self):
        """Persiapan direktori dengan logging"""
        directories = [self.photo_dir, self.base_path, self.report_dir, self.csv_dir]
        for dir_path in directories:
            try:
                os.makedirs(dir_path, exist_ok=True)
                self.log_message(f"Direktori berhasil dibuat: {dir_path}")
            except Exception as e:
                self.log_message(f"Gagal membuat direktori {dir_path}: {e}", level="error")
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['Timestamp', 'Daftar Objek'])
                self.log_message(f"File CSV baru dibuat: {self.csv_path}")
            except Exception as e:
                self.log_message(f"Gagal membuat file CSV: {e}", level="error")

    def configure_detection_settings(self):
        """Konfigurasi pengaturan deteksi lanjutan"""
        self.detection_config = {
            "confidence_threshold": 0.3,
            "nms_threshold": 0.5,
            "tracking_enabled": False,
            "alert_mode": False,
            "min_detection_area": 100
        }

    def load_model(self):
        """Load model dengan error handling komprehensif"""
        try:
            model = YOLO(self.model_path)
            self.log_message(f"Model berhasil dimuat dari {self.model_path}")
            return model
        except Exception as e:
            error_msg = f"Gagal memuat model: {str(e)}"
            self.log_message(error_msg, level="error")
            messagebox.showerror("Model Error", error_msg)
            self.root.quit()

    def init_video_capture(self):
        """Inisialisasi capture video dengan error handling untuk resolusi normal (640x360)"""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise IOError("Tidak dapat membuka kamera")
            # Atur resolusi normal 640x360 (16:9)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 30)
            self.log_message("Video capture berhasil diinisialisasi dengan resolusi 640x360")
            return cap
        except Exception as e:
            error_msg = f"Gagal menginisialisasi video capture: {e}"
            self.log_message(error_msg, level="error")
            messagebox.showerror("Kamera Error", error_msg)
            self.root.quit()

    def start_video_thread(self):
        """Memulai thread video capture dan deteksi"""
        capture_thread = threading.Thread(target=self.video_capture_thread, daemon=True)
        capture_thread.start()
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()

    def video_capture_thread(self):
        """Thread untuk capture video dengan resolusi 640x360"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("Gagal membaca frame dari kamera", level="warning")
                    break
                self.current_frame = frame.copy()
                # Tidak perlu resize lagi karena sudah 640x360
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            except Exception as e:
                self.log_message(f"Error di video capture thread: {e}", level="error")
                break

    def detection_thread(self):
        """Thread deteksi objek dengan validasi ketat"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                # Gunakan frame asli 640x360 untuk deteksi
                results = self.model(frame)
                filtered_objects = self.advanced_object_filtering(results)
                self.detected_objects = {obj["name"]: obj for obj in filtered_objects}
                self.root.after(0, self.update_detection_status)
                display_frame = frame.copy()  # Gunakan frame asli untuk tampilan (640x360)
                for obj in filtered_objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    name = obj["name"]
                    conf = obj["confidence"]
                    color_map = {
                        "Accessories Set": (255, 0, 0),
                        "Barcode": (0, 255, 0),
                        "Silica": (0, 0, 255),
                        "Strap": (255, 165, 0),
                        "Lower": (128, 0, 128),
                        "Mouthpiece": (255, 255, 0),
                        "Barrel": (0, 255, 255),
                        "Bell": (255, 0, 255),
                        "Upper": (75, 0, 130)
                    }
                    color = color_map.get(name, (255, 255, 255))
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                self.display_frame(display_frame)
            except queue.Empty:
                continue
            except Exception as e:
                self.log_message(f"Error di detection thread: {e}", level="error")

    def advanced_object_filtering(self, results):
        """Filtering objek dengan kriteria lanjutan"""
        filtered_objects = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                object_name = self.model.names[class_id]
                if conf > self.detection_config["confidence_threshold"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > self.detection_config["min_detection_area"]:
                        filtered_objects.append({
                            "name": object_name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2),
                            "area": area
                        })
        return filtered_objects

    def update_detection_stats(self, detected_objects):
        """Perbarui statistik deteksi"""
        self.detection_stats["total_detections"] += 1
        self.detection_stats["object_counts"].clear()
        self.detection_stats["confidence_levels"].clear()
        for obj in detected_objects:
            name = obj["name"]
            conf = obj["confidence"]
            self.detection_stats["object_counts"][name] = self.detection_stats["object_counts"].get(name, 0) + 1
            if name not in self.detection_stats["confidence_levels"]:
                self.detection_stats["confidence_levels"][name] = conf
            else:
                self.detection_stats["confidence_levels"][name] = (self.detection_stats["confidence_levels"][name] + conf) / 2

    def update_detection_status(self):
        """Update status deteksi di UI (tanpa logging ke CSV secara terus menerus)"""
        with self.detection_lock:
            detected = set(self.detected_objects.keys())
        status = "Lengkap" if self.required_objects.issubset(detected) else "NG"
        if status == "Lengkap":
            self.status_label.configure(text=f"Status: {status}", text_color="green")
        else:
            self.status_label.configure(text=f"Status: {status}", text_color="red")
        items_text = "List Barang:\n"
        for obj in self.required_objects:
            status_text = "ada" if obj in detected else "tidak ada"
            items_text += f"- {obj}: {status_text}\n"
        self.items_list.delete("1.0", ctk.END)
        self.items_list.insert("1.0", items_text)
        # Menghapus log ke CSV yang otomatis di sini

    def display_frame(self, frame):
        """Tampilkan frame di GUI (sesuai ukuran layar, minimal 640x360)"""
        # Hitung ukuran layar saat ini
        screen_width = self.root.winfo_width()
        screen_height = self.root.winfo_height()
        aspect_ratio = 16 / 9  # Rasio 16:9

        # Tentukan ukuran minimum berdasarkan program (640x360 untuk rasio 16:9)
        min_width = 640
        min_height = 360

        # Tentukan ukuran maksimum berdasarkan layar (menjaga rasio 16:9)
        max_width = int(screen_width * 0.33)  # 1/3 lebar layar untuk kolom kamera
        max_height = int(max_width / aspect_ratio)  # Hitung tinggi berdasarkan rasio

        # Gunakan ukuran minimum jika layar lebih kecil dari ukuran minimum
        display_width = max(min_width, min(max_width, 640))
        display_height = max(min_height, min(max_height, 360))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_resized = img.resize((display_width, display_height), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_resized)
        self.root.after(10, self.update_video_label, imgtk)

    def update_video_label(self, imgtk):
        """Update label video"""
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

    def create_modern_ui(self):
        """Membuat antarmuka pengguna modern dengan CustomTkinter sesuai layout gambar"""
        # Frame utama menggunakan grid
        main_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b")
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Konfigurasi grid utama (responsif)
        main_frame.grid_columnconfigure((0, 1, 2), weight=1, minsize=400)  # Setiap kolom minimal 400px
        main_frame.grid_rowconfigure((0, 1, 2), weight=1, minsize=200)  # Setiap baris minimal 200px

        # Kolom Kiri: Tampilan Kamera Realtime
        camera_frame = ctk.CTkFrame(main_frame, fg_color="#3a3a3a", border_width=1, border_color="black")
        camera_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(camera_frame, text="Tampilan Kamera Realtime", font=("Helvetica", 14, "bold"), 
                     text_color="white").pack(pady=5)
        self.video_label = ctk.CTkLabel(camera_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Kolom Tengah: Cek Kelengkapan
        check_frame = ctk.CTkFrame(main_frame, fg_color="#3a3a3a", border_width=1, border_color="black")
        check_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(check_frame, text="Cek Kelengkapan", font=("Helvetica", 20, "bold"), 
                     text_color="white").pack(pady=10)

        # Status
        self.status_label = ctk.CTkLabel(check_frame, text="Status: NG", text_color="red", 
                                       font=("Helvetica", 16, "bold"))
        self.status_label.pack(pady=5)

        # List Barang
        self.items_list = ctk.CTkTextbox(check_frame, height=300, width=200, fg_color="#4a4a4a", 
                                        text_color="white")
        self.items_list.pack(expand=True, fill="both", padx=5, pady=5)

        # Kolom Kanan: Hasil Tangkapan Gambar
        capture_frame = ctk.CTkFrame(main_frame, fg_color="#3a3a3a", border_width=1, border_color="black")
        capture_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(capture_frame, text="Hasil Tangkapan Gambar", font=("Helvetica", 14, "bold"), 
                     text_color="white").pack(pady=5)
        self.capture_label = ctk.CTkLabel(capture_frame, text="")
        self.capture_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Frame untuk log (di bawah kolom tengah)
        self.log_frame = ctk.CTkFrame(main_frame, fg_color="#3a3a3a", border_width=1, border_color="black")
        self.log_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.log_textbox = ctk.CTkTextbox(self.log_frame, height=150, fg_color="#4a4a4a", text_color="white")
        self.log_textbox.pack(expand=True, fill="both", padx=5, pady=5)

        # Tombol di bagian bawah, termasuk toggle fullscreen
        button_frame = ctk.CTkFrame(main_frame, fg_color="#2b2b2b", border_width=1, border_color="black")
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.quit_btn = ctk.CTkButton(button_frame, text="Quit", command=self.quit_app, 
                                     fg_color="#e74c3c", hover_color="#c0392b")
        self.quit_btn.grid(row=0, column=0, padx=5, pady=5)

        self.open_csv_btn = ctk.CTkButton(button_frame, text="Lihat CSV", command=self.open_csv_file, 
                                         fg_color="#3498db", hover_color="#2980b9")
        self.open_csv_btn.grid(row=0, column=1, padx=5, pady=5)

        self.capture_btn = ctk.CTkButton(button_frame, text="Submit", command=self.submit_data, 
                                        fg_color="#2ecc71", hover_color="#27ae60")
        self.capture_btn.grid(row=0, column=2, padx=5, pady=5)

        # Tombol Toggle Fullscreen
        self.fullscreen_btn = ctk.CTkButton(button_frame, text="Toggle Fullscreen", command=self.toggle_fullscreen,
                                           fg_color="#3498db", hover_color="#2980b9")
        self.fullscreen_btn.grid(row=0, column=3, padx=5, pady=5)

    def submit_data(self):
        """Mengambil foto dan menyimpan data kelengkapan ke .jpg dan .csv"""
        with self.detection_lock:
            detected_objects = set(self.detected_objects.keys())

        if detected_objects == self.required_objects:
            timestamp = datetime.now()
            # Simpan gambar sebagai .jpg (640x360)
            image_filename = os.path.join(self.photo_dir, f"detected_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
            if self.current_frame is not None:
                try:
                    cv2.imwrite(image_filename, self.current_frame)
                    self.log_message(f"Foto berhasil disimpan: {image_filename}")
                    self.update_capture_display(image_filename)  # Tampilkan di "Hasil Tangkapan Gambar"
                except Exception as e:
                    error_msg = f"Gagal menyimpan foto: {e}"
                    messagebox.showerror("Error", error_msg)
                    self.log_message(error_msg, level="error")

            # Simpan data kelengkapan ke .csv
            self.save_detection_data(timestamp, detected_objects, image_filename)
            success_msg = "Data dan foto berhasil disimpan."
            messagebox.showinfo("Sukses", success_msg)
            self.log_message(success_msg)
        else:
            missing = self.required_objects - detected_objects
            warning_msg = f"Objek belum lengkap. Kurang: {', '.join(missing)}" if missing else "Harus lengkap 9 objek untuk submit."
            messagebox.showwarning("Peringatan", warning_msg)
            self.log_message(warning_msg, level="warning")

    def add_control_features(self):
        """Tambahkan fitur kontrol tambahan (diletakkan di frame log)"""
        control_frame = ctk.CTkFrame(self.log_frame, fg_color="#3a3a3a")
        control_frame.pack(fill="x", padx=5, pady=5)

        self.confidence_label = ctk.CTkLabel(
            control_frame,
            text="Confidence Threshold",
            text_color="white"
        )
        self.confidence_label.pack(side="left", padx=5)

        self.confidence_slider = ctk.CTkSlider(
            control_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            command=self.update_confidence_threshold
        )
        self.confidence_slider.set(self.detection_config["confidence_threshold"])
        self.confidence_slider.pack(side="left", padx=5)

        self.tracking_var = ctk.BooleanVar(value=False)
        self.tracking_switch = ctk.CTkSwitch(
            control_frame,
            text="Object Tracking",
            variable=self.tracking_var,
            command=self.toggle_tracking
        )
        self.tracking_switch.pack(side="left", padx=5)

    def toggle_tracking(self):
        """Toggle object tracking mode"""
        tracking_enabled = self.tracking_var.get()
        self.detection_config["tracking_enabled"] = tracking_enabled
        tracking_status = "enabled" if tracking_enabled else "disabled"
        self.log_message(f"Object tracking {tracking_status}")

    def update_confidence_threshold(self, value):
        """Update confidence threshold dynamically"""
        self.detection_config["confidence_threshold"] = value
        self.log_message(f"Confidence threshold updated to {value:.2f}")

    def quit_app(self):
        """Keluar dari aplikasi"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.quit()

def main():
    root = ctk.CTk()
    app = AdvancedObjectDetectionSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()