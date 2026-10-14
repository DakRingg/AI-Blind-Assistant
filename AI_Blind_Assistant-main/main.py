import os
import time
import threading
import queue
import math
from turtle import left, right
import requests
from collections import deque, Counter
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from gtts import gTTS
import pygame
from ultralytics import YOLO

# ==============================================================
# 1. TRỢ LÝ GIỌNG NÓI TIẾNG VIỆT 
# ==============================================================
class VoiceAssistant:
    def __init__(self, cooldown=3.5):
        self.cooldown = cooldown
        self.last_speech_time = 0
        self.last_message = ""
        self.msg_queue = queue.Queue()
        self.is_running = True
        
        pygame.mixer.init()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.is_running:
            try:
                text = self.msg_queue.get(timeout=0.2)
                if not self.is_running:
                    break
                
                tts = gTTS(text=text, lang='vi', slow=False)
                temp_file = f"temp_voice_{int(time.time()*1000)}.mp3"
                tts.save(temp_file)
                
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.is_running:
                    time.sleep(0.05)
                
                pygame.mixer.music.unload()
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                        
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Lỗi Voice]: {e}")

    def speak(self, text):
        current_time = time.time()
        if (text != self.last_message) and (current_time - self.last_speech_time > self.cooldown):
            self.last_message = text
            self.last_speech_time = current_time
            
            while not self.msg_queue.empty():
                try:
                    self.msg_queue.get_nowait()
                except queue.Empty:
                    break
            self.msg_queue.put(text)

    def stop(self):
        self.is_running = False
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass


# ==============================================================
# 2. BỘ ĐỊNH TUYẾN GPS (LOCATION NAVIGATOR)
# ==============================================================
class LocationNavigator:
    def __init__(self, voice_assistant):
        self.voice = voice_assistant
        self.current_coords = None  # Tuple (longitude, latitude)
        self.route_steps = []
        self.current_step_index = 0
        self.is_navigating = False

    def start_navigation(self, start_lon, start_lat, dest_lon, dest_lat):
        self.is_navigating = True
        self.voice.speak("Đang tính toán tuyến đường.")
        
        # API OSRM dành cho người đi bộ (yêu cầu Kinh độ, Vĩ độ)
        url = f"http://router.project-osrm.org/route/v1/foot/{start_lon},{start_lat};{dest_lon},{dest_lat}?steps=true&overview=false"
        
        try:
            response = requests.get(url).json()
            if response.get("code") == "Ok":
                legs = response["routes"][0]["legs"][0]
                self.route_steps = legs["steps"]
                self.current_step_index = 0
                self.update_current_location(start_lon, start_lat) 
                
                self.voice.speak("Đã tìm thấy tuyến đường. Bắt đầu di chuyển.")
                threading.Thread(target=self._tracking_worker, daemon=True).start()
            else:
                self.voice.speak("Không thể tìm thấy tuyến đường.")
                self.is_navigating = False
        except Exception as e:
            print(f"[Lỗi API Navigation]: {e}")
            self.voice.speak("Lỗi kết nối bản đồ.")
            self.is_navigating = False

    def update_current_location(self, lon, lat):
        self.current_coords = (lon, lat)

    def _tracking_worker(self):
        while self.is_navigating and self.current_step_index < len(self.route_steps):
            if not self.current_coords:
                time.sleep(1)
                continue

            step = self.route_steps[self.current_step_index]
            step_lon, step_lat = step["maneuver"]["location"] 
            instruction = step["maneuver"].get("modifier", "straight")
            distance = step["distance"] 
            
            dist_to_turn = self._calculate_distance(
                self.current_coords[1], self.current_coords[0], 
                step_lat, step_lon
            )

            if dist_to_turn <= 10.0 and distance > 0:
                vi_instruction = self._translate_maneuver(instruction)
                self.voice.speak(vi_instruction)
                self.current_step_index += 1
            
            time.sleep(2) 
            
        if self.is_navigating and self.current_step_index >= len(self.route_steps):
            self.voice.speak("Bạn đã đến nơi.")
            self.is_navigating = False

    def _translate_maneuver(self, modifier):
        if modifier in ["left", "sharp left", "slight left"]:
            return "Sắp tới, chuẩn bị rẽ trái."
        elif modifier in ["right", "sharp right", "slight right"]:
            return "Sắp tới, chuẩn bị rẽ phải."
        else:
            return "Tiếp tục đi thẳng theo đường hiện tại."

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000 
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ==============================================================
# 3. LOGIC PHÂN TÍCH & BỘ LỌC CHỐNG RUNG VẬT CẢN
# ==============================================================
class NavigationEngine:
    def __init__(self):
        self.history = deque(maxlen=12)

    def analyze_frame(self, detections, frame_width, frame_height):
        left_boundary = frame_width / 3
        right_boundary = 2 * frame_width / 3
        zone_threats = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        close_objects = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            cx = (x1 + x2) / 2
            h = y2 - y1
            w = x2 - x1
            area_ratio = (w * h) / (frame_width * frame_height)

            if area_ratio > 0.22 or (h / frame_height) > 0.55:
                dist = "VERY_CLOSE"
                threat = 3.0
            elif area_ratio > 0.08 or (h / frame_height) > 0.30:
                dist = "CLOSE"
                threat = 1.5
            else:
                dist = "FAR"
                threat = 0.5

            if cx < left_boundary:
                zone = "LEFT"
            elif cx > right_boundary:
                zone = "RIGHT"
            else:
                zone = "CENTER"

            zone_threats[zone] += threat

            if dist in ["VERY_CLOSE", "CLOSE"]:
                close_objects.append({"label": label, "zone": zone, "distance": dist})

        center = zone_threats["CENTER"]
        left = zone_threats["LEFT"]
        right = zone_threats["RIGHT"]

        if center >= 1.5:
        # Nếu giữa bị chặn, ưu tiên rẽ sang bên có rủi ro thấp hơn
            if left <= right and left < 2.0:
                raw_cmd = "TURN_LEFT"
            elif right < left and right < 2.0:
                raw_cmd = "TURN_RIGHT"
            else:
                raw_cmd = "STOP"
        else:
        # Nếu 2 bên có rủi ro chênh lệch rõ rệt -> Né sang hướng an toàn hơn
            if left > right and left >= 1.5:
                raw_cmd = "SHIFT_RIGHT"
            elif right > left and right >= 1.5:
                raw_cmd = "SHIFT_LEFT"
            else:
                raw_cmd = "GO_STRAIGHT"

        self.history.append(raw_cmd)
        most_common_cmd, count = Counter(self.history).most_common(1)[0]

        instructions = {
            "GO_STRAIGHT": "Phía trước an toàn.",
            "TURN_LEFT": "Vật cản phía trước, bước sang trái.",
            "TURN_RIGHT": "Vật cản phía trước, bước sang phải.",
            "SHIFT_LEFT": "Chú ý, hãy đi chếch sang trái.",
            "SHIFT_RIGHT": "Chú ý, hãy đi chếch sang phải.",
            "STOP": "Đường bị chặn hoàn toàn! Dừng lại."
        }

        alert_level = "CRITICAL" if most_common_cmd == "STOP" else ("WARNING" if "TURN" in most_common_cmd else "SAFE")

        return {
            "instruction": instructions[most_common_cmd],
            "direction": most_common_cmd,
            "alert_level": alert_level,
            "close_objects": close_objects
        }


# ==============================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================
class BlindAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Blind Assistant - Trợ Lý Dẫn Đường & Tránh Vật Cản")
        self.root.geometry("1100x750")
        self.root.configure(bg="#121212")

        self.is_running = True
        self.voice = VoiceAssistant(cooldown=3.5)
        self.nav_engine = NavigationEngine()
        self.gps_navigator = LocationNavigator(self.voice) 

        self.model_path = os.path.join("weights", "best.pt")
        if not os.path.exists(self.model_path):
            self.model_path = "yolov8n.pt"
        self.model = YOLO(self.model_path)

        self.video_source = 2
        self.cap = cv2.VideoCapture(self.video_source)
        
        self.voice_enabled = tk.BooleanVar(value=True)
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_nav_result = None

        self._build_ui()

        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('q', lambda e: self.on_closing())

        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()

        self._update_gui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1F1F1F", height=50)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header, text="HỆ THỐNG TRỢ LÝ HỖ TRỢ NGƯỜI KHIẾM THỊ",
            font=("Helvetica", 13, "bold"), fg="#00E5FF", bg="#1F1F1F"
        )
        title_lbl.pack(pady=10)

        main_container = tk.Frame(self.root, bg="#121212")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.video_frame = tk.Label(main_container, bg="#000000", bd=2, relief=tk.RIDGE)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(main_container, bg="#1E1E1E", width=360)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="HƯỚNG ĐI ĐỀ XUẤT (VẬT CẢN)", font=("Helvetica", 10, "bold"), fg="#AAAAAA", bg="#1E1E1E").pack(pady=(10, 5))
        self.dir_icon_lbl = tk.Label(sidebar, text="↑", font=("Helvetica", 36, "bold"), fg="#00E676", bg="#1E1E1E")
        self.dir_icon_lbl.pack()
        self.dir_text_lbl = tk.Label(sidebar, text="ĐI THẲNG", font=("Helvetica", 14, "bold"), fg="#00E676", bg="#1E1E1E")
        self.dir_text_lbl.pack(pady=(0, 5))

        self.instruction_box = tk.Label(
            sidebar, text="Đang nhận dạng môi trường...", font=("Helvetica", 11),
            fg="#FFFFFF", bg="#2A2A2A", wraplength=320, height=2, relief=tk.GROOVE
        )
        self.instruction_box.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(sidebar, text="DẪN ĐƯỜNG (BẢN ĐỒ OSRM)", font=("Helvetica", 10, "bold"), fg="#00E5FF", bg="#1E1E1E").pack(pady=(15, 5))
        
        # Mặc định nhập sẵn một điểm đích gần vị trí của bạn để test (Chuẩn: Vĩ độ, Kinh độ)
        self.dest_entry = tk.Entry(sidebar, font=("Helvetica", 11), bg="#2A2A2A", fg="#FFFFFF", insertbackground="white")
        self.dest_entry.insert(0, "15.975000, 108.251000") 
        self.dest_entry.pack(fill=tk.X, padx=10, pady=5)
        
        gps_btn = tk.Button(sidebar, text="📍 Bắt đầu đi tới tọa độ này", font=("Helvetica", 10, "bold"), bg="#0277BD", fg="#FFFFFF", command=self.handle_start_gps)
        gps_btn.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(sidebar, text="VẬT CẢN PHÁT HIỆN", font=("Helvetica", 10, "bold"), fg="#AAAAAA", bg="#1E1E1E").pack(pady=(15, 5))
        self.obj_listbox = tk.Listbox(sidebar, bg="#2A2A2A", fg="#00E676", font=("Consolas", 10), height=4, bd=0)
        self.obj_listbox.pack(fill=tk.X, padx=10, pady=5)

        video_btn = tk.Button(sidebar, text="📁 Chọn Video Test", bg="#37474F", fg="#FFFFFF", command=self.load_test_video)
        video_btn.pack(fill=tk.X, padx=15, pady=(10, 5))

        cam_btn = tk.Button(sidebar, text="📷 Quay Lại Camera", bg="#263238", fg="#B0BEC5", command=self.switch_to_camera)
        cam_btn.pack(fill=tk.X, padx=15, pady=(0, 10))

        voice_chk = tk.Checkbutton(sidebar, text="Bật Giọng Nói Tiếng Việt", variable=self.voice_enabled, fg="#FFFFFF", bg="#1E1E1E", selectcolor="#2A2A2A")
        voice_chk.pack(pady=5)

        quit_btn = tk.Button(sidebar, text="THOÁT (Esc)", bg="#D50000", fg="#FFFFFF", font=("Helvetica", 11, "bold"), command=self.on_closing)
        quit_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

    def handle_start_gps(self):
        dest_txt = self.dest_entry.get().strip()
        try:
            # Nhận chuẩn Google Maps: Vĩ độ (Lat), Kinh độ (Lon)
            dest_lat, dest_lon = map(float, dest_txt.split(","))
            
            # Vị trí hiện tại của bạn đã được cập nhật
            current_lat = 15.972224
            current_lon = 108.249742
            
            # Hàm start_navigation nhận vào theo chuẩn OSRM (Kinh độ, Vĩ độ)
            self.gps_navigator.start_navigation(current_lon, current_lat, dest_lon, dest_lat)
        except Exception:
            messagebox.showerror("Lỗi", "Vui lòng nhập tọa độ đích đúng định dạng: VĩĐộ, KinhĐộ (vd: 15.975000, 108.251000)")

    def load_test_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_source = file_path
            if self.cap.isOpened(): self.cap.release()
            self.cap = cv2.VideoCapture(self.video_source)

    def switch_to_camera(self):
        self.video_source = 0
        if self.cap.isOpened(): self.cap.release()
        self.cap = cv2.VideoCapture(0)

    def _draw_hud(self, frame, nav_result):
        h, w, _ = frame.shape
        l_bound = int(w / 3)
        r_bound = int(2 * w / 3)

        cv2.line(frame, (l_bound, 0), (l_bound, h), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(frame, (r_bound, 0), (r_bound, h), (80, 80, 80), 1, cv2.LINE_AA)

        if nav_result:
            cmd = nav_result["direction"]
            alert = nav_result["alert_level"]
            
            # Xác định vùng sẽ highlight dựa trên hướng đi được đề xuất
            if cmd in ["TURN_LEFT", "SHIFT_LEFT"]:
                x_start, x_end = 0, l_bound          # Cột ngoài cùng bên trái
            elif cmd in ["TURN_RIGHT", "SHIFT_RIGHT"]:
                x_start, x_end = r_bound, w          # Cột ngoài cùng bên phải
            else: 
                # Nếu là STOP hoặc GO_STRAIGHT thì highlight cột giữa
                x_start, x_end = l_bound, r_bound    

            if alert == "CRITICAL":
                cv2.rectangle(frame, (x_start, 0), (x_end, h), (0, 0, 255), 6) # Màu đỏ, nét dày 6
            elif alert == "WARNING":
                cv2.rectangle(frame, (x_start, 0), (x_end, h), (0, 255, 255), 6) # Màu vàng, nét dày 6

        if self.gps_navigator.is_navigating:
            cv2.putText(frame, "[GPS] Dang Dan Duong", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame

    def _ai_worker(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            if self.video_source == 0:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            results = self.model(frame, verbose=False, conf=0.35)[0]
            detections = []

            for r in results.boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cls_id = int(r.cls[0])
                label = self.model.names[cls_id]
                conf = float(r.conf[0])

                if label in ["Person", "person", "Người"] and conf < 0.60:
                    continue

                box_area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                if box_area_ratio > 0.85 and conf < 0.75:
                    continue

                detections.append({"box": [x1, y1, x2, y2], "label": label, "conf": conf})

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            nav_result = self.nav_engine.analyze_frame(detections, w, h)
            frame = self._draw_hud(frame, nav_result)

            if self.voice_enabled.get() and nav_result["direction"] != "GO_STRAIGHT":
                self.voice.speak(nav_result["instruction"])

            self.latest_nav_result = nav_result
            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put(frame)

    def _update_gui(self):
        if not self.is_running: return

        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((700, 525), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

            if self.latest_nav_result:
                self._update_sidebar(self.latest_nav_result)
        except queue.Empty:
            pass

        if self.is_running:
            self.root.after(20, self._update_gui)

    def _update_sidebar(self, nav):
        self.instruction_box.config(text=nav["instruction"])

        cmd = nav["direction"]
        if cmd == "GO_STRAIGHT":
            self.dir_icon_lbl.config(text="↑", fg="#00E676")
            self.dir_text_lbl.config(text="ĐI THẲNG", fg="#00E676")
        elif cmd in ["TURN_LEFT", "SHIFT_LEFT"]:
            self.dir_icon_lbl.config(text="←", fg="#FFD600")
            self.dir_text_lbl.config(text="RẼ TRÁI", fg="#FFD600")
        elif cmd in ["TURN_RIGHT", "SHIFT_RIGHT"]:
            self.dir_icon_lbl.config(text="→", fg="#FFD600")
            self.dir_text_lbl.config(text="RẼ PHẢI", fg="#FFD600")
        elif cmd == "STOP":
            self.dir_icon_lbl.config(text="✖", fg="#FF1744")
            self.dir_text_lbl.config(text="DỪNG LẠI", fg="#FF1744")

        self.obj_listbox.delete(0, tk.END)
        for obj in nav["close_objects"]:
            zone_vn = {"LEFT": "Trái", "CENTER": "Giữa", "RIGHT": "Phải"}.get(obj["zone"])
            dist_vn = {"VERY_CLOSE": "RẤT GẦN", "CLOSE": "Gần", "FAR": "Xa"}.get(obj["distance"])
            self.obj_listbox.insert(tk.END, f"• {obj['label']} [{zone_vn} - {dist_vn}]")

    def on_closing(self):
        self.is_running = False
        try: self.voice.stop()
        except Exception: pass
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception: pass
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlindAssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()