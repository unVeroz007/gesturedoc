# app_streamlit.py - GestureDoc FINAL v2
import cv2
import numpy as np
import threading
import json
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from hand_tracker import HandTracker
from body_zones import ZONE_COLORS, ZONE_LABELS, get_nearest_zone
from ai_engine import get_health_info
from streamlit_autorefresh import st_autorefresh

# ── Konfigurasi ───────────────────────────────────────────
HOVER_NEEDED    = 20
PROCESS_EVERY_N = 2
RESIZE_W        = 480
RESIZE_H        = 360
RESULT_FILE     = "gesturedoc_result.json"

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

ZONE_LABELS_CV = {
    "Kepala": "Kepala", "Mata Kiri": "Mata Kiri", "Mata Kanan": "Mata Kanan",
    "Hidung": "Hidung", "Mulut": "Mulut", "Telinga Kiri": "Telinga Kiri",
    "Telinga Kanan": "Telinga Kanan", "Bahu Kiri": "Bahu Kiri", "Bahu Kanan": "Bahu Kanan"
}

# ── Helper State ──────────────────────────────────────────
def clear_result():
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

def fetch_ai(zone_name):
    result = get_health_info(zone_name)
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f)

# ── Video Processor ───────────────────────────────────────
class GestureDocProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker       = None  # Inisialisasi nanti di recv()
        self.hover_zone    = None
        self.hover_count   = 0
        self.current_zone  = None
        self.frame_counter = 0
        self.last_landmarks = {}
        self.last_tip      = None
        self.ai_fetching   = False
        self.lock          = threading.Lock()

    def _draw_pose(self, img, sx, sy):
        self.tracker.draw_pose(img)

    def _draw_zones(self, img, lms, sx, sy):
        # Implementasi drawing zones disesuaikan dengan kebutuhan
        pass

    def _draw_fingertip(self, img, tip, sx, sy):
        if tip:
            cv2.circle(img, (int(tip[0]), int(tip[1])), 10, (255, 255, 0), -1)

    def _draw_hover_bar(self, img):
        if self.hover_zone and self.hover_count > 0:
            progress = int((self.hover_count / HOVER_NEEDED) * 200)
            cv2.rectangle(img, (20, 50), (20 + progress, 60), (0, 255, 0), -1)

    def recv(self, frame):
        if self.tracker is None:
            from hand_tracker import HandTracker
            self.tracker = HandTracker()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        dh, dw = img.shape[:2]
        self.frame_counter += 1

        if self.frame_counter % PROCESS_EVERY_N == 0:
            small = cv2.resize(img, (RESIZE_W, RESIZE_H))
            self.tracker.process(small)
            tip = self.tracker.get_fingertip(small) if hasattr(self.tracker, 'get_fingertip') else None
            lms = self.tracker.get_body_landmarks(small) if hasattr(self.tracker, 'get_body_landmarks') else None
            with self.lock:
                self.last_tip       = tip
                self.last_landmarks = lms
            zone_name, _ = get_nearest_zone(tip, lms)
            with self.lock:
                if zone_name:
                    if zone_name == self.hover_zone:
                        self.hover_count += 1
                    else:
                        self.hover_zone  = zone_name
                        self.hover_count = 0
                    if self.hover_count >= HOVER_NEEDED and not self.ai_fetching:
                        if zone_name != self.current_zone:
                            self.current_zone = zone_name
                            self.ai_fetching  = True
                            def run_ai(z):
                                fetch_ai(z)
                                self.ai_fetching = False
                            threading.Thread(target=run_ai, args=(zone_name,), daemon=True).start()
                        self.hover_count = 0
                else:
                    self.hover_zone  = None
                    self.hover_count = 0

        sx, sy = dw/RESIZE_W, dh/RESIZE_H
        self._draw_pose(img, sx, sy)
        if self.last_landmarks:
            self._draw_zones(img, self.last_landmarks, sx, sy)
        if self.last_tip:
            self._draw_fingertip(img, self.last_tip, sx, sy)
        self._draw_hover_bar(img)

        cv2.rectangle(img,(0,0),(dw,36),(15,15,25),-1)
        cv2.putText(img,"GestureDoc | Tunjuk bagian tubuhmu", (10,24),cv2.FONT_HERSHEY_SIMPLEX,0.50,(100,220,255),1)
        cv2.rectangle(img,(0,dh-22),(dw,dh),(15,15,25),-1)
        status = f"Aktif: {ZONE_LABELS_CV.get(self.current_zone,'-')}" if self.current_zone else "Belum ada zona aktif"
        cv2.putText(img,status,(10,dh-6),cv2.FONT_HERSHEY_SIMPLEX,0.40,(200,200,200),1)

        return frame.from_ndarray(img, format="bgr24")

# ── Main UI ───────────────────────────────────────────────
st.set_page_config(page_title="GestureDoc", layout="wide")
st.title("🩺 GestureDoc")

if 'current_zone' not in st.session_state: st.session_state.current_zone = None
if 'ai_text' not in st.session_state: st.session_state.ai_text = ""

col_cam, col_info = st.columns([2, 1])

with col_cam:
    ctx = webrtc_streamer(
        key="gesturedoc",
        video_processor_factory=GestureDocProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("📋 Info Kesehatan")
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "r") as f:
            data = json.load(f)
            st.write(data)
    else:
        st.info("Arahkan jari telunjuk ke bagian tubuh untuk memuat info...")

# Auto-refresh UI setiap 2 detik agar hasil AI muncul otomatis
st_autorefresh(interval=2000, key="datarefresh")
