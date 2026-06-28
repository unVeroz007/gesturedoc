# app_streamlit.py - GestureDoc FINAL v3 (Fixed Thread Safety)
import cv2
import numpy as np
import threading
import json
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
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
def fetch_ai(zone_name):
    try:
        result = get_health_info(zone_name)
        with open(RESULT_FILE, "w") as f:
            json.dump(result, f)
    except Exception as e:
        print(f"AI Error: {e}")

# ── Video Processor ───────────────────────────────────────
class GestureDocProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker       = None
        self.hover_zone    = None
        self.hover_count   = 0
        self.current_zone  = None
        self.frame_counter = 0
        self.last_landmarks = {}
        self.last_tip      = None
        self.ai_fetching   = False
        self.lock          = threading.Lock()

    def recv(self, frame):
        # Inisialisasi tracker dengan aman di dalam thread
        if self.tracker is None:
            try:
                from hand_tracker import HandTracker
                self.tracker = HandTracker()
            except Exception as e:
                print(f"Gagal inisialisasi HandTracker: {e}")
                return frame

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        dh, dw = img.shape[:2]
        self.frame_counter += 1

        if self.frame_counter % PROCESS_EVERY_N == 0:
            try:
                small = cv2.resize(img, (RESIZE_W, RESIZE_H))
                self.tracker.process(small)
                tip = self.tracker.get_fingertip(small) if hasattr(self.tracker, 'get_fingertip') else None
                lms = self.tracker.get_body_landmarks(small) if hasattr(self.tracker, 'get_body_landmarks') else None
                
                with self.lock:
                    self.last_tip = tip
                    self.last_landmarks = lms
                
                zone_name, _ = get_nearest_zone(tip, lms)
                if zone_name:
                    if zone_name == self.hover_zone:
                        self.hover_count += 1
                    else:
                        self.hover_zone, self.hover_count = zone_name, 0
                    
                    if self.hover_count >= HOVER_NEEDED and not self.ai_fetching:
                        if zone_name != self.current_zone:
                            self.current_zone = zone_name
                            self.ai_fetching = True
                            threading.Thread(target=fetch_ai, args=(zone_name,), daemon=True).start()
                            self.ai_fetching = False
                        self.hover_count = 0
                else:
                    self.hover_zone, self.hover_count = None, 0
            except Exception as e:
                print(f"Tracking error: {e}")

        # Drawing
        if self.tracker and hasattr(self.tracker, 'draw_pose'):
            self.tracker.draw_pose(img)
            
        return frame.from_ndarray(img, format="bgr24")

# ── Main UI ───────────────────────────────────────────────
st.set_page_config(page_title="GestureDoc", layout="wide")
st.title("🩺 GestureDoc")

col_cam, col_info = st.columns([2, 1])

with col_cam:
    webrtc_streamer(
        key="gesturedoc",
        video_processor_factory=GestureDocProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("📋 Info Kesehatan")
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r") as f:
                st.write(json.load(f))
        except:
            st.info("Memuat data...")
    else:
        st.info("Arahkan jari telunjuk ke bagian tubuh...")

st_autorefresh(interval=2000, key="datarefresh")
