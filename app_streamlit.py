# app_streamlit.py - GestureDoc FINAL v2

import cv2
import numpy as np
import threading
import queue
import json
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from hand_tracker import HandTracker
from body_zones import ZONE_COLORS, ZONE_LABELS, get_nearest_zone
from ai_engine import get_health_info
from streamlit_autorefresh import st_autorefresh
import requests
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["EGL_PLATFORM"] = "surfaceless"

# ── Konfigurasi ───────────────────────────────────────────
HOVER_NEEDED    = 20
PROCESS_EVERY_N = 2
RESIZE_W        = 480
RESIZE_H        = 360
RESULT_FILE     = "gesturedoc_result.json"   # ← simpan hasil AI ke file


RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        {"urls": "stun:stun3.l.google.com:19302"},
    ]
})

ZONE_LABELS_CV = {
    "Kepala": "Kepala", "Mata Kiri": "Mata Kiri", "Mata Kanan": "Mata Kanan",
    "Hidung": "Hidung", "Mulut": "Mulut", "Telinga Kiri": "Telinga Kiri",
    "Telinga Kanan": "Telinga Kanan", "Dagu": "Dagu", "Leher": "Leher",
    "Dada & Paru-paru": "Dada", "Jantung": "Jantung", "Perut": "Perut",
    "Pinggul": "Pinggul", "Bahu Kiri": "Bahu Kiri", "Bahu Kanan": "Bahu Kanan",
    "Siku Kiri": "Siku Kiri", "Siku Kanan": "Siku Kanan",
    "Pergelangan Tangan": "Pergelangan", "Lutut": "Lutut",
    "Pergelangan Kaki": "Kaki",
}

# ── Simpan/baca hasil AI via file ─────────────────────────
def save_result(zone, text):
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({"zone": zone, "text": text}, f, ensure_ascii=False)

def load_result():
    if not os.path.exists(RESULT_FILE):
        return None, None
    try:
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("zone"), data.get("text")
    except:
        return None, None

def clear_result():
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

# ── AI fetch ──────────────────────────────────────────────
def fetch_ai(zone_name):
    result = get_health_info(zone_name)
    lines  = []
    icons  = {
        "ZONA": "📍", "KONDISI": "🔍",
        "GEJALA": "⚠️", "SARAN": "💡", "KE DOKTER": "🏥"
    }
    for part in result.split("|"):
        part = part.strip()
        if ":" in part:
            key, _, val = part.partition(":")
            key  = key.strip()
            icon = next((i for k,i in icons.items() if key.startswith(k)), "•")
            lines.append(f"{icon} **{key}:** {val.strip()}")
    text = "\n\n".join(lines) if lines else result
    save_result(zone_name, text)

# ── Video Processor ───────────────────────────────────────
class GestureDocProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker        = None        
        self.hover_zone     = None
        self.hover_count    = 0
        self.current_zone   = None
        self.frame_counter  = 0
        self.last_landmarks = {}
        self.last_tip       = None
        self.ai_fetching    = False
        self.lock           = threading.Lock()

    def _draw_zones(self, frame, landmarks, sx, sy):
        for name, (zx, zy, r) in landmarks.items():
            dx    = int(zx * sx)
            dy    = int(zy * sy)
            dr    = int(r * ((sx+sy)/2))
            color = ZONE_COLORS.get(name, (150,150,150))
            is_active  = (name == self.current_zone)
            is_hovered = (name == self.hover_zone)
            if is_active:
                ov = frame.copy()
                cv2.circle(ov, (dx,dy), dr, color, -1)
                cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
                cv2.circle(frame, (dx,dy), dr, color, 2)
                cv2.putText(frame, ZONE_LABELS_CV.get(name, name),
                            (dx-50, dy-dr-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif is_hovered:
                cv2.circle(frame, (dx,dy), dr, color, 2)
                cv2.putText(frame, ZONE_LABELS_CV.get(name, name),
                            (dx-50, dy-dr-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.circle(frame, (dx,dy), 5, color, -1)

    def _draw_fingertip(self, frame, tip, sx, sy):
        if not tip: return
        x = int(tip[0] * sx)
        y = int(tip[1] * sy)
        color = ZONE_COLORS.get(self.hover_zone,(80,255,160)) if self.hover_zone else (80,255,160)
        cv2.circle(frame, (x,y), 13, color, -1)
        cv2.circle(frame, (x,y), 17, (230,230,230), 2)
        cv2.line(frame, (x-22,y),(x-14,y),(230,230,230),1)
        cv2.line(frame, (x+14,y),(x+22,y),(230,230,230),1)
        cv2.line(frame, (x,y-22),(x,y-14),(230,230,230),1)
        cv2.line(frame, (x,y+14),(x,y+22),(230,230,230),1)

    def _draw_hover_bar(self, frame):
        if not self.hover_zone or self.hover_count <= 0: return
        h, w = frame.shape[:2]
        prog  = min(self.hover_count / HOVER_NEEDED, 1.0)
        color = ZONE_COLORS.get(self.hover_zone, (100,220,255))
        bx1, by = 15, h-24
        bx2     = w-15
        cv2.rectangle(frame, (bx1,by),(bx2,by+10),(35,35,55),-1)
        cv2.rectangle(frame, (bx1,by),
                      (bx1+int((bx2-bx1)*prog),by+10), color, -1)
        cv2.putText(frame,
                    f"Mendeteksi: {ZONE_LABELS_CV.get(self.hover_zone,self.hover_zone)}  {int(prog*100)}%",
                    (bx1,by-7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def _draw_pose(self, frame, sx, sy):
        if not self.tracker.pose_results or not self.tracker.pose_results.pose_landmarks:
            return
        lm = self.tracker.pose_results.pose_landmarks.landmark
        for a, b in [(11,12),(11,13),(13,15),(12,14),(14,16),
                     (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
            ax,ay = int(lm[a].x*RESIZE_W*sx), int(lm[a].y*RESIZE_H*sy)
            bx,by = int(lm[b].x*RESIZE_W*sx), int(lm[b].y*RESIZE_H*sy)
            cv2.line(frame,(ax,ay),(bx,by),(60,80,200),2)
        for i in range(11,33):
            cv2.circle(frame,(int(lm[i].x*RESIZE_W*sx),int(lm[i].y*RESIZE_H*sy)),3,(80,120,255),-1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        dh, dw = img.shape[:2]
        self.frame_counter += 1

        if self.frame_counter % PROCESS_EVERY_N == 0:
            small = cv2.resize(img, (RESIZE_W, RESIZE_H))
            self.tracker.process(small)
            tip = self.tracker.get_fingertip(small)
            lms = self.tracker.get_body_landmarks(small)
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
        cv2.putText(img,"GestureDoc  |  Tunjuk bagian tubuhmu",
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,0.50,(100,220,255),1)
        cv2.rectangle(img,(0,dh-22),(dw,dh),(15,15,25),-1)
        status  = f"Aktif: {ZONE_LABELS_CV.get(self.current_zone,'-')}" if self.current_zone else "Belum ada zona aktif"
        s_color = ZONE_COLORS.get(self.current_zone,(120,120,140)) if self.current_zone else (120,120,140)
        cv2.putText(img,status,(10,dh-6),cv2.FONT_HERSHEY_SIMPLEX,0.40,s_color,1)

        return frame.from_ndarray(img, format="bgr24")

# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="GestureDoc", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f0f1a; color: #e0e0e0; }
h1,h2,h3 { color: #64dcff !important; }
footer { display:none; }
.info-card {
    background: #16162a;
    border: 1px solid #32324a;
    border-radius: 10px;
    padding: 20px;
    min-height: 250px;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh setiap 2 detik — TANPA st.rerun() ───────
st_autorefresh(interval=2000, key="ai_refresh")

# ── Baca hasil AI dari file ───────────────────────────────
zone_from_file, text_from_file = load_result()
if zone_from_file:
    st.session_state.current_zone = zone_from_file
    st.session_state.ai_text      = text_from_file

if "current_zone" not in st.session_state: st.session_state.current_zone = None
if "ai_text"      not in st.session_state: st.session_state.ai_text      = ""

# ── Layout ────────────────────────────────────────────────
st.title("🩺 GestureDoc")
st.caption("Asisten Kesehatan Berbasis Gesture Tangan — Real-time WebRTC")

col_cam, col_info = st.columns([3, 2])

with col_cam:
    st.markdown("### 📷 Kamera Live")
    ctx = webrtc_streamer(
        key="gesturedoc",
        video_processor_factory=GestureDocProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 30},
            "audio": False
        },
        async_processing=True,
    )

with col_info:
    st.markdown("### 📋 Info Kesehatan")

    if st.session_state.current_zone:
        label = ZONE_LABELS.get(st.session_state.current_zone, st.session_state.current_zone)
        st.success(f"**Zona Aktif:** {label}")
    else:
        st.info("Belum ada zona dipilih — tunjuk bagian tubuhmu")

    if st.session_state.ai_text:
        st.markdown(st.session_state.ai_text)
    else:
        st.markdown("👆 Arahkan & tahan jari telunjuk ke bagian tubuhmu di kamera")

    st.divider()

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Reset", type="primary", use_container_width=True):
            st.session_state.current_zone = None
            st.session_state.ai_text      = ""
            clear_result()
            st.rerun()
    with col_r2:
        st.caption("Auto-refresh: 2 detik")

st.divider()
st.markdown("""
**Cara Penggunaan:**
1. Klik **START** dan izinkan akses webcam
2. Pastikan tubuhmu terlihat di kamera
3. Angkat **jari telunjuk** dan arahkan ke bagian tubuh
4. **Tahan** jari ~1 detik hingga progress bar penuh
5. Info kesehatan muncul otomatis dalam 2 detik
""")
