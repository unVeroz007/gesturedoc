# app_streamlit.py — GestureDoc Web v3.0
# Arsitektur: MediaPipe JS di browser (via declare_component),
#             Python hanya handle Groq AI.
# Tidak butuh: mediapipe, streamlit-webrtc, streamlit-autorefresh

import os
import streamlit as st
import streamlit.components.v1 as components
from ai_engine import get_health_info

# ── Custom Component ──────────────────────────────────────────────
# Streamlit serve file statis dari folder "frontend/"
# → index.html di-load dalam iframe dengan allow="camera"
_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
gesture_component = components.declare_component("gesture_doc", path=_FRONTEND)

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GestureDoc",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark Theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp                        { background-color: #0f0f1a; color: #e0e0e0; }
  h1, h2, h3                    { color: #64dcff !important; }
  .stCaption, .stMarkdown p     { color: #aaaacc; }
  .stAlert                      { border-radius: 8px; }
  .stButton > button            { border-radius: 6px; }
  footer, #MainMenu             { display: none !important; }
  .block-container              { padding-top: 1.5rem; padding-bottom: 1rem; }
  hr                            { border-color: #2a2a45 !important; margin: 0.8rem 0; }
  .stExpander                   { border: 1px solid #2a2a45 !important; border-radius: 8px; }
  .stSpinner > div              { color: #64dcff; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
_DEFAULTS = {
    "current_zone": None,   # zona yang sedang aktif
    "ai_text":      "",     # hasil AI (sudah di-parse)
    "last_zone":    None,   # zona terakhir yang diproses
    "last_ts":      -1,     # timestamp terakhir dari JS (hindari proses ganda)
    "zone_cache":   {},     # cache hasil AI per zona
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helper: Parse respons AI ──────────────────────────────────────
_ICONS = {
    "ZONA":       "📍",
    "KONDISI":    "🔍",
    "GEJALA":     "⚠️",
    "SARAN":      "💡",
    "KE DOKTER":  "🏥",
}

def parse_ai_response(raw: str) -> str:
    lines = []
    for part in raw.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key  = key.strip().upper()
        icon = next((i for k, i in _ICONS.items() if key.startswith(k)), "•")
        lines.append(f"{icon} **{key.title()}:** {val.strip()}")
    return "\n\n".join(lines) if lines else raw

# ── Header ────────────────────────────────────────────────────────
st.title("🩺 GestureDoc")
st.caption("Asisten Kesehatan Berbasis Gesture Tangan · Tugas Akhir Image Processing")

# ── Two-Column Layout ─────────────────────────────────────────────
col_cam, col_info = st.columns([3, 2], gap="medium")

with col_cam:
    st.markdown("### 📷 Kamera Live")

    # Render komponen — returns {zone: str, ts: int} saat zona terdeteksi
    comp_val = gesture_component(default=None, height=520)

with col_info:
    st.markdown("### 📋 Info Kesehatan")

    # ── Proses zona yang diterima dari JS ─────────────────
    if comp_val and isinstance(comp_val, dict):
        zone = comp_val.get("zone", "")
        ts   = comp_val.get("ts", 0)

        # Proses hanya jika timestamp baru (hindari re-run Streamlit yang trigger ulang)
        if zone and ts != st.session_state.last_ts:
            st.session_state.last_ts = ts

            if zone != st.session_state.last_zone:
                st.session_state.last_zone    = zone
                st.session_state.current_zone = zone

                # Gunakan cache jika zona pernah diquery
                if zone in st.session_state.zone_cache:
                    st.session_state.ai_text = st.session_state.zone_cache[zone]
                else:
                    with st.spinner(f"🤔 Menganalisis zona: **{zone}**..."):
                        raw_result = get_health_info(zone)
                        parsed     = parse_ai_response(raw_result)
                        st.session_state.zone_cache[zone] = parsed
                        st.session_state.ai_text          = parsed

    # ── Tampilan hasil ────────────────────────────────────
    if st.session_state.current_zone:
        st.success(f"**Zona Aktif:** {st.session_state.current_zone}")
    else:
        st.info("Belum ada zona dipilih — arahkan jari telunjukmu ke bagian tubuh di kamera.")

    if st.session_state.ai_text:
        st.markdown(st.session_state.ai_text)
    else:
        st.markdown(
            "👆 Arahkan & tahan jari telunjuk ke bagian tubuhmu.\n\n"
            "_Progress bar akan muncul saat zona terdeteksi._"
        )

    st.divider()

    # ── Tombol Reset ──────────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔄 Reset", type="primary", use_container_width=True):
            # Simpan ts saat ini agar zone yang sama tidak langsung trigger lagi
            cur_ts = comp_val.get("ts", 0) if (comp_val and isinstance(comp_val, dict)) else st.session_state.last_ts
            st.session_state.update({
                "current_zone": None,
                "ai_text":      "",
                "last_zone":    None,
                "last_ts":      cur_ts,   # kunci: tandai ts ini sudah "diproses"
            })
            st.rerun()
    with btn_col2:
        st.caption("AI: Groq llama-3.3-70b")

# ── Panduan ───────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Cara Penggunaan & Tips"):
    st.markdown("""
**Langkah-langkah:**
1. **Izinkan akses kamera** ketika browser meminta (klik Allow/Izinkan)
2. Pastikan **seluruh tubuh terlihat** di kamera, pencahayaan cukup terang
3. Angkat **jari telunjuk** dan arahkan ke bagian tubuh yang ingin diketahui
4. **Tahan posisi** ±1 detik — progress bar hijau akan terisi
5. Info kesehatan AI muncul otomatis di panel kanan
6. Klik **Reset** untuk berpindah ke zona baru

**Zona yang bisa dideteksi:**
Kepala · Mata · Hidung · Mulut · Telinga · Dagu · Leher · Dada & Paru-paru ·
Jantung · Perut · Pinggul · Bahu · Siku · Pergelangan Tangan · Lutut · Pergelangan Kaki

> 💡 **Tips:** Jika kamera tidak muncul, klik ikon 🔒 di address bar browser → Izinkan Kamera → Refresh.
    """)
