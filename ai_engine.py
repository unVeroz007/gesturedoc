# ai_engine.py — GestureDoc Web v3.0
# Standalone: tidak ada dependency ke hand_tracker / body_zones

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Deskripsi medis per zona untuk prompt yang lebih akurat
_ZONE_DESC = {
    "Kepala":              "kepala dan otak",
    "Mata Kiri":           "mata kiri",
    "Mata Kanan":          "mata kanan",
    "Hidung":              "hidung dan sinus",
    "Mulut":               "mulut, gigi, dan tenggorokan",
    "Telinga Kiri":        "telinga kiri",
    "Telinga Kanan":       "telinga kanan",
    "Dagu":                "dagu dan rahang",
    "Leher":               "leher",
    "Dada & Paru-paru":    "dada dan paru-paru",
    "Jantung":             "jantung dan sistem kardiovaskular",
    "Perut":               "perut dan sistem pencernaan",
    "Pinggul":             "pinggul",
    "Bahu Kiri":           "bahu kiri",
    "Bahu Kanan":          "bahu kanan",
    "Siku Kiri":           "siku kiri",
    "Siku Kanan":          "siku kanan",
    "Pergelangan Tangan":  "pergelangan tangan",
    "Lutut":               "lutut",
    "Pergelangan Kaki":    "pergelangan kaki dan engkel",
}


def _get_api_key() -> str:
    """Baca API key: prioritas st.secrets (Streamlit Cloud), fallback env var (lokal)."""
    # 1. Streamlit Cloud → st.secrets["GROQ_API_KEY"]
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 2. Lokal dengan .env → os.getenv
    return os.getenv("GROQ_API_KEY", "")


def _make_client() -> Groq:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY tidak ditemukan!\n"
            "• Lokal: buat file .env → GROQ_API_KEY=gsk_xxx\n"
            "• Streamlit Cloud: Settings → Secrets → GROQ_API_KEY = \"gsk_xxx\""
        )
    return Groq(api_key=api_key)


def get_health_info(zone_name: str) -> str:
    """
    Kirim permintaan ke Groq API dan kembalikan info kesehatan
    dalam format pipe-separated: ZONA|KONDISI UMUM|GEJALA|SARAN|KE DOKTER JIKA
    """
    desc   = _ZONE_DESC.get(zone_name, zone_name)
    prompt = f"""Kamu adalah asisten kesehatan edukatif berbahasa Indonesia.
Pengguna menunjuk bagian tubuh: {desc} ({zone_name}).

Berikan penjelasan dalam format TEPAT ini (gunakan | sebagai pemisah, tanpa baris baru):

ZONA: {zone_name}|KONDISI UMUM: [2-3 penyakit/gangguan umum]|GEJALA: [gejala utama]|SARAN: [saran kesehatan preventif]|KE DOKTER JIKA: [kapan harus ke dokter]

Aturan:
- Maksimal 15 kata per bagian
- Gunakan bahasa sederhana untuk masyarakat umum
- Jangan tambahkan teks lain di luar format di atas"""

    try:
        client = _make_client()
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()

    except ValueError as e:
        # API key tidak ada
        return (
            f"ZONA: {zone_name}|"
            f"KONDISI UMUM: ⚠️ API key tidak ditemukan|"
            f"GEJALA: -|"
            f"SARAN: Tambahkan GROQ_API_KEY ke Streamlit Secrets|"
            f"KE DOKTER JIKA: Ada keluhan serius"
        )
    except Exception as e:
        return (
            f"ZONA: {zone_name}|"
            f"KONDISI UMUM: Gagal memuat ({type(e).__name__})|"
            f"GEJALA: -|"
            f"SARAN: Periksa koneksi dan API key|"
            f"KE DOKTER JIKA: Ada keluhan serius"
        )
