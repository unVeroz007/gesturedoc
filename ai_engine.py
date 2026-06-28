# ai_engine.py - Web Version (tanpa TTS)

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY tidak ditemukan! Tambahkan di Streamlit Secrets atau file .env")

client = Groq(api_key=api_key)

def get_health_info(zone_name: str) -> str:
    prompt = f"""Kamu adalah asisten kesehatan edukatif berbahasa Indonesia.
Pengguna menunjuk bagian tubuh: {zone_name}.

Berikan penjelasan dalam format TEPAT ini (gunakan | sebagai pemisah):

ZONA: {zone_name}|KONDISI UMUM: [2-3 penyakit umum]|GEJALA: [gejala utama]|SARAN: [saran kesehatan]|KE DOKTER JIKA: [kapan harus ke dokter]

Maksimal 15 kata per bagian. Bahasa mudah dipahami masyarakat umum."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ZONA: {zone_name}|KONDISI UMUM: Gagal memuat|GEJALA: -|SARAN: Periksa koneksi|KE DOKTER JIKA: Ada keluhan serius"
