# body_zones.py - v4.0

import math

ZONE_COLORS = {
    # Wajah
    "Kepala":               (100, 220, 255),
    "Mata Kiri":            (255, 230, 80),
    "Mata Kanan":           (255, 230, 80),
    "Hidung":               (200, 200, 100),
    "Mulut":                (255, 150, 150),
    "Telinga Kiri":         (200, 180, 100),
    "Telinga Kanan":        (200, 180, 100),
    "Dagu":                 (180, 180, 180),
    # Badan
    "Leher":                (150, 255, 200),
    "Dada & Paru-paru":     (100, 130, 255),
    "Jantung":              (255, 70,  70),
    "Perut":                (80,  210, 120),
    "Pinggul":              (200, 130, 255),
    "Bahu Kiri":            (255, 160, 80),
    "Bahu Kanan":           (255, 160, 80),
    "Siku Kiri":            (180, 255, 130),
    "Siku Kanan":           (180, 255, 130),
    "Pergelangan Tangan":   (130, 200, 255),
    "Lutut":                (255, 200, 80),
    "Pergelangan Kaki":     (160, 255, 220),
}

ZONE_LABELS = {
    # Wajah
    "Kepala":               "🧠 Kepala",
    "Mata Kiri":            "👁️ Mata Kiri",
    "Mata Kanan":           "👁️ Mata Kanan",
    "Hidung":               "👃 Hidung",
    "Mulut":                "👄 Mulut",
    "Telinga Kiri":         "👂 Telinga Kiri",
    "Telinga Kanan":        "👂 Telinga Kanan",
    "Dagu":                 "🫦 Dagu",
    # Badan
    "Leher":                "🦒 Leher",
    "Dada & Paru-paru":     "🫁 Dada",
    "Jantung":              "🫀 Jantung",
    "Perut":                "🍽️ Perut",
    "Pinggul":              "🦴 Pinggul",
    "Bahu Kiri":            "💪 Bahu Kiri",
    "Bahu Kanan":           "💪 Bahu Kanan",
    "Siku Kiri":            "🦾 Siku Kiri",
    "Siku Kanan":           "🦾 Siku Kanan",
    "Pergelangan Tangan":   "⌚ Pergelangan",
    "Lutut":                "🦵 Lutut",
    "Pergelangan Kaki":     "🦶 Kaki",
}

def get_nearest_zone(fingertip, body_landmarks):
    """
    Cari zona tubuh terdekat dengan ujung jari telunjuk.
    Hanya return zona jika jari benar-benar di dalam radius zona tersebut.
    Jika ada overlap, pilih zona dengan radius TERKECIL (paling spesifik).
    """
    if not fingertip or not body_landmarks:
        return None, None

    fx, fy = fingertip
    best_zone = None
    best_radius = float("inf")

    for zone_name, (zx, zy, radius) in body_landmarks.items():
        dist = math.sqrt((fx - zx)**2 + (fy - zy)**2)
        # Jika jari dalam radius zona ini DAN radius zona ini lebih kecil
        # (lebih spesifik) dari zona terbaik sebelumnya
        if dist <= radius and radius < best_radius:
            best_radius = radius
            best_zone   = zone_name

    return (best_zone, best_radius) if best_zone else (None, None)