# hand_tracker.py - Web Version (tanpa TTS, tanpa window)
RESIZE_W = 480
RESIZE_H = 360

import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self):
        self.mp_hands     = mp.solutions.hands
        self.mp_pose      = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw      = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.hand_results = None
        self.pose_results = None
        self.face_results = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.hand_results = self.hands.process(rgb)
        self.pose_results = self.pose.process(rgb)
        self.face_results = self.face_mesh.process(rgb)
        return frame

    def draw_pose(self, frame):
        if self.pose_results and self.pose_results.pose_landmarks:
            lm = self.pose_results.pose_landmarks.landmark
            h, w, _ = frame.shape
            body_connections = [
                (11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
            ]
            for a, b in body_connections:
                ax,ay = int(lm[a].x*w), int(lm[a].y*h)
                bx,by = int(lm[b].x*w), int(lm[b].y*h)
                cv2.line(frame, (ax,ay), (bx,by), (60,80,200), 2)
            for i in range(11, 33):
                cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 3, (80,120,255), -1)
        return frame

    def draw_pose_scaled(self, frame, scale_x, scale_y):
        """Gambar skeleton pose dengan scaling ke ukuran display"""
        if not self.pose_results or not self.pose_results.pose_landmarks:
            return frame
        lm = self.pose_results.pose_landmarks.landmark
        h, w, _ = frame.shape

        body_connections = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
        ]
        for a, b in body_connections:
            ax = int(lm[a].x * RESIZE_W * scale_x)
            ay = int(lm[a].y * RESIZE_H * scale_y)
            bx = int(lm[b].x * RESIZE_W * scale_x)
            by = int(lm[b].y * RESIZE_H * scale_y)
            cv2.line(frame, (ax,ay), (bx,by), (60,80,200), 2)
        for i in range(11, 33):
            x = int(lm[i].x * RESIZE_W * scale_x)
            y = int(lm[i].y * RESIZE_H * scale_y)
            cv2.circle(frame, (x,y), 3, (80,120,255), -1)
        return frame

    def get_fingertip(self, frame):
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            tip = self.hand_results.multi_hand_landmarks[0].landmark[8]
            return (int(tip.x * w), int(tip.y * h))
        return None

    def get_body_landmarks(self, frame):
        h, w, _ = frame.shape
        zones = {}

        if self.face_results and self.face_results.multi_face_landmarks:
            fl = self.face_results.multi_face_landmarks[0].landmark
            def fpt(idx):
                return (int(fl[idx].x * w), int(fl[idx].y * h))

            forehead = fpt(10)
            pupil_l  = fpt(468) if len(fl) > 468 else fpt(159)
            pupil_r  = fpt(473) if len(fl) > 473 else fpt(386)
            nose_tip = fpt(1)
            mouth_c  = ((fpt(13)[0]+fpt(14)[0])//2, (fpt(13)[1]+fpt(14)[1])//2)
            ear_l    = fpt(234)
            ear_r    = fpt(454)
            chin     = fpt(152)
            head_top = (forehead[0], forehead[1] - 20)

            zones["Kepala"]       = (*head_top, 22)
            zones["Mata Kiri"]    = (*pupil_l,  18)
            zones["Mata Kanan"]   = (*pupil_r,  18)
            zones["Hidung"]       = (*nose_tip, 16)
            zones["Mulut"]        = (*mouth_c,  18)
            zones["Telinga Kiri"] = (*ear_l,    20)
            zones["Telinga Kanan"]= (*ear_r,    20)
            zones["Dagu"]         = (*chin,     18)

        if self.pose_results and self.pose_results.pose_landmarks:
            lm = self.pose_results.pose_landmarks.landmark
            PL = self.mp_pose.PoseLandmark
            def ppt(idx):
                return (int(lm[idx].x*w), int(lm[idx].y*h))
            def mid(a,b):
                return ((a[0]+b[0])//2,(a[1]+b[1])//2)

            l_sh = ppt(PL.LEFT_SHOULDER);  r_sh = ppt(PL.RIGHT_SHOULDER)
            l_el = ppt(PL.LEFT_ELBOW);     r_el = ppt(PL.RIGHT_ELBOW)
            l_wr = ppt(PL.LEFT_WRIST);     r_wr = ppt(PL.RIGHT_WRIST)
            l_hp = ppt(PL.LEFT_HIP);       r_hp = ppt(PL.RIGHT_HIP)
            l_kn = ppt(PL.LEFT_KNEE);      r_kn = ppt(PL.RIGHT_KNEE)
            l_an = ppt(PL.LEFT_ANKLE);     r_an = ppt(PL.RIGHT_ANKLE)
            nose = ppt(PL.NOSE)

            m_sh = mid(l_sh, r_sh);  m_hp = mid(l_hp, r_hp)
            m_kn = mid(l_kn, r_kn);  m_an = mid(l_an, r_an)

            neck    = (m_sh[0], m_sh[1] - int((m_sh[1]-nose[1])*0.30))
            chest   = (m_sh[0], m_sh[1] + int((m_hp[1]-m_sh[1])*0.28))
            heart   = (m_sh[0] + int((l_sh[0]-r_sh[0])*0.20), chest[1])
            abdomen = (m_sh[0], m_sh[1] + int((m_hp[1]-m_sh[1])*0.65))

            sw   = max(abs(l_sh[0]-r_sh[0]), 1)
            r_lg = int(sw*0.30); r_md = int(sw*0.22); r_sm = int(sw*0.16)

            zones["Leher"]             = (*neck,           r_sm)
            zones["Dada & Paru-paru"]  = (*chest,          r_lg)
            zones["Jantung"]           = (*heart,          r_md)
            zones["Perut"]             = (*abdomen,        r_lg)
            zones["Pinggul"]           = (*m_hp,           r_md)
            zones["Bahu Kiri"]         = (*l_sh,           r_sm)
            zones["Bahu Kanan"]        = (*r_sh,           r_sm)
            zones["Siku Kiri"]         = (*l_el,           r_sm)
            zones["Siku Kanan"]        = (*r_el,           r_sm)
            zones["Pergelangan Tangan"]= (*mid(l_wr,r_wr), r_sm)
            zones["Lutut"]             = (*m_kn,           r_md)
            zones["Pergelangan Kaki"]  = (*m_an,           r_md)

        return zones