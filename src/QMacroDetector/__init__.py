import os
from QMacroDetector.macro_dectector import MacroDetector
from QMacroDetector.MousePoint import MousePoint
from typing import List

import torch
import joblib
import json
from collections import deque

from sklearn.preprocessing import RobustScaler
from QMacroDetector.TransformerMacroDetector import TransformerMacroAutoencoder
from QMacroDetector.Response import ResponseBody

class Pattern_Game:
    def __init__(self):
        print(f"version 0.2.2")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CONFIG_PATH = os.path.join(BASE_DIR, "assets", "pattern_game", "config.json")
        DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "assets", "pattern_game", "model.pt")
        DEFAULT_SCALER_PATH = os.path.join(BASE_DIR, "assets", "pattern_game", "scaler.pkl")

        self.cfg:dict = {}
        with open(CONFIG_PATH, 'r') as f:
            self.cfg:dict = json.load(f)


        FEATURES = [
            # 평균, 표준 편차 => 그래프 형상
            "speed_mean", "speed_std", 
            "acc_mean", "acc_std", 
            "micro_shake_mean", "micro_shake_std", 
            "angle_vel_mean", "angle_vel_std",
            "straightness_mean", "straightness_std",

            # 왜곡, 거칠기 => 그래프의 비대칭성 및 불규칙성
            "speed_skew", "acc_skew", "micro_shake_skew", "angle_vel_skew",
            "speed_rough", "acc_rough", "micro_shake_rough", "angle_vel_rough",
            "straightness_skew", "straightness_rough",

            # 기록기 검거 지표 (무질서도 및 고유값 비율)
            "path_sinuosity", "bending_energy",
        ]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = len(FEATURES)
        
        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=self.input_size,
            d_model=self.cfg["d_model"],
            nhead=self.cfg["n_head"],
            num_layers=self.cfg["num_layers"],
            dim_feedforward=self.cfg["dim_feedforward"],
            dropout=self.cfg["dropout"]
        ).to(self.device)

        self.model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        self.scaler:RobustScaler = joblib.load(DEFAULT_SCALER_PATH)

        self.detector = MacroDetector(cfg=self.cfg, model=self.model, scaler=self.scaler, FEATURES=FEATURES, device=self.device)        

    def get_macro_result(self, receive_data_list: List[MousePoint]):
        print(f"송신받은 데이터 개수 {len(receive_data_list)}")

        self.detector.buffer = deque(maxlen=int(len(receive_data_list)))

        for data in receive_data_list:
            p_data = {
                'timestamp': data.timestamp,
                'x': data.x,
                'y': data.y,
                'deltatime': data.deltatime
            }
            
            self.detector.push(p_data)

        try:
            result = self.detector._infer()
        finally:
            self.detector.buffer.clear()

        send_data = ResponseBody(**result)

        return send_data

    # 개발 중        
    # def get_macro_result_live(self, receive_data_list:MousePoint):
    #     self.detector.buffer = deque(maxlen=10_000)

    #     p_data = {
    #         'timestamp': receive_data_list.timestamp,
    #         'x': receive_data_list.x,
    #         'y': receive_data_list.y,
    #         'deltatime': receive_data_list.deltatime
    #     }
    
    #     self.detector.push(p_data)

    #     try:
    #         result = self.detector._infer()
    #     except Exception:
    #         pass
        
    #     if result.get("status") == "1":
    #         return None

    #     send_data = ResponseBody(**result)

    #     return send_data