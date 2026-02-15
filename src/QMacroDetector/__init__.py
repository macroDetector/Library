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

class Pattern_Game:
    def __init__(self):
        print(f"version 0.0.5")    
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

        try:
            all_data = []
            result = {}        
            if len(receive_data_list) < self.detector.allowable_add_data:
                return {
                    "status": "1",
                    "message": f"데이터가 부족합니다. 현재 {len(receive_data_list)}개 보냈습니다. 최소 {self.detector.allowable_add_data}개 이상 넣어주세요.",
                    "hint": {}
                }
            
            for data in receive_data_list:

                p_data = {
                    'timestamp': data.timestamp,
                    'x': data.x,
                    'y': data.y,
                    'deltatime': data.deltatime
                }
                
                self.detector.push(p_data)

            all_data = self.detector._infer()

            result = {
                "status": "0",
                "data" : all_data
            }
            self.detector.buffer.clear()

            return result
        except Exception as e:
            return {
                "status": "1",
                "message": f"데이터 형식 오류입니다. 해당 데이터 형식으로 전달 해주세요.",
                "hint": {
                    "example": [
                        {
                            "timestamp": "2026-02-08T20:48:29",
                            "x": 100,
                            "y": 200,
                            "deltatime": 0.016
                        }
                    ],
                    "description": "위와 같은 형식의 객체를 리스트에 담아 최소 51개 이상 POST 요청으로 보내야 분석이 시작됩니다."
                }
            }     