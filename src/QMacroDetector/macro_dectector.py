import torch
import numpy as np
import pandas as pd
from collections import deque

from sklearn.preprocessing import RobustScaler
from QMacroDetector.indicators import indicators_generation

from QMacroDetector.make_sequence import make_seq
from QMacroDetector.loss_caculation import Loss_Calculation

class MacroDetector:
    def __init__(self, cfg:dict, model, scaler, FEATURES, device):
        self.seq_len = cfg.get("seq_len", 50)
        self.tolerance = cfg.get("tolerance", 0.02)
        self.chunk_size = cfg.get("chunk_size", 50)

        self.FEATURES = FEATURES

        self.filter_tolerance = self.tolerance * 100
        self.weight_threshold = cfg["weight_threshold"]

        self.base_threshold = cfg['threshold']

        self.device = device
        self.model = model
        self.scaler:RobustScaler = scaler

        # 안정 장치
        self.buffer = deque(maxlen=10)

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
 
    def _infer(self):
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
    
        df = df[df["deltatime"] <= self.filter_tolerance].reset_index(drop=True)
        
        df = indicators_generation(
            df_chunk=df, 
            chunk_size=self.chunk_size,
            offset=int(self.chunk_size * 1.5)
        )
        
        df_filter_chunk = df[self.FEATURES].copy()
        
        chunks_scaled_array = self.scaler.transform(df_filter_chunk)
        
        chunks_scaled_df = pd.DataFrame(chunks_scaled_array, columns=self.FEATURES)
        
        chunks_scaled_df = chunks_scaled_df * 10 # train이랑 동일 하게

        if len(chunks_scaled_df) < self.seq_len:
            return {
                "status": "1",
                "message": f"데이터가 부족합니다. 현재 {len(chunks_scaled_df)}개 분석 가능한 데이터가 있습니다. 최소 {self.seq_len}개 이상 넣어주세요.",
                "hint": {}
            }
        
        final_input:np.array = make_seq(data=chunks_scaled_df, seq_len=self.seq_len, stride=1)

        send_data = []
        for i, input in enumerate(final_input):
            last_seq = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(last_seq)

                sample_errors = Loss_Calculation(outputs=output, batch=last_seq).item()

                # 임계치 판정 logic
                is_human = sample_errors <= self.base_threshold


            _error = sample_errors / self.base_threshold * 100
                    
            send_data.append({
                "is_human": is_human,
                "error_pct": _error, 
            })
        
        return {
            "status": "0",
            "data" : send_data
        }