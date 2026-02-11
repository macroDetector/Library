import pandas as pd
import numpy as np
import sys

def make_gauss(data: pd.DataFrame, chunk_size: int, chunk_stride: int, offset: int, train_mode:bool=True) -> np.array:
    data_np = data.values[offset:] 
    chunks = []
    eps = 1e-9

    loop_range = range(0, len(data_np) - chunk_size + 1, chunk_stride)
    total_steps = len(loop_range)

    for idx, i in enumerate(loop_range):
        window = data_np[i : i + chunk_size]
        
        # 1. 기존 통계량
        m = np.mean(window, axis=0)
        s = np.std(window, axis=0)
        diff = window - m
        
        # 0으로 나누기 방지
        s_safe = s + eps
        sk = np.mean(diff**3, axis=0) / (s_safe**3)
        kt = np.mean(diff**4, axis=0) / (s_safe**4) - 3
        
        # 2. 실측 엔트로피 계산 (Numpy 벡터 연산)
        # 각 컬럼별(feature별)로 엔트로피를 구해야 합니다.
        actual_entropy = []
        for col in range(window.shape[1]):
            # bins=10으로 구간화
            counts, _ = np.histogram(window[:, col], bins=10)
            p = counts / (counts.sum() + eps)
            p = p[p > 0] # log(0) 방지
            actual_entropy.append(-np.sum(p * np.log2(p)))
        actual_entropy = np.array(actual_entropy)

        # 3. 이론적 가우시안 엔트로피 (Differential Entropy)
        # H_gauss = 0.5 * log2(2 * pi * e * sigma^2)
        # s_safe**2 이 0에 가까우면 음의 무한대로 발산하므로 np.maximum 처리
        theo_entropy = 0.5 * np.log2(2 * np.pi * np.e * (s_safe**2) + eps)
        
        # 4. 엔트로피 갭 (Gap이 클수록 매크로일 가능성 농후)
        entropy_gap = theo_entropy - actual_entropy

        # 5. 선형성 및 연속성 지표
        diff_1 = np.diff(window, axis=0)
        roughness = np.mean(np.abs(diff_1), axis=0)

        input_feature = [
            sk,                 # 왜도 (분포의 찌그러짐)
            actual_entropy,     # 실측 엔트로피 (무질서도)
            entropy_gap,        # 가우시안과의 괴리 (인위성)
            roughness,          # 거칠기 (미세 진동)
        ]
        
        chunks.append(np.concatenate(input_feature))

        # --- 진행바 로직 (기존 유지) ---
        if train_mode:
            if (idx + 1) % max(1, (total_steps // 50)) == 0 or (idx + 1) == total_steps:
                progress = (idx + 1) / total_steps
                bar = '■' * int(20 * progress) + '□' * (20 - int(20 * progress))
                sys.stdout.write(f'\r진행중: [{bar}] {progress*100:>5.1f}% ({idx+1}/{total_steps})')
                sys.stdout.flush()

    return np.array(chunks)