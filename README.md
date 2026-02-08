## Credits
This library is based on the [Original Project Name](https://github.com/qqqqaqaqaqq/mouseMacroDetector.git) source code.
- Original Model: Transformer based Autoencoder
- Changes: Refactored for library use, added preprocessing scripts, etc.

pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

```
import macro_detector

# 임포트 경로가 잘 잡혔는지 확인
print(macro_detector.__file__) 

# 테스트 데이터 (마우스 좌표 1점)
sample_data = {
    'x': 100, # int
    'y': 200, # int
    'timestamp': 2026-02-03T19:26:54.887758, # str
    'deltatime': 0.01 # float
}

# 실행 테스트
result = macro_detector.get_macro_result(sample_data)

# 결과 출력 (초반 SEQ_LEN개까지는 데이터 쌓는 중이라 None이 나옵니다)
print(f"결과: {result}")
```