## Credits
This library is based on the [Original Project Name](https://github.com/qqqqaqaqaqq/mouseMacroDetector.git) source code.
- Original Model: Transformer based Autoencoder
- Changes: Refactored for library use, added preprocessing scripts, etc.

## install
pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

## uninstall
pip uninstall QMacroDetector

---

## Class
Pattern_Game : 보안용 마우스 좌표 측정 매크로 탐지
![Pattern_Game](./public/pattern_game.png)

--- 

## python
```
from QMacroDetector import Pattern_Game, MousePoint

sample_data = {
    'x': 100, # int
    'y': 200, # int
    'timestamp': 2026-02-03T19:26:54.887758, # str
    'deltatime': 0.01 # float
}

result = Pattern_Game().get_macro_result(sample_data)

print(f"결과: {result}")
```

```
# fail
{
    "status": "1",
    "message": f"데이터가 부족합니다. 현재 {len(receive_data_list)}개 보냈습니다. 최소 51개 이상 넣어주세요.",
    "hint": {}
}

{
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

# success
{
    'status': '0', 
    'data': [
        {'raw_error': 0.01729, 'threshold': 0.054254673421382904, 'is_macro': np.False_}, 
        {'raw_error': 0.01732, 'threshold': 0.054254673421382904, 'is_macro': np.False_}, 
        {'raw_error': 0.01729, 'threshold': 0.054254673421382904, 'is_macro': np.False_},
        ...
        ]
}
```

