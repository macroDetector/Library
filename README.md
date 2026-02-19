## install
pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

## uninstall
pip uninstall QMacroDetector

---
## Test HomePage
http://136.119.234.87

---
## 0.2.4 update
* drawing 추가
* 모바일 마우스 패드 지원
* **지표추가** : 기록기 탐지 강화

## 0.2.3 update
안정화, 필요 갯수 감소 최소 110 안정적으로는 130~

## 0.2.2 update
안정화

## Class
Pattern_Game : 보안용 마우스 좌표 측정 매크로 탐지
![Pattern_Game](./public/pattern_game.png)

--- 

## python
```
from QMacroDetector import Pattern_Game, MousePoint
from QMacroDetector.Response import ResponseBody
import dataclasses

def get_mouse_pointer(data: List[MousePoint]):
    result:ResponseBody = Pattern_Game().get_macro_result(data)

    received_data = result.data
    
    print(received_data)
```

#