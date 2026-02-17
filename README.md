## install
pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

## uninstall
pip uninstall QMacroDetector

---

# 0.1.8
* **경량화 버전** : 최소 필요 마우스 포인터 갯수 106개
* **매크로 타겟 도메인으로 변경**

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