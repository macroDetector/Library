import dataclasses

@dataclasses.dataclass
class ResponseBody():
    status:int
    message:str
    hint:str
    data:dict