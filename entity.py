from numpy.core.getlimits import _fr1
from pydantic import BaseModel
from typing import List, Optional


class Request(BaseModel):
    ts_code: str = "600000"
    start_time: str = "2000-09-01"
    end_time: str
    period: str = "1"
    target: str = "xgb"
    conditions: Optional[List[str]] = []


class Response(BaseModel):
    ts_code: str
    target: float


class Model_daily_reponse(BaseModel):
    ts_code: str = "600000"
    update_time: str = "2000-09-01"
    accuracy: str = "0"
    f1: str = "0"
    y_now: str = "0"


class Recommend_ind_request(BaseModel):
    update_time: str = "20211005"
