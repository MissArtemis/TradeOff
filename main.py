from fastapi import FastAPI
from entity import Model_daily_reponse, Request
from func import calculate_target, model_xgb_daily

app = FastAPI()


@app.post("/query")
async def query(request: Request):
    return calculate_target(request)


@app.post("/model", response_model=Model_daily_reponse)
async def model(request: Request):
    if request.target == 'xgb':
        return model_xgb_daily(request.ts_code, request.start_time, request.end_time)
    else:
        return Model_daily_reponse()
