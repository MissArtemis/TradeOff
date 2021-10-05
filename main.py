from fastapi import FastAPI
from entity import Model_daily_reponse, Request, Recommend_ind_request
from service import calculate_target
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from service import recommand_industry_offline, recommend_industry_online
import logging

sched = None
logger = logging.getLogger(__name__)
app = FastAPI()


@app.post("/query")
async def query(request: Request):
    return calculate_target(request)


@app.on_event("startup")
async def load_schedule():
    global sched
    sched = BackgroundScheduler()
    sched.add_job(recommand_industry_offline, 'cron', hour=16, minute=30)
    logger.info('Start Offline Job')
    sched.start()


@app.post('/recommend/industry')
async def recommend_ind(request: Recommend_ind_request):
    return recommend_industry_online(request.update_time)
