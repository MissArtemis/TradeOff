from fastapi import FastAPI
from entity import Model_daily_reponse, Request
from service import calculate_target
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from service import recommand_industry_offline
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
    sched.add_job(recommand_industry_offline, 'cron', hour=10, minute=36)
    logger.info('Start Offline Job')
    sched.start()

