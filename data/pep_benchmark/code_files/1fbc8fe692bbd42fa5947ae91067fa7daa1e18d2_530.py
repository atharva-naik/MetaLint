import datetime as dt

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dateutil.parser import parse
from sqlalchemy import and_

from app.misc import bot
from app.models import UserSettings
from app.models.product import Product


async def advance_notification_checker():
    users = await UserSettings.query.where(and_(
        UserSettings.notifications_general_enabled, UserSettings.notifications_advance_enabled)
    ).gino.all()
    for u in users:
        answer = "Истекает срок годности:\n"
        p = await Product.query.where(and_(Product.user_id == u.user_id,
                                           Product.expiration_date == dt.datetime.today() + dt.timedelta(u.notifications_advance_days_until_expiration))).gino.all()
        for pr in p:
            answer += pr.name + " - " + parse(str(pr.expiration_date)).strftime('%d.%m') + "\n"
        await bot.send_message(u.user_id, answer)


scheduler = AsyncIOScheduler()
scheduler.add_job(advance_notification_checker, 'cron', hour='18', minute='00')
scheduler.start()
