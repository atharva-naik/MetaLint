import asyncio
import aiohttp
import time
import os


api_key = os.getenv("ALPHAVANTAGE_API_KEY")
url = "https://www.alphavantage.co/query?function=OVERVIEW&symbol={}&apikey={}"


symbols = ["GOOG", "TSLA", "APPL", "MCFT", "SPFY", "ASUS", "AMAZ", "SMSG"]
results = []

start = time.time()

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_tasks(session):
    tasks = []
    for symbol in symbols:
        tasks.append(asyncio.create_task(session.get(url.format(symbol, api_key), ssl=False)))
    return tasks

async def get_symobls():
    async with aiohttp.ClientSession() as session:
        tasks = get_tasks(session)
        responses = await asyncio.gather(*tasks)    # (*tasks) -> (session.get(APPL API Call), session.get(GOOG API Call)) etc....
        # So it makes task easier than hardcoding
        for response in responses:
            results.append(await response.json())
            
asyncio.run(get_symobls())

end = time.time()
total = end - start

print(
    "Time taken is {} seconds to make {} API Calls".format(total, len(symbols))
)
print("Done!")

# It all the API's within 2 secs. This code makes it much faster