# backend/workers/worker.py
import os
from redis import Redis
from rq import Worker, Queue

listen = ["tts"]
redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

conn = Redis.from_url(redis_url)

if __name__ == "__main__":
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn)
    worker.work()