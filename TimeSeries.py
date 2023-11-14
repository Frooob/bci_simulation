import time
from collections import deque

class TimeSeriesQueue:
    def __init__(self):
        self.data= list()
    
    def insert(self, value):
        timestamp = time.time()
        self.data.append((timestamp, value))
    
    def fetch_updates(self, last_timestamp):
        return [record for record in self.data if record[0] > last_timestamp]


if __name__ == "__main__":
    # Usage
    ts_queue = TimeSeriesQueue()
    ts_queue.insert(100)
    time.sleep(1)
    ts_queue.insert(200)

    last_known_timestamp = 0
    updates = ts_queue.fetch_updates(last_known_timestamp)
    print(updates)  # Will print all updates since the epoch
    last_known_timestamp = time.time()
    ts_queue.insert(300)
    updates = ts_queue.fetch_updates(last_known_timestamp)
    print(updates)  # Will print all updates since the epoch
