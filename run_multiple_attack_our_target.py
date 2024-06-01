import threading
import time
import random
import datetime
from run_single_attack_base import run_single_process
import os
# make the timestamp utc-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--defense', type=str, default="no_defense")
parser.add_argument('--behaviors_config', type=str, default="behaviors_config.json")
parser.add_argument('--output_path', type=str,
                    default='ours')


args = parser.parse_args()
device_list = [0,1,2,3]

defense=args.defense
timestamp = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

output_path=os.path.join("Our_GCG_target_len_20",args.output_path)
output_path=os.path.join(output_path,str(timestamp))

behaviors_config=args.behaviors_config
behavior_id_list = [i + 1 for i in range(50)]
# add id to black_list to skip the id
# black_list = []
# add id to white_list to only run the id
# white_list =[]


# behavior_id_list = [i for i in behavior_id_list if i not in black_list]
# behavior_id_list = [i for i in behavior_id_list if i in white_list]


class Card:
    def __init__(self, id):
        self.id = id
        self.lock = threading.Lock()


class ResourceManager:
    def __init__(self, device_list):
        self.cards = [Card(i) for i in device_list]

    def request_card(self):
        for card in self.cards:
            if card.lock.acquire(False):
                return card
        return None

    def release_card(self, card):
        card.lock.release()


def worker_task(task_list, resource_manager):
    while True:
        task = None
        with task_list_lock:
            if task_list:
                task = task_list.pop()

        if task is None:  # No more tasks left
            break

        card = resource_manager.request_card()
        while card is None:  # Keep trying until a card becomes available
            time.sleep(0.01)
            card = resource_manager.request_card()

        print(f"Processing task {task} using card {card.id}")
        run_single_process(task, card.id, output_path,defense,behaviors_config)
        resource_manager.release_card(card)


tasks = behavior_id_list
task_list_lock = threading.Lock()

resource_manager = ResourceManager(device_list)

# Create and start 8 worker threads
threads = [threading.Thread(target=worker_task, args=(tasks, resource_manager)) for _ in range(len(device_list))]

for t in threads:
    t.start()

for t in threads:
    t.join()

print("All tasks completed!")