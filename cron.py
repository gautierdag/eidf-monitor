import json
import time
from datetime import datetime, timedelta

from utils import get_pods_not_using_gpus_stats

FILE_PATH = "/nfs/user/s2234411-infk8s/cluster_gpu_usage.json"


def load_data():
    try:
        with open(FILE_PATH, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def save_data(data):
    with open(FILE_PATH, "w") as file:
        json.dump(data, file, indent=4)


def main():
    new_data_list = get_pods_not_using_gpus_stats()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for data in new_data_list:
        data["timestamp"] = timestamp

    existing_data = load_data()
    # Append the new data
    existing_data.extend(new_data_list)

    # Delete out data older than 14 days
    cutoff = datetime.now() - timedelta(days=14)
    filtered_data = [
        entry
        for entry in existing_data
        if datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") > cutoff
    ]

    save_data(filtered_data)


if __name__ == "__main__":
    while True:
        print(f"Running main function at {datetime.now()}")
        main()
        # sleep for 15 minutes
        time.sleep(15 * 60)
