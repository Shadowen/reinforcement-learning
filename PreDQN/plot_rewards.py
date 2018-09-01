import csv
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from PreDQN.util import BASE_DATA_DIR

if __name__ == '__main__':
    # Load all data from all experiments.
    directories = list(os.listdir(BASE_DATA_DIR))
    BLACKLIST = [re.compile('cartpole', re.IGNORECASE)]
    all_data = {}
    for experiment_name in directories:
        if any(b.match(experiment_name) for b in BLACKLIST):
            continue
        with open(os.path.join(BASE_DATA_DIR, experiment_name, 'stats.csv'), 'r') as f:
            headers = [h.strip() for h in f.readline().split(',')]
            data = [[] for _ in headers]
            reader = csv.reader(f)
            for r in reader:
                for i, d in enumerate(data):
                    d.append(float(r[i]))
        all_data[experiment_name] = {h: d for h, d in zip(headers, data)}

    # Start plotting.
    plt.figure()
    for exp, data in all_data.items():
        plt.plot(data['episode_num'], pd.Series(data['episode_rewards']).rolling(10, min_periods=10).mean())
    plt.legend(all_data.keys())

    plt.show()
