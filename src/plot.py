import csv
import matplotlib.pyplot as plt
from os import path


def plot_learning(csv_file_name: str):
    numEpisode = []
    sumRewards = []
    runningReward = []

    if not path.exists(csv_file_name):
        print("Cannot open file " + csv_file_name)
        return

    reader = csv.reader(open(csv_file_name))
    for row in reader:
        numEpisode.append(int(row[0]))
        sumRewards.append(float(row[1]))
        runningReward.append(float(row[2]))

    # Plot total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(numEpisode, sumRewards, label="Total Reward per Episode", color="blue")
    plt.plot(numEpisode, runningReward, label="Running Average Reward", color="orange")

    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title("Total Reward and Running Average Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(csv_file_name.split(".")[0] + "_plot.png")
    print("Plot saved as " + csv_file_name.split(".")[0] + "_plot.png")
    plt.close()
