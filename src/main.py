import csv
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
from time import time

from Agent import Agent
from create_video import create_final_video
from plot import plot_learning
import parameters as p

trialNumber = 1
episodeNumber = 3000
recordingPeriod = 100  # Specify every how many episodes to record a clip

for trial in range(trialNumber):
    renderMode = "human" if p.render else "rgb_array" #TODO None when not recording
    training_env = gym.make(p.envName, render_mode=renderMode, hardcore=p.hardcore)
    training_env.name = f"{p.envName}_{'hardcore_' if p.hardcore else ''}{trial}"

    saveFolder = os.path.join(p.parentSaveFolder, training_env.name)
    os.makedirs(saveFolder, exist_ok=True)

    videoFolder = os.path.join(saveFolder, "videos")
    os.makedirs(videoFolder, exist_ok=True)
    test_env = training_env
    test_env = RecordVideo(
        env=test_env,
        video_folder=videoFolder,
        name_prefix=test_env.name,
        episode_trigger=lambda x: x % recordingPeriod == 0,
    )

    csvName = os.path.join(saveFolder, f"{training_env.name}-data.csv")

    agent = Agent(training_env, p.learningRate, p.gamma, p.tau, p.resume, saveFolder)
    state, info = training_env.reset()
    step = 0
    runningReward = None

    # Determine the last episode if we have saved training in progress
    numEpisode = 0
    if p.resume and os.path.exists(csvName):
        with open(csvName, newline="") as f:
            fileData = list(csv.reader(f))
            if fileData:
                lastLine = fileData[-1]
                numEpisode = int(lastLine[0])
                runningReward = float(lastLine[2])

    test_env.start_video_recorder()
    start_time = time()

    while numEpisode < episodeNumber and (
        runningReward is None or runningReward <= p.satisfyingScore
    ):
        # choose an action from the agent's policy
        action = agent.getNoisyAction(state, p.actionSigma)
        # take a step in the environment and collect information
        nextState, reward, terminated, truncated, info = training_env.step(action)
        # store data in buffer
        agent.buffer.store(state, action, reward, nextState, terminated)

        if terminated or truncated:
            elapsed_time = time() - start_time
            start_time = time()
            print(f"training episode: {elapsed_time:.2f} s")
            numEpisode += 1
            # evaluate the deterministic agent on a test episode
            sumRewards = 0.0
            state, info = test_env.reset()
            terminated = truncated = False
            while not terminated and not truncated:
                action = agent.getDeterministicAction(state)
                nextState, reward, terminated, truncated, info = test_env.step(action)
                if p.render:
                    test_env.render()
                state = nextState
                sumRewards += reward
            elapsed_time = time() - start_time
            start_time = time()
            print(f"testing episode: {elapsed_time:.2f} s")
            state, info = training_env.reset()
            # keep a running average to see how well we're doing
            runningReward = (
                sumRewards
                if runningReward is None
                else runningReward * 0.99 + sumRewards * 0.01
            )
            # log progress in csv file
            fields = [numEpisode, sumRewards, runningReward]
            with open(csvName, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            agent.save()
            # print episode tracking
            print(
                f"episode {numEpisode:6d} --- "
                f"total reward: {sumRewards:7.2f} --- "
                f"running average: {runningReward:7.2f}",
                flush=True,
            )
            plot_learning(csvName)
        else:
            state = nextState
        step += 1

        shouldUpdatePolicy = step % p.policyDelay == 0
        agent.update(
            p.miniBatchSize, p.trainingSigma, p.trainingClip, shouldUpdatePolicy
        )

    training_env.close()
    test_env.close()
    create_final_video(videoFolder, should_delete_clips=False)
