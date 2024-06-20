import gymnasium as gym
from Agent import Agent
from parameters import *
#TODO correct this

def visualize(
    trial: int,
    render: bool = True,
    hardcoreEnv: bool = False,
    hardcoreModel: bool = False,
):
    env = gym.make(
        envName, render_mode="human" if render else None, hardcore=hardcoreEnv
    )

    if hardcoreModel:
        env.name = envName + "_" + "hardcore" + "_" + str(trial)
    else:
        env.name = envName + "_" + str(trial)

    # env.name = envName + "_" + str(trial)
    saveFolder = "saved"

    # Load the saved agent
    agent = Agent(env, learningRate, gamma, tau, shouldLoad=True, saveFolder=saveFolder)

    # Initialize the environment
    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Get the deterministic action from the agent
        action = agent.getDeterministicAction(state)

        # Step in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render:
            env.render()

        # Update state and accumulate reward
        state = next_state
        total_reward += reward

    print(f"Total reward: {total_reward}")


trial = 0
visualize(trial, render=True, hardcoreEnv=True, hardcoreModel=True)
