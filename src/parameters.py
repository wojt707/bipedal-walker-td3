# HYPERPARAMETERS BELOW
gamma = 0.99  # discount factor for rewards
learningRate = 3e-4  # learning rate for actor and critic networks
tau = 0.005  # tracking parameter used to update target networks slowly
actionSigma = 0.1  # contributes noise to deterministic policy output
trainingSigma = 0.2  # contributes noise to target actions
trainingClip = 0.5  # clips target actions to keep them close to true actions
miniBatchSize = 100  # how large a mini-batch should be when updating
policyDelay = 2  # how many steps to wait before updating the policy

resume = True  # resume from previous checkpoint if possible?
render = False  # render out the environment on-screen?
hardcore = False  # set the environment mode to hardcore?

satisfyingScore = 300.0
envName = "BipedalWalker-v3"
parentSaveFolder = "saved"
