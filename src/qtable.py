import gymnasium as gym
import numpy as np
from scorelogger import ScoreLogger


env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.005
DISCOUNT = 0.99
EPISODES = 50000


train_score_logger = ScoreLogger('train_qtable_5', 1, 20)

# Number of bins to divide each dimension of the observation space
num_bins = 20 

# The observation space has two dimensions, so we create a list with two 20s
# Meaning each dimension will be divided into 20 bins
discrete_space_size = [num_bins] * len(env.observation_space.high) 
print(discrete_space_size)  # Output: [20, 20]

# Print the upper limits of the observation space for each dimension
print(env.observation_space.high)  # Output: [0.6, 0.07]

# Print the full observation space (range for each dimension)
print(env.observation_space)  # Output: Box([-1.2, -0.07], [0.6, 0.07], (2,), float32)

# Print the lower limits of the observation space for each dimension
print(env.observation_space.low)  # Output: [-1.2, -0.07]

# Calculate the size of each bin (discretization step) in the observation space
bin_size = (env.observation_space.high - env.observation_space.low) / discrete_space_size

# Create the Q-table with all values initialized to zero, with the shape determined
# by the discrete observation space and the number of possible actions.
q_table = np.random.uniform(low=0, high=0, size=(discrete_space_size + [env.action_space.n]))

# Function to convert a continuous state to a discrete state (bin index)
def get_discrete_state(continuous_state):
    # Normalize the state and determine which bin it falls into for each dimension
    discrete_state = (continuous_state - env.observation_space.low) / bin_size
    # Convert the bin indices to integers and return them as a tuple
    return tuple(discrete_state.astype(np.int32))


exploitation = 0
exploration = 0

epsilon = 1.0  # initial exploration rate
epsilon_decay = 0.999976  # rate of decay for epsilon
min_epsilon = 0.01  # minimum epsilon value

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    score = 0
    while not done:
        if np.random.random() > epsilon:
            exploitation += 1
            action = np.argmax(q_table[discrete_state])
        else:
            exploration += 1
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = truncated or terminated
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif terminated:
            print(f"Congratulation! We reached the goal! Episode: {episode} and terminated = {terminated} and reward = {reward}")
            q_table[discrete_state + (action, )] = 50

        discrete_state = new_discrete_state

    train_score_logger.add_score(score, episode, epsilon)

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

env.close()

print(q_table)
print(f"{exploration} times exploration and {exploitation} times exploitation")

env = gym.make("MountainCar-v0", render_mode='human')

env.reset()

test_score_logger = ScoreLogger('test_qtable_5', 1, 20)

done = False
test_episodes = 100
successes = 0
for ep in range(test_episodes):
    discrete_state = get_discrete_state(env.reset()[0])
    score = 0
    while not done:
        discrete_state = get_discrete_state(env.state)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        new_discrete_state = get_discrete_state(new_state)
        done = truncated or terminated
        if terminated:
            successes += 1
        discrete_state = new_discrete_state
    done = False
    test_score_logger.add_score(score, episode, 0)

print(f"Had {successes}/{test_episodes} successes")
env.close()
