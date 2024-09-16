import numpy as np
import keras
from datetime import datetime
from scorelogger import ScoreLogger
import imageio
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import gymnasium as gym
from collections import deque
import random
from gym.wrappers import RecordVideo

# Define the environment
env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 
action_space = env.action_space

class Agent:
    def __init__(self, state_size, action_size, action_space):
        self.replay_buffer = deque(maxlen=100_000)
        self.learning_rate = 0.005
        self.epsilon = 1   
        self.max_eps = 1
        self.min_eps = 0.01          
        self.eps_decay = 0.99954  #to get 0,3 in 2500
        self.gamma = 0.99
        self.batch_size = 32
        self.max_steps = 200
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.model = self.build_nn()


    def build_nn(self):
        model = Sequential()                                                          
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # add tuple to experience buffer
    def add_experience(self, new_state, reward, terminated, state, action):
        self.replay_buffer.append((new_state, reward, terminated, state, action)) 

    # e-greedy policy
    def action(self, state):
        num = np.random.rand()
        if num < self.epsilon: 
            #exploration              
            return self.action_space.sample()
        else:
            #exploitation
            return np.argmax(self.model.predict(state)[0])

    #used in testing
    def predict(self, state):
        return np.argmax(self.model.predict(state)[0])

    def replay(self, episode):
        minibatch = random.sample(self.replay_buffer, self.batch_size)   
        for new_state, reward, terminated, state, action in minibatch:                           
            target = reward
            if not terminated:
                target += self.gamma * np.max(self.model.predict(new_state)[0])
            target_function = self.model.predict(state)    
            target_function[0][action] = target
            self.model.fit(state, target_function, epochs=1, verbose=0)              

        if self.epsilon > self.min_eps:
            self.epsilon *= self.eps_decay

# initialize stuff for training
train_episodes = 1
agent = Agent(state_size, action_size, action_space)
train_score_logger = ScoreLogger('train_dqn_11', 1, 10)

#start_time = datetime.now()
#print(f"Training started at: {start_time}")

train_end = 0

# Train our model
for episode in range(train_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    # min_x = float('inf')
    # max_x = float('-inf')
    for step in range(agent.max_steps):

        action = agent.action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_state = np.reshape(new_state, [1, state_size])   
        # add a reward if the agent moves in the same direction as car's momentum
        if new_state[0][0] - state[0][0] > 0 and action == 2: 
            reward = reward + 3
        if new_state[0][0] - state[0][0] < 0 and action == 0: 
            reward = reward + 3
        #curr_x = new_state[0][0]
        #if curr_x > max_x:
        #    max_x = curr_x
        #if curr_x < min_x:
        #    min_x = curr_x
        if terminated:
            train_end += 1
            reward = 50
            score += reward
            agent.add_experience(new_state, reward, terminated, state, action)
            break
        else:
            score += reward
            agent.add_experience(new_state, reward, terminated, state, action)

        state = new_state 
        
    train_score_logger.add_score(score, episode, agent.epsilon)
    #train_score_logger.add_x_positions([min_x, max_x], episode)

    if len(agent.replay_buffer) > agent.batch_size:
        agent.replay(episode) 

# Print the end time
#end_time = datetime.now()
#print(f"Training ended at: {end_time}")
#print(f"Training duration: {end_time - start_time}")

#print(f"{train_end}/{train_episodes} episodes reached the flag in training")

#initialize stuff for testing
test_score_logger = ScoreLogger('test_dqn_11', 1, 10)
test_episodes= 100
test_end = 0
eval_time = datetime.now()
print(f"Testing started at: {eval_time}")
# Define the environment
env = gym.make('MountainCar-v0', render_mode= "rgb_array")

#env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: x % 10 == 0)

# Evaluate the model
for episode in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    score=0
    for step in range(agent.max_steps):
        action = agent.predict(state)
        new_state, reward, terminated, _, _ = env.step(action)
        new_state = np.reshape(new_state, [1, state_size])
        if terminated:
            test_end += 1
            score += reward
            break
        state = new_state
        score += reward
    test_score_logger.add_score(score, episode, agent.epsilon)

# Print the end time
eval__end_time = datetime.now()
print(f"Testing ended at: {eval_time}")
print(f"Testing duration: {eval__end_time - eval_time}")

print(f"{test_end}\{test_episodes} episodes reached the flag in testing")

