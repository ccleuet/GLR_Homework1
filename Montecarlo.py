import numpy as np
import gym
import random

def epsilon_greedy_policy(Q, epsilon, actions):
    """ Q is a numpy array, epsilon between 0,1 
    and a list of actions"""
    
    def policy_fn(state):
        if np.random.rand()>epsilon:
            action = np.argmax(Q[state,:])
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn


env = gym.make("FrozenLake-v0")

gamma = 0.99 
n_episodes = 100

Q = np.zeros([env.observation_space.n, env.action_space.n])
R = np.zeros([env.observation_space.n, env.action_space.n])
N = np.zeros([env.observation_space.n, env.action_space.n])

actions = range(env.action_space.n)

score = []    
for j in range(n_episodes):

    done = False
    state = env.reset()
    episode = []
    policy = epsilon_greedy_policy(Q, epsilon=10./(j+1), actions = actions )       
    t=0
	
    while not done:
        t+=1
        action = policy(state)    
        new_state, reward, done, _ =  env.step(action)
        episode.append((state,action))
        state=new_state

    for s,a in episode:
        G = gamma**t*reward
        N[state,action] +=1
        Q[state,action] += (G-Q[state,action])/N[state,action]

    if done:
        if len(score) < 100:
             score.append(reward)
        else:
             score[j % 100] = reward
    print("INFO: Episode {} finished after {} timesteps with r={}.".format(j+1, t, reward))
env.close()
