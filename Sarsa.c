import numpy as np
import gym


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
Q = np.zeros([env.observation_space.n, env.action_space.n])

gamma = 0.99 
alpha = 0.1
n_episodes = 100


actions = range(env.action_space.n)

score = []    
for j in range(n_episodes):
    done = False
    state = env.reset()
    
    # Play randomly 10 episodes, then reduce slowly the randomness
    policy = epsilon_greedy_policy(Q, epsilon=10./(j+1), actions = actions ) 
    
    
    ### Generate sample episode
    t=0
    while not done:
        t+=1
        action = policy(state)    
        new_state, reward, done, _ =  env.step(action)
        new_action = policy(new_state)
        
        #Book-keeping
        if done:
            Q[state,action] = Q[state,action] + alpha*(reward-Q[state,action])
            pass
        else:
            Q[state,action] = Q[state,action] + alpha*(reward+gamma*Q[new_state,new_action]-Q[state,action]) 
            pass
            
        state, action = new_state, new_action
            
        if done:
            if len(score) < 100:
                score.append(reward)
            else:
                score[j % 100] = reward
                
                
            if (j+1)%1000 == 0:
                print("INFO: Episode {} finished after {} timesteps with r={}. \
                Running score: {}".format(j+1, t, reward, np.mean(score)))
            

env.close()