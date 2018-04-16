import numpy as np
from collections import deque
from sklearn.linear_model import SGDRegressor

import gym
import random

class FunctionEstimator:
    def __init__(self,env,n_actions):
        self.env = env
        self.n_actions = n_actions
        self.models = []
        self.initial_state = self.featurize(env.reset())

	self.replay=deque(maxlen=100);
	self.batch=64

        for _ in range(n_actions):
            model = self._build_model()
            self.models.append(model)
        
    def _build_model(self):   
        model = SGDRegressor(
            tol = 1e-3
            ,learning_rate='constant'
        )
        model.partial_fit([self.initial_state]*self.n_actions,range(self.n_actions))
        return model
    
    def featurize(self,state):
        return state
    
    def update(self,state, action, td_target):
        state = self.featurize(state)
        return self.models[action].partial_fit([state],[td_target])
        
    def predict(self,state):
        state = self.featurize(state)
        return [self.models[a].predict([state])[0] for a in range(self.n_actions)]
        
    def training(self):
        if len(self.replay) < self.batch:
            return
        batch = random.sample(self.replay,self.batch)
        self.model.partial_fit(([r[0] for r in self.replay],[r[1] for r in self.replay]))   
		
def make_policy(estimator, epsilon, actions):
    def policy_fn(state):
        preds = estimator.predict(state)
        if np.random.rand()>epsilon:
            action = np.argmax(preds)
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn

	
env = gym.make("CartPole-v0")

n_episodes = 10000
gamma = 0.99
estimator = FunctionEstimator(env,env.action_space.n)

states = []
last_states = deque(maxlen=100)
	   
for ep in range(n_episodes):
	state = env.reset()
	done = False
	policy = make_policy(estimator, 0.99**ep, range(env.action_space.n))
	ep_reward = 0

	while not done:
		action = policy(state)
		new_state, reward, done, _ = env.step(action)
		ep_reward += reward

		# Keep track of the states    
		states.append(new_state[0])
		last_states.append(new_state[0])

		# Update the Q-function
		if done:
		td_target = reward
		if not done:
			td_target = reward + gamma*np.amax(estimator.predict(new_state))
	  
		estimator.update(state,action,td_target)
		estimator.training();
		
	# Show stats
	if (ep+1) % 100 == 0:
		print('*'*100)
		print("INFO: Reward at episode {0} is {1}".format(ep+1,ep_reward))
		print("Best state reached (overall):", np.max(states))
		print("Best state reached (last 100):", np.max(last_states))  
env.close()








