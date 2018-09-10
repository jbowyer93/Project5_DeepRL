tf.reset_default_graph()
import gym
from gym.wrappers import Monitor
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque

# Some parameters we can play with
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1
EPSILON_END = .005
EPSILON_DECAY = .95
LR = 5e-4
UPDATE_EVERY = 4
N1 = 50
N2 = 40
REPLACE_EVERY = 100

class DQN():
    def __init__(self, env, seed):
        # Tell the agent what game space we're playing in
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
                
        # Set the seed and create any parameters we'll be adjusting internally
        self.seed = random.seed(seed)
        self.epsilon = EPSILON_START

        # Creating the neural networks
        self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
        self.state_input_target = tf.placeholder(tf.float32, [None, self.state_size])
        self.qnetwork_train = self.QNeuralNetworkTrain(self.state_input)
        self.qnetwork_target = self.QNeuralNetworkTarget(self.state_input)
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])

        # Get the weights for target and train networks, and create fxn to update target params
        train_params = tf.get_collection('Train_params')
        target_params = tf.get_collection('Target_params')
        self.replace_params = [tf.assign(t, r) for t, r in zip(target_params, train_params)]

        # Creating the target we'll optimize against, loss and optimizer
        self.Q_target = tf.placeholder(tf.float32, [None, BATCH_SIZE])
        self.Q_train = tf.placeholder(tf.float32, [None, BATCH_SIZE])
        self.loss = tf.reduce_mean(tf.losses.huber_loss(self.Q_train, self.Q_target))
        self.optimizer = tf.train.AdamOptimizer(LR)
        
        # Creating the memory for experience replay, and a time step
        self.memory = deque()
        self.t_step = 0
        self.t_step_2 = 0
        self.episodes = 0

        # Some necessary steps
        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.global_variables_initializer()

    def QNeuralNetworkTrain(self, state_input, scope = 'Train'):
        with tf.variable_scope(scope):
            namespace, layer1_nodes, layer2_nodes, weights, biases = [scope+'_params', tf.GraphKeys.GLOBAL_VARIABLES], N1, N2, tf.random_normal_initializer(0.0,0.5), tf.constant_initializer(0.1)

            w1 = tf.get_variable("w1", [self.state_size, N1], initializer = weights, collections = namespace)                                      
            b1 = tf.get_variable('b1', [1, N1], initializer = biases, collections = namespace)
            layer1 = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)

            w2 = tf.get_variable("w2", [N1, N2], initializer = weights, collections = namespace)
            b2 = tf.get_variable('b2', [1, N2], initializer = biases, collections = namespace)
            layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

            w3 = tf.get_variable("w3", [N2, self.action_size], initializer = weights, collections = namespace)
            b3 = tf.get_variable('b3', [1, self.action_size], initializer = biases, collections = namespace)

            self.Q_value_train = tf.matmul(layer2, w3) + b3

            return self.Q_value_train
        
    def QNeuralNetworkTarget(self, state_input_target, scope = 'Target'):        
        with tf.variable_scope(scope):
            namespace, layer1_nodes, layer2_nodes, weights, biases = [scope+'_params', tf.GraphKeys.GLOBAL_VARIABLES], N1, N2, tf.random_normal_initializer(0.0,0.5), tf.constant_initializer(0.1)
            
            w1ta = tf.get_variable("w1ta", [self.state_size, N1], initializer = weights, collections = namespace)                                      
            b1ta = tf.get_variable('b1ta', [1, N1], initializer = biases, collections = namespace)
            layer1ta = tf.nn.relu(tf.matmul(self.state_input, w1ta) + b1ta)

            w2ta = tf.get_variable("w2ta", [N1, N2], initializer = weights, collections = namespace)
            b2ta = tf.get_variable('b2ta', [1, N2], initializer = biases, collections = namespace)
            layer2ta = tf.nn.relu(tf.matmul(layer1ta, w2ta) + b2ta)

            w3ta = tf.get_variable("w3ta", [N2, self.action_size], initializer = weights, collections = namespace)
            b3ta = tf.get_variable('b3ta', [1, self.action_size], initializer = biases, collections = namespace)

            self.Q_value_target = tf.matmul(layer2ta, w3ta) + b3ta

            return self.Q_value_target
            
    def step(self, state, action, reward, next_state, done):
        # The step. We do the one hot action so we can easily get the q reward for ONLY the action taken
        one_hot_action = np.zeros(self.action_size)
        one_hot_action[action] = 1
        self.memory.append((state, one_hot_action, reward, next_state, done))

        # Set how often we'll update our train network weights
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                self.update()

        # Set how often we'll update our target network with train weights
        self.t_step_2 = (self.t_step_2 + 1) % REPLACE_EVERY
        if self.t_step_2:
            self.replace_params

        # Tell it to pop if the memory is at buffer size
        if len(self.memory) > BUFFER_SIZE:
            self.memory.popleft()
        
    def act(self, state):
        # Take a random action if we roll a number less than epsilon, else do what our train network says to do
        threshold = random.random()
        if threshold < self.epsilon:
            action = env.action_space.sample()
        if threshold >= self.epsilon:
            action = np.argmax(self.sess.run(self.qnetwork_train, feed_dict={self.state_input: [state]})[0])

        # Decay epsilon after each action
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            
        return action

    def update(self):
        experiences = random.sample(self.memory, BATCH_SIZE)

        states_in_batch = [experience[0] for experience in experiences]
        actions_in_batch = [experience[1] for experience in experiences]
        rewards_in_batch = [experience[2] for experience in experiences]
        next_states_in_batch = [experience[3] for experience in experiences]
        done_in_batch = [experience[4] for experience in experiences]
        
        self.Q_train_input = tf.reduce_sum(tf.multiply(self.sess.run(self.Q_value_train, feed_dict = {self.state_input: states_in_batch}), self.action_input), axis=0)

        # For our target batch, we want to grab whatever the target network outputs for the current state + its reward for the next state
        Q_target_batch = []

        # Find the max prediction for the Q values of the next states from the target model
        Q_target_future = self.sess.run(self.qnetwork_target, feed_dict={self.state_input: next_states_in_batch})

        for i in range(0,BATCH_SIZE):
            if done_in_batch:
                Q_target_batch.append(rewards_in_batch[i])
            else:
                Q_target_batch.append(rewards_in_batch[i] + GAMMA*np.max(Q_target_future[i]))

        feed_dict = {self.Q_train: self.Q_train_input,
                     self.action_input: actions_in_batch,
                     self.Q_target: Q_target_batch}
        
        self.session.run(self.optimizer.minimize(self.loss), feed_dict)
