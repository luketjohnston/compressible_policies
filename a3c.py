import tensorflow as tf
import tensorflow.contrib.layers as layers
import threading
import random
import time
import numpy as np
import math
import gym
from envs import create_atari_env

FRAMES_FOR_INPUT = 4
SAVE_STEP = 1000
SAVE_PATH = "/Users/lukejohnston/NEO/saves"
SAVE_NAME = SAVE_PATH + '/model.cpkt'
TRAINING = False


# given shape of input to convolutional layer and strides (and padding scheme),
# outputs the shape of the output of the convolutional layer.
def get_conv_out_shape(input_shape, strides, padding, filters):
  if padding=='SAME':
    shape = [None,0,0,filters]
    for i in (1,2):
      shape[i] = math.ceil(input_shape[i] / strides[i])
    return shape
  # TODO other padding

class Model():

  def __init__(self, scope, num_actions, global_model=None):
    with tf.variable_scope(scope):
      self.var_list = []
      input_shape = (None, 42, 42, 4)
      self.s = tf.placeholder(dtype=tf.uint8, shape=input_shape)
      self.adv = tf.placeholder(dtype=tf.float32, shape=[None])
      self.v_target = tf.placeholder(dtype=tf.float32, shape=[None])
      self.action_taken = tf.placeholder(dtype=tf.uint8, shape=[None])

      # first conv layer
      W1 = tf.get_variable(name="W1",dtype=tf.float32, shape=[8,8,4,16], 
          initializer=tf.contrib.layers.xavier_initializer())
      b1 = tf.get_variable(name="b1",dtype=tf.float32, shape=[16], 
          initializer=tf.zeros_initializer())
      self.var_list += [W1,b1]
      strides1 = [1,4,4,1]; padding1='SAME'
      x = tf.cast(self.s, tf.float32)
      h = tf.nn.conv2d(x, W1, strides=strides1, padding=padding1) + b1
      h = tf.nn.relu(h)
      hshape = get_conv_out_shape(input_shape,strides1,padding1,16)

      # second conv layer
      W2 = tf.get_variable(name="W2",dtype=tf.float32, shape=[4,4,16,32], 
          initializer=tf.contrib.layers.xavier_initializer())
      b2 = tf.get_variable(name="b2",dtype=tf.float32, shape=[32], 
          initializer=tf.zeros_initializer())
      self.var_list += [W2,b2]
      strides2 = [1,2,2,1]; padding2='SAME'
      h = tf.nn.conv2d(h, W2, strides=strides2, padding=padding2) + b2
      h = tf.nn.relu(h)
      hshape = get_conv_out_shape(hshape,strides2,padding2,32)

      # fully connected layer
      flatshape = hshape[1]*hshape[2]*hshape[3]
      h = tf.reshape(h, [-1, flatshape]) # flatten
      W3 = tf.get_variable(name="W3",dtype=tf.float32, shape=[flatshape, 256],
          initializer=tf.contrib.layers.xavier_initializer())
      b3 = tf.get_variable(name="b3",dtype=tf.float32, shape=[256],
          initializer=tf.zeros_initializer())
      self.var_list += [W3,b3]

      h = tf.matmul(h, W3) + b3
      h = tf.nn.relu(h)
      W4 = tf.get_variable(name="W4",dtype=tf.float32, shape=[256, num_actions + 1],
          initializer=tf.contrib.layers.xavier_initializer())
      b4 = tf.get_variable(name="b4",dtype=tf.float32, shape=[num_actions + 1],
          initializer=tf.zeros_initializer())
      self.var_list += [W4,b4]

      h = tf.matmul(h, W4) + b4

      pi_logits = h[:,:num_actions]
      self.pi = tf.nn.softmax(pi_logits)

      beta = 0.01
      entropy = - beta * tf.reduce_sum(self.pi * tf.log(self.pi)) # over all batch elements

      self.v = h[:,num_actions]

      action_onehot = tf.one_hot(self.action_taken, depth=num_actions)
      pi_loss = - tf.reduce_sum(tf.log(self.pi * action_onehot), axis=1) * self.adv
      # TODO entropy term
      v_loss = tf.reduce_sum(tf.pow(self.v - self.v_target,2))
      v_loss *= 0.5 # apparently paper does this

      self.grads = tf.gradients(- pi_loss - v_loss + entropy, self.var_list) # add opp to compute gradients

      self.grad_inputs = [tf.placeholder(dtype=v.dtype, shape=v.shape) for v in self.var_list]
      self.apply_grads = tf.train.RMSPropOptimizer(10e-3, use_locking=True).apply_gradients(zip(self.grad_inputs,self.var_list))


      # copy parameters from model to self. Used to update local models with global.
      if not global_model is None:
        sync_ops = []
        for (my_v, source_v) in zip(self.var_list, global_model.var_list):
          sync_ops.append(tf.assign(my_v, source_v, use_locking=True))
        self.sync_op =  tf.group(*sync_ops)


          
  #@profile
  def compute_grads(self, states, actions, advantages, value_targets, sess):
    feed_dict = {self.s : states, self.action_taken : actions, 
        self.adv : advantages, self.v_target : value_targets}
    return sess.run(self.grads, feed_dict)


  #@profile
  def get_policy_and_value(self, state, sess):
    state = np.expand_dims(state, 0) # add batch axis (but a3c doesn't use batches)
    feed_dict = {self.s : state}
    return sess.run((self.pi, self.v),feed_dict)
 
  #@profile
  def get_value(self, state, sess):
    state = np.expand_dims(state, 0) # add batch axis (but a3c doesn't use batches)
    feed_dict = {self.s : state}
    return sess.run(self.v,feed_dict)

  #@profile
  def sync_from_global(self,sess):
    sess.run(self.sync_op)


  # asynchronous update with cumulated gradients
  # (only called for the global model)
  #@profile
  def apply_gradients(self, gradlist, sess):
    feed_dict = {placeholder : grads for (placeholder, grads) in zip(self.grad_inputs, gradlist)}
    sess.run(self.apply_grads, feed_dict)


# a function that initializes an environment for a thread
def initEnv():
  env = create_atari_env('PongDeterministic-v4')
  state = env.reset()
  return env, state

# to simplify, epsilon will stay the same over each episode
def getEpsilon(total_frames):
  r = random.random()
  if r < 0.4:
    return max(0.1, 1 - (1 - 0.1) / 4e9)
  if r < 0.7:
    return max(0.01, 1 - (1 - 0.01) / 4e9)
  return max(0.5, 1 - (1 - 0.5) / 4e9)



# will call from a bunch of different threads
class Worker:
  total_frames = 0
  frame_count_lock = threading.Lock()
  sess = None
  episode_rewards_lock = threading.Lock()
  episode_rewards = []
  average_episode_reward = 0
  starttime = time.time()
  max_frames = 100000000
  last_save = 0
  def __init__(self, ID, model, initEnv, global_model, saver):
    # build up tensorflow graph
    self.ID = ID
    self.env, start_frame = initEnv()
    self.state = np.repeat(start_frame, FRAMES_FOR_INPUT, axis=2) # first state is just start frame repeated FRAMES_FOR_INPUT times
    self.model = model
    self.global_model = global_model
    self.saver = saver

  def updateState(self, frame):
    # state is a numpy array
    # for now, since using grayscale, just use third axis
    self.state = np.concatenate((self.state, frame), axis = 2)
    self.state = self.state[:,:,1:] # remove memory from 5 frames ago

  def startSession(num_threads):
    Worker.sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=num_threads))
    Worker.sess.run(tf.global_variables_initializer())

  #@profile
  def updateFrameCount(frames_to_add):
    Worker.frame_count_lock.acquire()
    Worker.total_frames += frames_to_add
    Worker.frame_count_lock.release()

  #@profile
  def updateEpisodeRewards(episode_reward):
    er = Worker.episode_rewards
    Worker.episode_rewards_lock.acquire()
    er.append(episode_reward)
    Worker.average_episode_reward = sum(er[-20:]) / len(er[-20:])
    Worker.episode_rewards_lock.release()
    f = Worker.total_frames; aer = Worker.average_episode_reward
    t = (time.time() - Worker.starttime)
    fps = f / t

    print("Frames: %d, AER: %f, Hours: %f, FPS: %f" % (f,aer,t/60/60,fps))


  #@profile
  def train(self):
    # train
    episode_rewards = 0
    episode_frame_count = 0
    epsilon = getEpsilon(Worker.total_frames)
    while Worker.total_frames < Worker.max_frames:


      # copy global variables to local graph
      self.model.sync_from_global(Worker.sess)

      # act in environment for awhile
      state_action_rewards = []
      states = []
      actions = []
      value_ests = []
      rewards = []
      num_action_before_update = 5
      for _ in range(num_action_before_update):
        episode_frame_count += 1
        # use model to determine policy for current state, and then pick action
        policy, value = self.model.get_policy_and_value(self.state, Worker.sess)
        if random.random() < epsilon:
          action = self.env.action_space.sample()
        else:
          action = np.random.choice(range(num_actions),p=policy)
        # execute action in environment, receive transition and rewards
        next_frame, last_reward, done, _ = self.env.step(action)
        # record data from this frame
        episode_rewards += last_reward
        states.append(self.state)
        actions.append(action)
        value_ests.append(value[0]) # value is array, so convert to primitive
        rewards.append(last_reward)
        self.updateState(next_frame)
        Worker.updateFrameCount(1)
        #if episode_frame_count % 100 == 0:
        #  print(episode_frame_count)
        if (done):
          Worker.updateEpisodeRewards(episode_rewards)
          episode_rewards = 0
          episode_frame_count = 0
          self.state = np.repeat(self.env.reset(), FRAMES_FOR_INPUT, axis=2) # first state is just start frame repeated FRAMES_FOR_INPUT times
          epsilon = getEpsilon(Worker.total_frames)
        if (self.ID == 0 and Worker.total_frames - Worker.last_save > SAVE_STEP and TRAINING):
          self.saver.save(self.sess, SAVE_NAME)
          last_save = Worker.total_frames
        if not TRAINING: # we are rendering
          self.env.render()
          
        
      # FOR MY ALGO, here we would have other predictor predict
      # so we can get losses

      if TRAINING:
        last_value = 0
        if not done: # estimate value of last state with value function
          last_value = self.model.get_value(self.state, Worker.sess)[0]

        rewards = [r for r in rewards] + [last_value]
        gamma = 0.99
        discounted_rewards = [r*gamma**i for i,r in enumerate(rewards)]
        value_targets = [sum(discounted_rewards[i:])/gamma**i 
            for i,_ in enumerate(discounted_rewards)][:-1]
        advantages = [est - target for (est,target) in zip(value_ests, value_targets)]

        grads = self.model.compute_grads(states, actions, advantages, value_targets, Worker.sess)
        self.global_model.apply_gradients(grads,Worker.sess)




def main():
  num_threads = 4
  if not TRAINING: num_threads = 1
  tempEnv, _ = initEnv()
  num_actions = tempEnv.action_space.n
  tempEnv.close()
  # create global and local models
  global_model = Model('global',num_actions)
  # create saver now, so that it only saves global model
  saver = tf.train.Saver()
  thread_models = [Model(str(ID), num_actions, global_model) for ID in range(num_threads)]
  # make a Worker instance for each worker
  threads = [threading.Thread(target=Worker(ID,model,initEnv,global_model,saver).train) for ID,model in enumerate(thread_models)]
  Worker.startSession(num_threads)
  for t in threads: t.start()



if __name__ == "__main__": main()
