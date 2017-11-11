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
NUM_THREADS = 4
SAVE_STEP = 100000
SAVE_PATH = "/Users/lukejohnston/NEO/saves"
SAVE_NAME = SAVE_PATH + '/model.cpkt'
TRAINING = True
RESTORING = False
RENDERING = False
EPSILON_ANNEAL_FRAMES = 1e9 # paper was 4e9



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
      self.s = tf.placeholder(dtype=tf.float32, shape=input_shape)
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
      h = tf.nn.conv2d(self.s, W1, strides=strides1, padding=padding1) + b1
      h = tf.nn.relu(h)
      self.h1 = h
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
      self.h2 = h
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


      self.v = h[:,num_actions]

      action_onehot = tf.one_hot(self.action_taken, depth=num_actions)
      self.action_onehot = action_onehot
      # Avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
      self.pi_loss = - tf.reduce_sum(log_pi * action_onehot * tf.expand_dims(self.adv, 1))
      beta = 0.01
      self.entropy = - tf.reduce_sum(self.pi * log_pi) 
      self.v_loss = 0.5 * tf.nn.l2_loss(self.v - self.v_target) # 0.5 hyperparam from paper

      # RMSprop apply gradients expects gradients of the loss
      self.grads = tf.gradients(self.pi_loss + self.v_loss - beta * self.entropy, self.var_list) # add opp to compute gradients
      self.grads = [tf.clip_by_norm(g, 10) for g in self.grads]

      self.grad_inputs = [tf.placeholder(dtype=v.dtype, shape=v.shape) for v in self.var_list]
      self.apply_grads = tf.train.RMSPropOptimizer(1e-3, decay=0.99, use_locking=True).apply_gradients(zip(self.grad_inputs,self.var_list))


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
    return sess.run((self.grads,self.pi_loss,self.v_loss,self.entropy), feed_dict)


  #@profile
  def get_policy_and_value(self, state, sess):
    state = np.expand_dims(state, 0) # add batch axis
    feed_dict = {self.s : state}
    return sess.run((self.pi, self.v),feed_dict)

  def test(self, state, sess):
    state = np.expand_dims(state, 0) # add batch axis
    feed_dict = {self.s : state}
    return sess.run((self.s, self.h1, self.h2, self.pi, self.v),feed_dict)
 
  #@profile
  def get_value(self, state, sess):
    state = np.expand_dims(state, 0) # add batch axis 
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



# will call from a bunch of different threads
class Worker:
  total_frames = 0
  frame_count_lock = threading.Lock()
  sess = None
  episode_rewards_lock = threading.Lock()
  episode_rewards = []
  ave_ep_R = -21 # average episode reward
  ave_ep_len = 1000 # average episode length
  ave_V_loss = 0 # average value function loss
  ave_Pi_loss = 0 # average policy loss
  ave_entropy = 0 # average entropy
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

  def startSession():
    Worker.sess = tf.Session()
    if TRAINING:
      Worker.sess.run(tf.global_variables_initializer())

  #@profile
  def updateFrameCount(frames_to_add):
    Worker.frame_count_lock.acquire()
    Worker.total_frames += frames_to_add
    Worker.frame_count_lock.release()

  #@profile
  def updateEpisodeRewards(ep_R, ep_len, ep_v_loss, ep_Pi_loss, ep_entropy):
    alpha = 0.9 # blending factor to accumulate averages
    Worker.episode_rewards_lock.acquire()
    Worker.ave_ep_R = Worker.ave_ep_R * (alpha) + (1 - alpha) * ep_R
    Worker.ave_ep_len = Worker.ave_ep_len * (alpha) + (1 - alpha) * ep_len
    Worker.ave_V_loss = Worker.ave_V_loss * (alpha) + (1 - alpha) * ep_v_loss
    Worker.ave_Pi_loss = Worker.ave_Pi_loss * (alpha) + (1 - alpha) * ep_Pi_loss
    Worker.ave_entropy = Worker.ave_entropy * (alpha) + (1 - alpha) * ep_entropy
    Worker.episode_rewards_lock.release()
    f = Worker.total_frames; 
    t = (time.time() - Worker.starttime)
    fps = f / t
    print("AER: %f, AEL: %f, V loss: %f, Pi Loss: %f, H: %f, Hrs: %f, FPS: %f, Frames: %d" 
        % (Worker.ave_ep_R, Worker.ave_ep_len, Worker.ave_V_loss, Worker.ave_Pi_loss, Worker.ave_entropy, t/60/60,fps,f))




  #@profile
  def train(self):
    # train
    episode_rewards = 0
    episode_frame_count = 0
    v_losses = []
    pi_losses = []
    entropies = []
    while Worker.total_frames < Worker.max_frames:


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
        #print(policy)
        action = np.random.choice(range(policy.shape[1]),p=policy[0])
        #print(action)
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
        if (done):
          ave_V_loss = sum(v_losses) / episode_frame_count
          ave_Pi_loss = sum(pi_losses) / episode_frame_count
          ave_entropy = sum(entropies) / episode_frame_count
          Worker.updateEpisodeRewards(episode_rewards, episode_frame_count, 
              ave_V_loss, ave_Pi_loss, ave_entropy)
          episode_rewards = 0
          episode_frame_count = 0
          v_losses = []; pi_losses = []; entropies = []
          self.state = np.repeat(self.env.reset(), FRAMES_FOR_INPUT, axis=2) # first state is just start frame repeated FRAMES_FOR_INPUT times
        if (self.ID == 0 and Worker.total_frames - Worker.last_save > SAVE_STEP):
          self.saver.save(self.sess, SAVE_NAME)
          Worker.last_save = Worker.total_frames

          
        
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
        advantages = [target - est for (est,target) in zip(value_ests, value_targets)]

        grads,pi_loss,v_loss,entropy = self.model.compute_grads(states, actions, advantages, value_targets, Worker.sess)
        v_losses.append(v_loss)
        pi_losses.append(pi_loss)
        entropies.append(entropy)
        self.global_model.apply_gradients(grads,Worker.sess)





def main():
  num_threads = NUM_THREADS
  if not TRAINING: num_threads = 1
  tempEnv, _ = initEnv()
  num_actions = tempEnv.action_space.n
  tempEnv.close()
  # create global and local models
  global_model = Model('global',num_actions)
  # create saver now, so that it only saves global model
  saver = tf.train.Saver(tf.global_variables())
  Worker.startSession()
  if RESTORING:
    saver.restore(Worker.sess, SAVE_NAME)
  if TRAINING:
    thread_models = [Model(str(ID), num_actions, global_model) for ID in range(num_threads)]
    # make a Worker instance for each worker
    threads = [threading.Thread(target=Worker(ID,model,initEnv,global_model,saver).train) for ID,model in enumerate(thread_models)]
    for t in threads: t.start()
  else:
    w = Worker(0,global_model,initEnv,global_model,saver)
    w.train()



if __name__ == "__main__": main()
