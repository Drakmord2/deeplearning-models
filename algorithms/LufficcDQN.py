import numpy as np
import tensorflow as tf
import random
import sys
import collections
import gym


def copy_model_parameters(from_scope, to_scope):
    """
    Copies the model's parameters of `from_model` to `to_model`.
    Args:
        from_model: model to copy the paramters from
        to_model:   model to copy the parameters to
    """
    from_model_paras = [v for v in tf.trainable_variables() if v.name.startswith(from_scope)]
    from_model_paras = sorted(from_model_paras, key=lambda v: v.name)

    to_model_paras = [v for v in tf.trainable_variables() if v.name.startswith(to_scope)]
    to_model_paras = sorted(to_model_paras, key=lambda v: v.name)

    update_ops = []
    for from_model_para, to_model_para in zip(from_model_paras, to_model_paras):
        op = to_model_para.assign(from_model_para)
        update_ops.append(op)

    return update_ops


class NeuralModel:
    def __init__(self, layers, name_scope='simple_neural_network'):
        self.num_inputs = layers[0]
        self.layers = layers
        self.num_outputs = layers[-1]
        self.name_scope = name_scope

    def get_num_outputs(self):
        return self.num_outputs

    def definition(self):
        with tf.name_scope(self.name_scope):
            inputs = tf.placeholder(tf.float32, [None, self.num_inputs], name='inputs')
            layer = inputs
            current_layer = 2

            for i, j in zip(self.layers[:-1], self.layers[1:]):
                w = tf.Variable(tf.truncated_normal([i, j]), name="layer_" + str(current_layer) + "_weights")
                b = tf.Variable(tf.constant(0.01, shape=[j]), name="layer_" + str(current_layer) + "_biases")

                dense = tf.matmul(layer, w) + b

                if len(self.layers) != current_layer:
                    dense = tf.nn.relu(dense)

                layer = dense
                current_layer += 1

            outputs = layer

        return inputs, outputs


class DeepQNetwork:
    def __init__(self,
                 model,
                 env,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.001,
                 gamma=0.9,
                 replay_memeory_size=10000,
                 batch_size=32,
                 initial_epsilon=0.5,
                 final_epsilon=0.01,
                 decay_factor=1,
                 explore_policy=None,
                 logdir=None,
                 save_per_step=1000,
                 test_per_episode=100):
        '''Q-Learning algorithm
        Args:
            model:                q funtion
            env:                  environment
            optimizer:            Tensorflow optimizer
            learning_rate:        learning rate
            gamma:                decay factor of future reward
            replay_memeory_size:  replay memeory size (Experience Replay)
            batch_size:           batch size for every train step
            initial_epsilon:      ε-greedy exploration's initial ε
            final_epsilon:        ε-greedy exploration's final ε
            decay_factor:         ε-greedy exploration's decay factor
            explore_policy:       explore policy, default is `lambda epsilon: random.randint(0, self.num_actions - 1)`
            logdir：              dir to save model
            save_per_step:        save per step
            test_per_episode:       test per episode
        '''
        self.model = model
        self.env = env
        self.num_actions = model.get_num_outputs()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

        if explore_policy is None:
            explore_policy = lambda epsilon: random.randint(0, self.num_actions - 1)

        self.explore_policy = explore_policy

        self.decay_factor = decay_factor
        self.logdir = logdir
        self.should_save_model = logdir is not None
        self.test_per_episode = test_per_episode

        self.replay_memeory = collections.deque()
        self.replay_memeory_size = replay_memeory_size
        self.batch_size = batch_size
        self.define_q_network()

        # reward of every episode
        self.rewards = []
        # session
        self.sess = tf.InteractiveSession()
        # check saved model
        self.__check_model(save_per_step)
        self.halt = False

    def __check_model(self, save_per_step):
        if self.logdir is not None:
            if not self.logdir.endswith('/'): 
                self.logdir += '/'
                
            self.save_per_step = save_per_step
            self.saver = tf.train.Saver()
            checkpoint_state = tf.train.get_checkpoint_state(self.logdir)
            
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                path = checkpoint_state.model_checkpoint_path
                self.saver.restore(self.sess, path)
                print('Restore from {} successfully.'.format(path))
            else:
                print('No checkpoint.')
                self.sess.run(tf.global_variables_initializer())
                
            self.summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
            sys.stdout.flush()
        else:
            self.sess.run(tf.global_variables_initializer())

    def define_q_network(self):
        self.input_states, self.q_values = self.model.definition()
        self.input_actions = tf.placeholder(tf.float32, [None, self.num_actions], name='actions')
        self.input_q_values = tf.placeholder(tf.float32, [None], name='target_q_values')

        # only use selected q values
        action_q_values = tf.reduce_sum(tf.multiply(self.q_values, self.input_actions), reduction_indices=1)

        # define cost
        self.cost = tf.reduce_mean(tf.square(self.input_q_values - action_q_values), name='cost')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.optimizer = self.optimizer(self.learning_rate).minimize(self.cost, global_step=self.global_step)

        tf.summary.scalar('cost_summary', self.cost)
        tf.summary.scalar('reward_summary', tf.reduce_mean(action_q_values))

    def egreedy_action(self, state):
        # Exploration
        if random.random() <= self.epsilon:
            action_index = self.explore_policy(self.epsilon)
        else:
            # Exploitation
            action_index = self.action(state)
            
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.decay_factor
            
        return action_index

    def action(self, state):
        q_values = self.q_values.eval(feed_dict={self.input_states: [state]})[0]
            
        return np.argmax(q_values)

    def q_values_function(self, states):
        return self.q_values.eval(feed_dict={self.input_states: states})

    def do_train(self, episode):
        # randomly select a batch
        mini_batches = random.sample(self.replay_memeory, self.batch_size)
        state_batch = [batch[0] for batch in mini_batches]
        action_batch = [batch[1] for batch in mini_batches]
        reward_batch = [batch[2] for batch in mini_batches]
        next_state_batch = [batch[3] for batch in mini_batches]

        # target q values
        target_q_values = self.q_values_function(next_state_batch)
        input_q_values = []
        for i in range(len(mini_batches)):
            terminal = mini_batches[i][4]
            if terminal:
                input_q_values.append(reward_batch[i])
            else:
                # Discounted Future Reward
                input_q_values.append(reward_batch[i] + self.gamma * np.max(target_q_values[i]))

        feed_dict = {
            self.input_actions: action_batch,
            self.input_states: state_batch,
            self.input_q_values: input_q_values
        }
        
        self.optimizer.run(feed_dict=feed_dict)

        self.save(feed_dict, episode)

        return feed_dict

    def save(self, feed_dict, episode=1):
        step = self.global_step.eval()
        if self.should_save_model and episode > 0 and step % self.save_per_step == 0:
            summary = self.sess.run(self.summaries, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()
            self.saver.save(self.sess, self.logdir + 'dqn', self.global_step)

    # num_episodes: train episodes
    def train(self, num_episodes):
        for episode in range(num_episodes):
            # total rewards for one episode
            episode_rewards = 0
            state = self.env.reset()

            for step in range(9999999999):
                # ε-greedy exploration
                action_index = self.egreedy_action(state)
                next_state, reward, terminal, info = self.env.step(action_index)

                # one-hot action
                one_hot_action = np.zeros([self.num_actions])
                one_hot_action[action_index] = 1

                # store trans in replay_memeory
                self.replay_memeory.append((state, one_hot_action, reward, next_state, terminal))

                # remove element if exceeds max size
                if len(self.replay_memeory) > self.replay_memeory_size:
                    self.replay_memeory.popleft()

                # now train the model
                if len(self.replay_memeory) > self.batch_size:
                    feed_dict = self.do_train(episode)

                # state change to next state
                state = next_state
                episode_rewards += reward
                if terminal:
                    # Game over. One episode ended.
                    # record every episode's total rewards
                    self.rewards.append(episode_rewards)
                    break

            print("Episode {} - Reward: {} - Epsilon: {}".format(episode, episode_rewards, self.epsilon))
            sys.stdout.flush()

            # evaluate model
            if episode > 0 and episode % self.test_per_episode == 0:
                self.test(episode, feed_dict=feed_dict, max_step_per_test=99999999)

            if self.halt:
                break

    def test(self, episode, num_tests=10, max_step_per_test=300, render=False, feed_dict=None):
        total_rewards = 0
        print('\n\t - Testing -')
        sys.stdout.flush()
        
        for _ in range(num_tests):
            state = self.env.reset()
            
            for step in range(max_step_per_test):
                if render:
                    self.env.render()
                    
                action = self.action(state)
                state, reward, terminal, info = self.env.step(action)
                total_rewards += reward
                
                if terminal:
                    break

        average_reward = total_rewards / num_tests
        print("Episode {} - Reward: {}\n".format(episode, average_reward))
        sys.stdout.flush()

        # In training
        if render is False and average_reward == 500:
            self.halt = True

            if feed_dict:
                self.save(feed_dict)

            print("\n\t- Reached Maximum Reward -")
            sys.stdout.flush()


class DQN:

    def __init__(self, environment, layers_dims, learning_rate):
        model = NeuralModel(layers_dims)
        env = gym.make(environment)

        self.model = model
        self.env = env
        self.learning_rate = learning_rate

    def train(self):
        qnetwork = DeepQNetwork(model=self.model, env=self.env, learning_rate=self.learning_rate, logdir='./tmp/CartPole/')
        qnetwork.train(num_episodes=6001)

    def run(self):
        qnetwork = DeepQNetwork(model=self.model, env=self.env, learning_rate=self.learning_rate, logdir='./tmp/CartPole/')
        qnetwork.test('-', 10, 500, True)
