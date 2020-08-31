import gym
import numpy as np


class OpenAI(object):
    def __init__(self, environment, agent=None):
        self.env = gym.make(environment)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.halt = False

        self.env.seed(27)

        agent.set_parameters(self.state_size, self.action_size)
        self.model = agent

    def train(self, render):
        print("  - Training Model")

        for e in range(self.EPISODES):
            if self.halt:
                print("\t - Optimal parameters reached -")
                break

            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                if render:
                    self.env.render()

                # Init network
                if e == 0 and i == 0:
                    self.model.fit(state, np.zeros((self.model.n_y,)), np.zeros((1, self.action_size)))

                action = self.model.act(state)
                next_state, reward, done, _ = self.env.step(action)

                # one-hot action
                one_hot_action = np.zeros([1, self.action_size])
                one_hot_action[0][action] = 1
                action = one_hot_action

                self.model.remember(state, action, reward, next_state, done)

                state = next_state
                i += 1
                if done:
                    print("    Episode: {}/{}, Score: {}, e: {:.2}".format(e, self.EPISODES, i, self.model.epsilon))
                    if e > 0 and e % 1000 == 0:
                        print("Saving Trained Parameters")
                        self.model.save_params()
                    if i >= 450:
                        self.halt = True
                    break

                self.model.replay()

            if e > 0 and e % 100 == 0:
                self.test(render=False)

    def test(self, render):
        self.model.load_params()
        scores = []

        print('\n\t - Testing -')
        for e in range(10):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            i = 0
            while not done:
                if render:
                    self.env.render()

                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1

                if done:
                    scores.append(i)
                    break

        print("Average Score: ", np.max(scores))
