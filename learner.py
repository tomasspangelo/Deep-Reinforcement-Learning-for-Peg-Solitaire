from splitgd import SplitGD
import numpy as np
import random


# TODO: epsilon decay factor should be a part of Actor state?
class Learner:

    def __init__(self, actor, critic, simworld, discount_factor, decay_factor, epsilon_decay_factor=0.5):
        self.actor = actor
        self.discount_factor = discount_factor
        self.decay_factor = decay_factor
        self.epsilon_decay_factor = epsilon_decay_factor
        actor.simworld = simworld
        self.critic = critic
        self.simworld = simworld

    def learn(self, episodes):

        actor = self.actor
        critic = self.critic
        simworld = self.simworld

        remaining_pegs = []

        for e in range(episodes):
            print("epiosde {e}".format(e=e))
            episode = []

            simworld.reset_game()
            episode_running = True

            actor.reset_eligibilities()
            critic.reset_eligibilities()

            state = simworld.flatten_state()
            action = actor.get_action(state)

            if action is None:
                raise ValueError("The game doesn't have any legal moves.")

            episode.append((state, action.__str__()))
            while episode_running:
                # Set eligibilities to 1
                actor.set_eligibility(state, action)
                critic.set_eligibility(state)

                # Perform action, receive reward
                r = actor.perform_action(action)

                # New state after action is performed
                new_state = simworld.flatten_state()

                # Best action for new state according to policy
                new_action = actor.get_action(new_state)

                td_error = r + self.discount_factor * critic.value(new_state) - critic.value(state)

                for (state, action) in episode:
                    critic.update_value(state, td_error)

                    critic.update_eligibility(state, self.discount_factor, self.decay_factor)

                    actor.update_policy(state, action, td_error)

                    actor.update_eligibility(state, action, self.discount_factor, self.decay_factor)

                state = new_state
                action = new_action
                episode.append((state, action.__str__()))

                if simworld.is_finished():
                    episode_running = False
                    actor.epsilon *= self.epsilon_decay_factor
                    remaining_pegs.append(simworld.remaining_pegs)
                    #print(simworld.remaining_pegs)
                    #print(simworld.flatten_state())
                   # print("------------")
        return remaining_pegs


class Actor:

    def __init__(self, learning_rate, epsilon):
        self.simworld = None
        self.eligibilities = {}
        self.policy = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def reset_eligibilities(self):
        self.eligibilities = {}

    def get_action(self, state):
        legal_actions = []
        best_action = None
        best_value = float('-inf')
        for action in self.simworld.get_legal_actions():
            value = self.policy.get((np.array_str(state), action.__str__()), 0)
            legal_actions.append((action, value))

            if value > best_value:
                best_value = value
                best_action = action

        if random.random() < self.epsilon and legal_actions:
            return random.choice(legal_actions)[0]

        return best_action

    def perform_action(self, action):
        return self.simworld.perform_action(action)

    def eligibility(self, state, action):
        return self.eligibilities.get((np.array_str(state), action.__str__()), 0)

    def set_eligibility(self, state, action, val=1):
        self.eligibilities[(np.array_str(state), action.__str__())] = val

    def update_eligibility(self, state, action, discount_factor, decay_factor):
        self.eligibilities[(np.array_str(state), action.__str__())] = self.eligibility(state,
                                                                                       action) * discount_factor * decay_factor

    def get_policy(self, state, action):
        return self.policy.get((np.array_str(state), action.__str__()), 0)

    def update_policy(self, state, action, td_error):
        self.policy[(np.array_str(state), action.__str__())] = self.get_policy(state,
                                                                               action.__str__()) + self.learning_rate * td_error * self.eligibility(
            state, action.__str__())


# Critic = TableCritic
class Critic:

    def __init__(self, learning_rate):
        self.eligibilities = {}
        self.values = {}
        self.learning_rate = learning_rate

    def reset_eligibilities(self):
        self.eligibilities = {}

    def value(self, state):
        small_random_value = np.random.uniform(1.e-17, 0.1)
        return self.values.get(np.array_str(state), small_random_value)

    def update_value(self, state, td_error):
        self.values[np.array_str(state)] = self.value(state) + self.learning_rate * td_error * self.eligibility(state)

    def eligibility(self, state):
        return self.eligibilities.get(np.array_str(state), 0)

    def update_eligibility(self, state, discount_factor, decay_factor):
        self.eligibilities[np.array_str(state)] *= discount_factor * decay_factor

    def set_eligibility(self, state, val=1):
        self.eligibilities[np.array_str(state)] = val


class TableCritic(Critic):

    def __init__(self):
        pass


class NNCritic(Critic):

    def __init__(self):
        pass


# TODO: How to get the TD error
# TODO: Learning rate, where is it passed?
class Network(SplitGD):

    def __init__(self, keras_model):
        super().__init__(keras_model)
        self.td_error = 0.00000000000000001
        self.eligibility = []

    def modify_gradients(self, gradients):
        if not self.eligibility:
            self.eligibility = np.zeros(gradients.shape)

        v_grad = -1 / self.td_error * gradients
        self.eligibility += v_grad

        return self.td_error * self.eligibility
