from splitgd import SplitGD
import numpy as np
import random


# TODO: epsilon decay factor should be a part of Actor state?
class Learner:

    def __init__(self, actor, critic, simworld):
        self.actor = actor
        actor.simworld = simworld  # TODO: Maybe get_action() should take simworld as argument, not the array?
        self.critic = critic
        self.simworld = simworld

    def run_episode(self, simworld, visualize=False, frame_delay=0):
        simworld.reset_game(visualize=visualize, frame_delay=frame_delay)
        actor = self.actor
        actor.update_epsilon(zero=True)

        while True:
            state = simworld.flatten_state()
            action = actor.get_action(state)
            actor.perform_action(action)

            if simworld.is_finished():
                # simworld.visualize_game()
                return simworld.remaining_pegs

    def learn(self, episodes):

        actor = self.actor
        critic = self.critic
        simworld = self.simworld

        remaining_pegs = []

        for e in range(episodes):
            print("Epiosde {e}/{episodes}".format(e=e + 1, episodes=episodes))
            episode = []

            simworld.reset_game()
            episode_running = True

            actor.reset_eligibilities()
            critic.reset_eligibilities()

            state = simworld.flatten_state()
            action = actor.get_action(state)

            if action is None:
                raise ValueError("The game doesn't have any legal moves.")

            episode.append((state, action))

            while episode_running:

                # Set eligibilities to 1
                actor.set_eligibility(state, action)
                if isinstance(critic, TableCritic):
                    critic.set_eligibility(state)

                # Perform action, receive reward
                r = actor.perform_action(action)

                # New state after action is performed
                new_state = simworld.flatten_state()

                # Best action for new state according to policy
                new_action = actor.get_action(new_state)

                target = r + critic.discount_factor * critic.value(new_state)
                td_error = target - critic.value(state)
                if r == 50 or r == -50:
                    print("The target is {target}".format(target=target))
                    print("The TD Error is {td_error}".format(td_error=td_error))
                    print("Move: {action}".format(action=action.__str__()))

                for (state, action) in episode:
                    if isinstance(critic, TableCritic):
                        critic.update_value(state, td_error)
                        critic.update_eligibility(state)
                    else:
                        critic.update_value(state, td_error, target)

                    actor.update_policy(state, action, td_error)

                    actor.update_eligibility(state, action)

                state = new_state
                action = new_action
                episode.append((state, action))

                if simworld.is_finished():
                    episode_running = False
                    actor.update_epsilon()
                    remaining_pegs.append(simworld.remaining_pegs)
                    print("Remaining pegs: {remaining}".format(remaining=simworld.remaining_pegs))
                    print(critic.value(episode[len(episode) - 2][0]))
        return remaining_pegs


class Actor:

    def __init__(self, learning_rate, epsilon, epsilon_decay_factor, decay_factor, discount_factor):
        self.simworld = None
        self.eligibilities = {}
        self.policy = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor

    def reset_eligibilities(self):
        self.eligibilities = {}

    def update_epsilon(self, zero=False):
        self.epsilon *= self.epsilon_decay_factor
        if zero:
            self.epsilon = 0

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

    def update_eligibility(self, state, action):
        self.eligibilities[(np.array_str(state), action.__str__())] = self.eligibility(state,
                                                                                       action) * self.discount_factor * self.decay_factor

    def get_policy(self, state, action):
        return self.policy.get((np.array_str(state), action.__str__()), 0)

    def update_policy(self, state, action, td_error):
        self.policy[(np.array_str(state), action.__str__())] = self.get_policy(state,
                                                                               action.__str__()) + self.learning_rate * td_error * self.eligibility(
            state, action.__str__())


class TableCritic:

    def __init__(self, learning_rate, decay_factor, discount_factor):
        self.eligibilities = {}
        self.values = {}
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor

    def reset_eligibilities(self):
        self.eligibilities = {}

    def value(self, state):
        small_random_value = np.random.uniform(-1.e-9, 1.e-9)  # TODO: Too smal??
        return self.values.get(np.array_str(state), small_random_value)

    def update_value(self, state, td_error):
        self.values[np.array_str(state)] = self.value(state) + self.learning_rate * td_error * self.eligibility(state)

    def eligibility(self, state):
        return self.eligibilities.get(np.array_str(state), 0)

    def update_eligibility(self, state):
        self.eligibilities[np.array_str(state)] *= self.discount_factor * self.decay_factor

    def set_eligibility(self, state, val=1):
        self.eligibilities[np.array_str(state)] = val


class NNCritic(SplitGD):

    def __init__(self, keras_model, decay_factor, discount_factor):
        super().__init__(keras_model)
        self.td_error = 0
        self.eligibilities = []
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor

    def modify_gradients(self, gradients):
        if not self.eligibilities:
            for gradient_layer in gradients:
                self.eligibilities.append(np.zeros(gradient_layer.shape))

        for i in range(len(gradients)):
            gradient_layer = gradients[i]
            v_grad = 0 * gradient_layer if self.td_error == 0 else 1 / self.td_error * gradient_layer
            self.eligibilities[i] = self.discount_factor * self.decay_factor * self.eligibilities[i] + v_grad

        return [self.td_error * eligibility for eligibility in self.eligibilities]

    def value(self, state):
        nn_input = state.reshape((-1, len(state)))
        return self.model.predict(nn_input)

    def update_value(self, state, td_error, target):
        self.td_error = td_error[-1][-1]
        nn_input = state.reshape((-1, len(state)))
        self.fit(nn_input, target, epochs=1, verbosity=0)

    def reset_eligibilities(self):
        self.eligibilities = []


if __name__ == "__main__":
    pass
