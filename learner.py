from splitgd import SplitGD
import numpy as np
import random


class Learner:
    """
    A class to run the RL agent, with input from a simulated world, an actor and a critic
    """

    def __init__(self, actor, critic, simworld):
        """
        Initializes the learner
        :param actor: The actor that will choose actions
        :param critic: The critic, will evaluate states
        :param simworld: The simworld that will handle the game
        """
        self.actor = actor
        actor.simworld = simworld  # TODO: Maybe get_action() should take simworld as argument, not the array?
        self.critic = critic
        self.simworld = simworld

    def run_episode(self, simworld, visualize=False, frame_delay=0):
        """
        Runs one episode of the game, for visualization purposes
        :param simworld: The simworld that handles the game
        :param visualize: True if run should be visualized
        :param frame_delay: Seconds of delay between frames of the visualization
        :return: None
        """
        simworld.reset_game(visualize=visualize, frame_delay=frame_delay)
        actor = self.actor
        actor.update_epsilon(zero=True)

        while True:
            state = simworld.flatten_state()
            action = actor.get_action(state)
            actor.perform_action(action)

            if simworld.is_finished():
                return simworld.get_result()

    def learn(self, episodes):
        """
        Manages the actor-critic algorithm
        :param episodes: Number of episodes the algorithm will run
        :return: List of results
        """
        actor = self.actor
        critic = self.critic
        simworld = self.simworld

        results = []

        for e in range(episodes):
            print("Epiosde {e}/{episodes}".format(e=e + 1, episodes=episodes))
            episode = []

            # Resets the game and eligibilities
            simworld.reset_game()
            episode_running = True
            actor.reset_eligibilities()
            critic.reset_eligibilities()

            # Gets the initial state and action
            state = simworld.flatten_state()
            action = actor.get_action(state)

            if action is None:
                raise ValueError("The game doesn't have any legal moves.")

            episode.append((state, action))

            # The main loop
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

                # Calculates target and td_error, records
                target = r + critic.discount_factor * critic.value(new_state)
                td_error = target - critic.value(state)
                if r == 50 or r == -50:
                    print("The target is {target}".format(target=target))
                    print("The TD Error is {td_error}".format(td_error=td_error))
                    print("Move: {action}".format(action=action.__str__()))
                episode[-1] += (target, td_error)

                # Updates policy and eligibility in actor, values and eligibilities if TableCritic
                for (state, action, _, _) in episode:
                    if isinstance(critic, TableCritic):
                        critic.update_value(state, td_error)
                        critic.update_eligibility(state)

                    actor.update_policy(state, action, td_error)

                    actor.update_eligibility(state, action)

                # Records last state-action pair
                state = new_state
                action = new_action
                episode.append((state, action))

                # Checks if end state
                if simworld.is_finished():

                    # Updates the value function of a NNCritic
                    if isinstance(critic, NNCritic):
                        for i in range(len(episode) - 1):
                            state, _, target, td_error = episode[i]
                            critic.update_value(state, td_error, target)

                    # Finalizes episode run
                    episode_running = False
                    actor.update_epsilon()
                    result = simworld.get_result()
                    results.append(result)
                    print("Remaining pegs: {remaining}".format(remaining=result))
                    print(critic.value(episode[len(episode) - 2][0]))
        return results


class Actor:
    """
    The actor class, chooses actions based on updated policy, tracks eligibilities of SAPs
    """

    def __init__(self, learning_rate, epsilon, epsilon_decay_factor, decay_factor, discount_factor):
        """
        Initializes the actor
        :param learning_rate: Actor learning rate
        :param epsilon: Initial chance of random action
        :param epsilon_decay_factor: Multiplicative decay factor of epsilon
        :param decay_factor: Actor decay factor (delta)
        :param discount_factor: Actor discount factor (gamma)
        """
        self.simworld = None
        self.eligibilities = {}
        self.policy = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor

    def reset_eligibilities(self):
        """
        Resets eligibilities to 0.
        :return: None
        """
        self.eligibilities = {}

    def update_epsilon(self, zero=False):
        """
        Updates eligibilities, either decay or set to 0
        :param zero: True if eligibilities should be set to 0
        :return: None
        """
        self.epsilon *= self.epsilon_decay_factor
        if zero:
            self.epsilon = 0

    def get_action(self, state):
        """
        Returns an action based on state and policy
        :param state: The current state
        :return: Action object
        """
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
        """
        Sends an action to simworld to be performed
        :param action: The action to be performed
        :return: Reward of action
        """
        return self.simworld.perform_action(action)

    def eligibility(self, state, action):
        """
        Returns eligibility of SAP
        :param state: The state in the SAP
        :param action: The action in the SAP
        :return: None
        """
        return self.eligibilities.get((np.array_str(state), action.__str__()), 0)

    def set_eligibility(self, state, action, val=1):
        """
        Sets eligibility to chosen value
        :param state: The state of the SAP
        :param action: The action in the SAP
        :param val: Value eligibility is set to, default to 1
        :return: None
        """
        self.eligibilities[(np.array_str(state), action.__str__())] = val

    def update_eligibility(self, state, action):
        """
        Updates eligibility of SAP
        :param state: The state of the SAP
        :param action: The action in the SAP
        :return: None
        """
        self.eligibilities[(np.array_str(state), action.__str__())] = self.eligibility(state,
                                                                                       action) * self.discount_factor * self.decay_factor

    def get_policy(self, state, action):
        """
        Returns the value of the SAP decided by the policy
        :param state: The state of the SAP
        :param action: The action in the SAP
        :return: Value (float)
        """
        return self.policy.get((np.array_str(state), action.__str__()), 0)

    def update_policy(self, state, action, td_error):
        """
        Updates the policy on specific SAP
        :param state: The state of the SAP
        :param action: The action in the SAP
        :param td_error: The latest td_error
        :return: None
        """
        self.policy[(np.array_str(state), action.__str__())] = self.get_policy(state, action.__str__()) \
                                                               + self.learning_rate * td_error \
                                                               * self.eligibility(state, action.__str__())


class TableCritic:
    """
    A critic class using a table for state value lookup
    """

    def __init__(self, learning_rate, decay_factor, discount_factor):
        """
        Initializes the critic
        :param learning_rate: Learning rate for the critic
        :param decay_factor: Decay factor for the eligibilities
        :param discount_factor: Discount factor for the critic
        :return: None
        """
        self.eligibilities = {}
        self.values = {}
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor

    def reset_eligibilities(self):
        """
        Resets eligibilities
        :return: None
        """
        self.eligibilities = {}

    def value(self, state):
        """
        Returns the value of a state, random small number if None
        :param state: The state that is evaluated
        :return: Value of state
        """
        small_random_value = np.random.uniform(-1.e-9, 1.e-9)
        return self.values.get(np.array_str(state), small_random_value)

    def update_value(self, state, td_error):
        """
        Updates value of state
        :param state: The state to be given new value
        :param td_error: The td_error used to calculate new value
        :return: None
        """
        self.values[np.array_str(state)] = self.value(state) + self.learning_rate * td_error * self.eligibility(state)

    def eligibility(self, state):
        """
        Returns eligibility of state
        :param state: The state
        :return: Eligibility of state
        """
        return self.eligibilities.get(np.array_str(state), 0)

    def update_eligibility(self, state):
        """
        Updates eligibility of state
        :param state: The state to be updated
        :return: None
        """
        self.eligibilities[np.array_str(state)] *= self.discount_factor * self.decay_factor

    def set_eligibility(self, state, val=1):
        """
        Sets eligibility to chosen value
        :param state: The state
        :param val: Value eligibility is set to
        :return: None
        """
        self.eligibilities[np.array_str(state)] = val


class NNCritic(SplitGD):
    """
    A critic class using a keras NN to approximate state values.
    Inherits from SplitGD, which is a wrapper class for accessing and modifying gradients.
    """

    def __init__(self, keras_model, decay_factor, discount_factor):
        """
        Initializes the critic
        :param keras_model: The keras model NN
        :param decay_factor: Decay factor for the eligibilities
        :param discount_factor: Discount factor for the critic
        :return: None
        """
        super().__init__(keras_model)
        self.td_error = 0
        self.eligibilities = []
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor

    def modify_gradients(self, gradients):
        """
        Modifies the gradient according to the eligibilities to fit the RL actor-critic algorithm.
        :param gradients: The gradients given by the keras model
        :return: Array of updated gradients
        """
        if not self.eligibilities:
            for gradient_layer in gradients:
                self.eligibilities.append(np.zeros(gradient_layer.shape))

        for i in range(len(gradients)):
            gradient_layer = gradients[i]
            v_grad = 0 * gradient_layer if self.td_error == 0 else 1 / self.td_error * gradient_layer
            self.eligibilities[i] = self.discount_factor * self.decay_factor * self.eligibilities[i] + v_grad

        return [self.td_error * eligibility for eligibility in self.eligibilities]

    def value(self, state):
        """
        Returns the predicted value of a state
        :param state: The state that is evaluated
        :return: Value of state
        """
        nn_input = state.reshape((-1, len(state)))
        return self.model.predict(nn_input)

    def update_value(self, state, td_error, target):
        """
        Trains the NN according to td_error and target value
        :param state: The current state
        :param td_error: The td_error used to calculate new value
        :param target: The target value
        :return: None
        """
        self.td_error = td_error[-1][-1]
        nn_input = state.reshape((-1, len(state)))
        self.fit(nn_input, target, epochs=1, verbosity=0)

    def reset_eligibilities(self):
        """
        Resets state eligibilities to 0
        :return: None
        """
        self.eligibilities = []


if __name__ == "__main__":
    pass
