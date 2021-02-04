from learner import Learner, Actor, Critic
from SimWorld import SimWorld

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # s = SimWorld([(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)], board_type="diamond", board_size=3)
    s = SimWorld([(2, 1)], board_type="triangular", board_size=5)

    actor = Actor(learning_rate=0.01, epsilon=0.5)
    critic = Critic(learning_rate=0.01)
    learner = Learner(actor=actor,
                      critic=critic,
                      simworld=s,
                      discount_factor=0.9,
                      decay_factor=0,
                      epsilon_decay_factor=0.99)

    remaining_pegs = learner.learn(1000)
    plt.plot(remaining_pegs)
    plt.show()
