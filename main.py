from learner import Learner, Actor, TableCritic, NNCritic
from SimWorld import SimWorld
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import sys
from configparser import ConfigParser

def init_sim_world(sim_config):
    board_type = sim_config['board_type']
    board_size = int(sim_config['board_size'])
    open_cells = eval(sim_config['open_cells'])

    r_win = int(sim_config['r_win'])
    r_loss = int(sim_config['r_loss'])
    r_step = int(sim_config['r_step'])

    sim_world = SimWorld(open_cells=open_cells,
                         board_type=board_type,
                         board_size=board_size,
                         r_win=r_win,
                         r_loss=r_loss,
                         r_step=r_step
                         )

    return sim_world


def init_actor(actor_config):
    learning_rate = float(actor_config['learning_rate'])
    epsilon = float(actor_config['epsilon'])
    epsilon_decay_factor = float(actor_config['epsilon_decay_factor'])
    decay_factor = float(actor_config['decay_factor'])
    discount_factor = float(actor_config['discount_factor'])

    actor = Actor(learning_rate=learning_rate,
                  epsilon=epsilon,
                  epsilon_decay_factor=epsilon_decay_factor,
                  decay_factor=decay_factor,
                  discount_factor=discount_factor)
    return actor


def init_critic(critic_config, state_shape):
    critic_type = critic_config['critic_type']
    learning_rate = float(critic_config['learning_rate'])
    decay_factor = float(critic_config['decay_factor'])
    discount_factor = float(critic_config['discount_factor'])
    if critic_type == 'table':
        return TableCritic(learning_rate=learning_rate,
                           decay_factor=decay_factor,
                           discount_factor=discount_factor)

    layers = eval(critic_config['NN_layers'])
    keras_model = tf.keras.models.Sequential()
    keras_model.add(tf.keras.Input(
        shape=state_shape
    ))

    for neurons in layers:
        keras_model.add(
            tf.keras.layers.Dense(
                units=neurons,
                activation='tanh'
            )
        )

    keras_model.add(
        tf.keras.layers.Dense(
            units=1,
            activation='tanh'
        )
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    keras_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[
                            tf.keras.metrics.MeanSquaredError()
                        ])

    critic = NNCritic(keras_model=keras_model,
                      decay_factor=decay_factor,
                      discount_factor=discount_factor)
    return critic


def init_learner(actor, critic, sim_world):

    learner = Learner(actor=actor,
                      critic=critic,
                      simworld=sim_world,
                      )
    return learner


def main():

    if len(sys.argv) < 2:
        print("No configuration file provided, try again.")
        return
    config = ConfigParser()

    config.read("./config/" + sys.argv[1])

    # config = ConfigParser()
    # config.read("./config/config_triangle5_nn.ini")

    sim_world = init_sim_world(config['simworld'])

    actor = init_actor(config['actor'])

    state_shape = sim_world.flatten_state().shape
    critic = init_critic(config['critic'], state_shape)

    learner = init_learner(actor, critic, sim_world)

    episodes = int(config['learner']['episodes'])
    remaining_pegs = learner.learn(episodes=episodes)
    fig1 = plt.figure(1)
    plt.plot(remaining_pegs)
    plt.xlabel("Episode")
    plt.ylabel("Remaining pegs")
    fig1.show()

    visualize_after = config['visualization'].getboolean('visualize_after')

    if visualize_after:
        frame_delay = float(config['visualization']['frame_delay'])
        remaining = learner.run_episode(sim_world, visualize_after, frame_delay)
        print("Remaining pegs was {remaining}".format(remaining=remaining))
        input("Press enter to quit")




if __name__ == "__main__":
    main()
