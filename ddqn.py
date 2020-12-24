import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from environment import GameEnv

num_iterations = 20_000

initial_collect_steps = 1_000
collect_steps_per_iteration = 1
replay_buffer_max_length = 10_000

batch_size = 32
learning_rate = 0.001
log_interval = 200

num_eval_episodes = 10
eval_interval = 1_000

env = GameEnv()
train_env = TFPyEnvironment(env)
eval_env = TFPyEnvironment(env)
fc_layer_params = (40, 40, )
train_step_counter = tf.Variable(0)

q_net = QNetwork(train_env.observation_spec(),
                 train_env.action_spec(),
                 fc_layer_params=fc_layer_params)

agent = DdqnAgent(train_env.time_step_spec(),
                  train_env.action_spec(),
                  q_network=q_net,
                  optimizer=Adam(learning_rate=learning_rate),
                  td_errors_loss_fn=common.element_wise_squared_loss,
                  train_step_counter=train_step_counter)

agent.initialize()
agent.train = common.function(agent.train)
random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0

    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                      batch_size=train_env.batch_size,
                                      max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                   sample_batch_size=batch_size,
                                   num_steps=2).prefetch(3)
iterator = iter(dataset)

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(f"{step=}; train loss={train_loss.numpy():.6f}")

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print(f"{step=}; {avg_return=}")
        returns.append(avg_return)
