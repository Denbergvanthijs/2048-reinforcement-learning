import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common

from environment import GameEnv

num_iterations = 20_000
warmup_steps = 20_000
collect_steps_per_iteration = 1
replay_buffer_max_length = 10_000
target_update_period = 500
decay_episodes = 5_000
min_epsilon = 0.1
gamma = 0.99
n_step_update = 1

batch_size = 32
learning_rate = 0.001
log_interval = 500

num_eval_episodes = 10
eval_interval = 1_000

env = GameEnv()
train_env = TFPyEnvironment(env)
eval_env = TFPyEnvironment(env)
fc_layer_params = (40, 40, )
train_step_counter = tf.Variable(0)
epsilon_greedy = tf.compat.v1.train.polynomial_decay(1.0, train_step_counter, decay_episodes, end_learning_rate=min_epsilon)

q_net = QNetwork(train_env.observation_spec(),
                 train_env.action_spec(),
                 fc_layer_params=fc_layer_params)

agent = DdqnAgent(train_env.time_step_spec(),
                  train_env.action_spec(),
                  q_network=q_net,
                  optimizer=Adam(learning_rate=learning_rate),
                  td_errors_loss_fn=common.element_wise_squared_loss,
                  train_step_counter=train_step_counter,
                  epsilon_greedy=epsilon_greedy,
                  gamma=gamma,
                  target_update_period=target_update_period,
                  n_step_update=n_step_update)

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

warmup_driver = DynamicStepDriver(train_env,
                                  random_policy,
                                  observers=[replay_buffer.add_batch],
                                  num_steps=warmup_steps)

collect_driver = DynamicStepDriver(train_env,
                                   agent.collect_policy,
                                   observers=[replay_buffer.add_batch],
                                   num_steps=collect_steps_per_iteration)

warmup_driver.run = common.function(warmup_driver.run)
collect_driver.run = common.function(collect_driver.run)

warmup_driver.run(time_step=None, policy_state=random_policy.get_initial_state(train_env.batch_size))

dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                   sample_batch_size=batch_size,
                                   num_steps=n_step_update + 1).prefetch(3)
iterator = iter(dataset)

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

ts = None
policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
for _ in range(num_iterations):
    ts, policy_state = collect_driver.run(time_step=ts, policy_state=policy_state)

    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(f"{step=}; train loss={train_loss.numpy():.2f}")

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print(f"\x1b[6;30;42m{step=}; {avg_return=}\x1b[0m")
        returns.append(avg_return)

time_step = eval_env.reset()
episode_return = 0.0
fig, ax = plt.subplots()
ims = []
actions = {0: "Up", 1: "Left", 2: "Down", 3: "Right"}
while not time_step.is_last():
    action_step = agent.policy.action(time_step)
    time_step = eval_env.step(action_step.action)
    line = plt.imshow(time_step.observation.numpy().reshape((4, 4)), animated=True)
    episode_return += time_step.reward

    text = f"Action: {actions.get(action_step.action.numpy()[0])}; Score: {episode_return[0]}"
    title = ax.text(0.5, 1.05, text, size=plt.rcParams["axes.titlesize"], ha="center", transform=ax.transAxes)
    ims.append([line, title])

plt.rcParams['animation.ffmpeg_path'] = "D:/ffmpeg/bin/ffmpeg.exe"
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False, repeat=True)
ani.save("2048trained1.gif")  # Can take a few seconds
plt.show()
