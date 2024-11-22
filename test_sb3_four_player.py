from soccer.simple_four_players import soccer_simple_4player
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import torch
import argparse
import csv


# Custom callback class, extends BaseCallback from stable_baselines3
# This is so we can override the _on_rollout_end method to log loss values to a file
class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.csv_file = open('test_sb3_two_player_logs.csv', 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        # Write the header row
        self.writer.writerow(['timesteps', 'loss', 'policy_gradient_loss', 'value_loss', 'entropy_loss', 'approx_kl', 'clip_fraction', 'explained_variance'])

    def _on_step(self) -> bool:
        # Need this as an abstract method to avoid errors
        # Check ...\stable_baselines3\common\callbacks.py
        return True
    
    def _on_rollout_end(self):
        # Get the current logs
        logs = self.model.logger.name_to_value
        # Get the desired metrics
        timestep = self.num_timesteps
        loss = logs.get('train/loss')
        pg_loss = logs.get('train/policy_gradient_loss')
        value_loss = logs.get('train/value_loss')
        entropy_loss = logs.get('train/entropy_loss')
        approx_kl = logs.get('train/approx_kl')
        clip_fraction = logs.get('train/clip_fraction')
        explained_variance = logs.get('train/explained_variance')

        # Write the metrics to CSV
        self.writer.writerow([timestep, loss, pg_loss, value_loss, entropy_loss, approx_kl, clip_fraction, explained_variance])
        self.csv_file.flush()

    def _on_training_end(self):
        self.csv_file.close()

def train():
    env = soccer_simple_4player.parallel_env(max_cycles=175, render_mode=None)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 8, num_cpus=8, base_class="stable_baselines3")

    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"

    print(f"Device: {device}, Device name: {device_name}")

    # Create an instance of the callback for logging
    logging_callback = LoggingCallback()

    model = PPO("MlpPolicy", env, verbose=1, device=device,
                learning_rate=0.0001, ent_coef=0.01, gamma=0.97, batch_size=256)
    # model.learn(total_timesteps=4200000, callback=logging_callback)
    # model.learn(total_timesteps=3100000, callback=logging_callback)
    model.learn(total_timesteps=1500000, callback=logging_callback)
    # model.learn(total_timesteps=1048576, callback=logging_callback)
    # model.learn(total_timesteps=700000, callback=logging_callback)
    # model.learn(total_timesteps=524288, callback=logging_callback)
    # model.learn(total_timesteps=300000, callback=logging_callback)
    # model.learn(total_timesteps=262144, callback=logging_callback)
    # model.learn(total_timesteps=196608, callback=logging_callback)
    # model.learn(total_timesteps=163840, callback=logging_callback)
    # model.learn(total_timesteps=131072, callback=logging_callback)
    # model.learn(total_timesteps=100000, callback=logging_callback)
    # model.learn(total_timesteps=65536, callback=logging_callback
    model.save("simple_four_player")

    env.close()


def eval():
    env = soccer_simple_4player.env(
        max_cycles=400, render_mode="human")

    model = PPO.load("simple_four_player", device="cuda")
    obs = env.reset()
    print(env.possible_agents)
    rewards = {agent: 0 for agent in env.possible_agents}

    # List to keep track of rewards for all agents
    group_rewards = []

    for idx, agent in enumerate(env.agent_iter()):
        obs, reward, termination, truncation, info = env.last()

        # Append reward information for the agent, with simplified naming
        agent_name_short = agent.replace("blue_", "b").replace("red_", "r")
        group_rewards.append(f"{agent_name_short}, : {reward:3.1f}")

        # Print rewards for four agents on the same line
        if len(group_rewards) == 4:
            print(" , ".join(group_rewards))
            group_rewards = []

        for a in env.agents:
            rewards[a] += env.rewards[a]

        if termination or truncation:
            break
        else:
            act = model.predict(obs, deterministic=True)[0]
        env.step(act)

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        eval()