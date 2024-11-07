from soccer.simple_four_players import soccer_simple_4player
import supersuit as ss
from stable_baselines3 import PPO


def train():
    env = soccer_simple_4player.parallel_env(max_cycles=175, render_mode=None)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 8, num_cpus=8, base_class="stable_baselines3")

    model = PPO("MlpPolicy", env, verbose=1, device="cuda",
                learning_rate=0.0001, ent_coef=0.01, gamma=0.97, batch_size=256)
    # model.learn(total_timesteps=4200000)
    # model.learn(total_timesteps=3100000)
    model.learn(total_timesteps=1500000)
    # model.learn(total_timesteps=1048576)
    # model.learn(total_timesteps=700000)
    # model.learn(total_timesteps=524288)
    # model.learn(total_timesteps=300000)
    # model.learn(total_timesteps=262144)
    # model.learn(total_timesteps=196608)
    # model.learn(total_timesteps=163840)
    # model.learn(total_timesteps=131072)
    # model.learn(total_timesteps=100000)
    # model.learn(total_timesteps=65536)
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
    # train()
    eval()