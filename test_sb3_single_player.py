from soccer.single_player import soccer
import supersuit as ss
from stable_baselines3 import PPO


def train():
    env = soccer.parallel_env(max_cycles=100, render_mode=None)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 8, num_cpus=1, base_class="stable_baselines3")

    model = PPO("MlpPolicy", env, verbose=1, device="cuda", learning_rate=0.0003)
    # model.learn(total_timesteps=1048576)
    # model.learn(total_timesteps=524288)
    # model.learn(total_timesteps=300000)
    model.learn(total_timesteps=262144)
    # model.learn(total_timesteps=196608)
    # model.learn(total_timesteps=163840) 
    # model.learn(total_timesteps=131072) 
    # model.learn(total_timesteps=100000) 
    # model.learn(total_timesteps=65536) 
    model.save("single_player")

    env.close()

def eval():
    env = soccer.env(
        max_cycles=100, render_mode="human")

    model = PPO.load("single_player", device="cuda")
    obs = env.reset()
    print(env.possible_agents)
    rewards = {agent: 0 for agent in env.possible_agents}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        print(f"Agent: {agent}, Reward: {reward}")
        for a in env.agents:
            rewards[a] += env.rewards[a]

        if termination or truncation:
            break
        else:
            act = model.predict(obs, deterministic=True)[0]
            print(f"Agent: {agent}, Action: {act}")
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
    train()
    eval()
