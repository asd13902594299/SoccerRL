from pettingzoo.test import api_test
from soccer.single_player import soccer_single_player
from soccer.simple_two_players import soccer_simple_2player
from soccer.simple_four_players import soccer_simple_4player

env = soccer_simple_4player.env(max_cycles=1000, render_mode="human")
env.reset()

print(f'action_spaces: {env.action_spaces}')

group_rewards = []

for idx, agent in enumerate(env.agent_iter()):
    observation, reward, termination, truncation, info = env.last()

    # Append reward information for the agent, with simplified naming
    agent_name_short = agent.replace("blue_", "b").replace("red_", "r")
    group_rewards.append(f"{agent_name_short}: {reward:3.1f}")

    # Print rewards for four agents on the same line
    if len(group_rewards) == 4:
        print(" , ".join(group_rewards))
        group_rewards = []

    if termination or truncation:
        action = None
    else:
        # Insert your policy here
        action = env.action_space(agent).sample()
        if agent == "red_0":
            action = 3
        else:
            action = 0

    env.step(action)
env.close()
