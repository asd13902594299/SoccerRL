from pettingzoo.test import api_test
from soccer.single_player import soccer

env = soccer.env(max_cycles=1000, render_mode="human")
env.reset()

print(f'action_spaces: {env.action_spaces}')    

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    # if agent == "red_0":
    print(f"{agent}, Reward: {reward}")

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        # action = env.action_space(agent).sample()
        # print(f"Action: {action}")
        action = 1

    env.step(action)
env.close()