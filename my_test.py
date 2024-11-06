from pettingzoo.test import api_test
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.mpe import simple_adversary_v3

env = simple_adversary_v3.env(max_cycles=100, render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(f"{agent}, Reward: {reward}")

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()