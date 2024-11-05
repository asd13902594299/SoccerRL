

import numpy as np
from gymnasium.utils import EzPickle

from soccer.single_player.core_soccer import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from soccer.single_player.simple_env_soccer import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=0,
        num_obstacles=3,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_tag_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=0, num_obstacles=0):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.width = 1
        world.height = 0.6
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_landmarks = 3
        num_agents = num_adversaries + num_good_agents

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_good_agents else False
            base_name = "red" if agent.adversary else "blue"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True  # if not agent.adversary else False
            agent.movable = True
            agent.silent = True
            agent.size = 0.1
            agent.accel = 5.0  # if not agent.adversary else 0
            agent.max_speed = 10.0
            # increased mass to prevent agents from being pushed back by the ball
            agent.initial_mass = 5

        # add landmarks(goal)
        world.goals = [Landmark() for i in range(2)]
        world.goals[1].name = "goal_blue"
        world.goals[1].color = np.array([0.2, 0.3, 1.0])
        world.goals[1].movable = False
        world.goals[1].collide = False
        world.goals[1].state.p_pos = np.array([-1, 0])
        world.goals[1].size = 0.15
        world.goals[0].name = "goal_red"
        world.goals[0].color = np.array([1.0, 0.3, 0.3])
        world.goals[0].movable = False
        world.goals[0].collide = False
        world.goals[0].state.p_pos = np.array([1, 0])
        world.goals[0].size = 0.15

        # add landmarks(ball)
        world.ball = Landmark()
        world.ball.name = "ball"
        world.ball.color = np.array([0.9, 0.9, 0.2])  # Yellow for visibility
        world.ball.collide = True
        world.ball.movable = True
        world.ball.size = 0.05  # Smaller than agents
        world.ball.initial_mass = 2

        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.85, 0.35, 0.35])
                if agent.adversary
                else np.array([0.35, 0.35, 0.85])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        world.goals[0].color = np.array([0.2, 0.3, 1.0])
        world.goals[1].color = np.array([1.0, 0.3, 0.3])
        # set random initial states
        # world.agents[0].state.p_pos = np_random.uniform(-1, 0, world.dim_p)
        world.agents[0].state.p_pos = np.array([0.6, 0])
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)
        # world.agents[1].state.p_pos = np.array([1, 0])
        # world.agents[1].state.p_vel = np.zeros(world.dim_p)
        # world.agents[1].state.c = np.zeros(world.dim_c)
        # set ball state
        world.ball.state.p_pos = np.array([0, 0])
        world.ball.state.p_vel = np.zeros(world.dim_p)
        # set goal colors and positions
        for goal in world.goals:
            goal.state.p_vel = np.zeros(world.dim_p)  # Goals are immovable

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = (
        #     self.adversary_reward(agent, world)
        #     if agent.adversary
        #     else self.agent_reward(agent, world)
        # )
        main_reward = self.adversary_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x, border=1):
            if x < border-0.1:
                return 0
            if x < border:
                return (x - border-0.1) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        rew = 0

        # 1. Positive reward for kicking the ball
        if self.is_collision(agent, world.ball):
            rew += 10  # Increased positive reward for touching the ball to encourage interaction

        # 2. Negative reward based on distance to the ball (only if not touching the ball)
        distance_to_ball = np.linalg.norm(agent.state.p_pos - world.ball.state.p_pos)
        max_distance = np.sqrt(world.width**2 + world.height**2)  # Diagonal distance of the field
        if not self.is_collision(agent, world.ball):
            rew -= (distance_to_ball / max_distance) * 2  # Smaller negative reward proportional to distance to the ball

        # 3. Positive reward for reducing ball's distance to the opponent's goal
        opponent_goal = world.goals[0] if agent.adversary else world.goals[1]
        distance_to_goal = np.linalg.norm(world.ball.state.p_pos - opponent_goal.state.p_pos)
        previous_distance_to_goal = getattr(agent, "previous_distance_to_goal", None)
        if previous_distance_to_goal is not None and abs(distance_to_goal - previous_distance_to_goal) > 1e-3:
            rew += 2.5  # Reward for getting the ball closer to the goal
        agent.previous_distance_to_goal = distance_to_goal

        # 4. Positive reward for scoring a goal
        if distance_to_goal < opponent_goal.size:
            rew += 200  # Huge positive reward for getting the ball into the goal

        # 5. Penalty for inactivity (optional, to encourage movement)
        if np.linalg.norm(agent.state.p_vel) < 1e-3:
            rew -= 1.0  # Smaller penalty for standing still to prevent inactivity

        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_vel = []
        for entity in (world.goals + [world.ball]):
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_vel.append(entity.state.p_vel)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + entity_vel
            + other_pos
            + other_vel
        )
