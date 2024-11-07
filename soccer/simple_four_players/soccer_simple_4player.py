

import numpy as np
from gymnasium.utils import EzPickle

from soccer.simple_four_players.core_soccer import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from soccer.simple_four_players.simple_env_soccer import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=2,
        num_adversaries=2,
        num_obstacles=0,
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
        self.metadata["name"] = "simple_4players_soccer"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=2, num_adversaries=2, num_obstacles=0):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.width = 1
        world.height = 0.6
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = False if i < num_good_agents else True
            base_name = "red" if agent.adversary else "blue"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.movable = True
            agent.silent = True
            agent.size = 0.1
            agent.accel = 3.0
            agent.max_speed = 5.0
            # increased mass to prevent agents from being pushed back by the ball
            agent.initial_mass = 7
            agent.size = 0.05

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
        world.ball.size = 0.03  # Smaller than agents
        world.ball.initial_mass = 3

        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.85, 0.35, 0.35])
                if agent.adversary
                else np.array([0.35, 0.35, 0.85])
            )

        world.goals[0].color = np.array([0.2, 0.3, 1.0])
        world.goals[1].color = np.array([1.0, 0.3, 0.3])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([
                np_random.uniform(-world.width, 0),
                np_random.uniform(-world.height, world.height),
            ]) if agent.adversary else np.array([
                np_random.uniform(0, world.width),
                np_random.uniform(-world.height, world.height),
            ])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        # world.agents[2].state.p_pos = np.array([0, 0.6])
        # world.agents[3].state.p_pos = np.array([0, -0.6])
        
        # set ball state
        world.ball.state.p_pos = np.array([0, 0])
        world.ball.state.p_vel = np.zeros(world.dim_p)

        # set goal colors and positions
        for goal in world.goals:
            goal.state.p_vel = np.zeros(world.dim_p)  # Goals are immovable

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents in same team as agent
    def get_wholeteam(self, agent, world):
        return [a for a in world.agents if a.adversary == agent.adversary]

    # return all agents in same team as agent(excluding agent)
    def get_teammates(self, agent, world):
        return [a for a in world.agents if a is not agent and a.adversary == agent.adversary]
    # return all agents in the opposite team

    def get_opponents(self, agent, world):
        return [a for a in world.agents if a is not agent and a.adversary != agent.adversary]

    # return the angle between two vectors (in radians)
    def angle_between(self, v1, v2):
        # Calculate the cosine of the angle between the vectors
        cos_theta = np.dot(v1, v2) / (
            np.linalg.norm(v1) *
            np.linalg.norm(v2)
        )
        # Clip to handle floating point errors
        return np.arccos(cos_theta)

    def reward(self, agent, world):
        rew = 0

        # 1. Reward for kicking the ball
        for a in world.agents:
            if self.is_collision(a, world.ball):
                world.last_ball_touch = a.name
                if world.last_ball_touch.split("_")[0] == agent.name.split("_")[0]:
                    rew += 5
                else:
                    rew -= 3

        # 2. Negative reward based on distance to the ball (only if not touching the ball)
        distance_to_ball = np.linalg.norm(
            agent.state.p_pos - world.ball.state.p_pos)
        # Diagonal distance of the field
        max_distance = np.sqrt(world.width**2 + world.height**2)
        if not self.is_collision(agent, world.ball):
            # Smaller negative reward proportional to distance to the ball
            rew -= (distance_to_ball / max_distance) * 2

        # 3. Positive reward for reducing ball's distance to the opponent's goal
        opponent_goal = world.goals[0] if agent.adversary else world.goals[1]
        distance_to_goal = np.linalg.norm(
            world.ball.state.p_pos - opponent_goal.state.p_pos)
        for a in self.get_wholeteam(agent, world):
            previous_distance_to_goal = getattr(
                a, "previous_distance_to_goal", None)
            # Reward for reducing the distance to the opponent's goal, with greater weight for significant progress
            if previous_distance_to_goal is not None:
                distance_difference = previous_distance_to_goal - distance_to_goal
                if distance_difference > 0:
                    rew += 1.5 * distance_difference  # Reward scales with the amount of progress made
                    # print(f'distance_difference: {distance_difference}')

            a.previous_distance_to_goal = distance_to_goal

        # 4. Positive reward for scoring a goal
        if distance_to_goal < opponent_goal.size:
            rew += 500  # Huge positive reward for getting the ball into the goal

        # 5. Negative reward if the ball is closer to the agent's own goal
        own_goal = world.goals[1] if agent.adversary else world.goals[0]
        distance_to_own_goal = np.linalg.norm(
            world.ball.state.p_pos - own_goal.state.p_pos)
        # Larger negative reward if the ball is closer to own goal
        rew -= (1 - (distance_to_own_goal / max_distance)) * 2
        # print(f'distance_to_own_goal{agent.name}: {distance_to_own_goal / max_distance}')

        # 6. Huge penalty if the ball goes into the agent's own goal
        if distance_to_own_goal < own_goal.size:
            rew -= 500  # Increased penalty for getting the ball into own goal

        # 7. Penalty for inactivity to encourage movement
        if np.linalg.norm(agent.state.p_vel) < 1e-3:
            rew -= 2.0  # Increased penalty for standing still to prevent inactivity

        # 8. Penalty for ball going out of the field. Only penalty the agent who kicked the ball out
        if ((abs(world.ball.state.p_pos[0]) > world.width or abs(world.ball.state.p_pos[1]) > world.height) and
            world.last_ball_touch.split("_")[0] == agent.name.split("_")[0]
            ):
            rew -= 20  # Increased negative reward for the ball going out of the field

        # 9. Positive reward for kicking the ball towards the goal direction (only if the ball is moving)
        ball_velocity = np.linalg.norm(world.ball.state.p_vel)
        # Ensure the ball is moving
        if world.last_ball_touch is None:
            pass
        elif ball_velocity > 1e-3 and world.last_ball_touch.split("_")[0] == agent.name.split("_")[0]:
            ball_to_goal_vector = opponent_goal.state.p_pos - world.ball.state.p_pos
            agent_to_ball_vector = world.ball.state.p_pos - agent.state.p_pos
            ball_to_opponent_vectors = [
                opponent.state.p_pos - world.ball.state.p_pos for opponent in self.get_opponents(agent, world)]
            team_to_ball_vectors = [
                world.ball.state.p_pos - teammate.state.p_pos for teammate in self.get_wholeteam(agent, world)]

            # Positive reward for kicking the ball towards the goal (whole team reward)
            for team_to_ball_v in team_to_ball_vectors:
                angle = self.angle_between(
                    -ball_to_goal_vector, team_to_ball_v)
                # Reward if the angle is within 0 to 30 degrees
                if np.deg2rad(150) <= angle <= np.deg2rad(180):
                    # Increased reward for pushing the ball towards the opponent's goal
                    rew += ((np.rad2deg(angle)-150)/3)*1.15

            # Negative reward for kicking the ball towards the opponent (single agent reward)
            for ball_to_opponent_v in ball_to_opponent_vectors:
                angle_opponent = self.angle_between(
                    ball_to_opponent_v, -agent_to_ball_vector)
                if np.deg2rad(150) <= angle_opponent <= np.deg2rad(180):
                    # Negative reward for kicking the ball towards the opponent
                    rew -= (np.rad2deg(angle_opponent)-150)*0.9

            # Positive reward for kicking the ball towards the teammate (single agent reward)
            for teammate in self.get_teammates(agent, world):
                team_to_ball_v = world.ball.state.p_pos - teammate.state.p_pos
                angle_team = self.angle_between(
                    -team_to_ball_v, -agent_to_ball_vector)
                if np.deg2rad(150) <= angle_team <= np.deg2rad(180):
                    # Positive reward for kicking the ball towards the teammate
                    rew += ((np.rad2deg(angle_team)-150)/3)*0.95

        # 10. Penalty if the ball is not moving
        if ball_velocity <= 1e-3:
            rew -= 2  # Penalty for the ball being stationary to encourage the agent to keep it moving

        # 11. Penalty for collision with the teammate
        for teammate in self.get_teammates(agent, world):
            if self.is_collision(agent, teammate):
                rew -= 1

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_vel = []

        # reference position: agent_goal, ball_goal
        opponent_goal = world.goals[0] if agent.adversary else world.goals[1]
        own_goal = world.goals[1] if agent.adversary else world.goals[0]
        entity_pos.append(opponent_goal.state.p_pos - world.ball.state.p_pos)
        entity_pos.append(opponent_goal.state.p_pos - agent.state.p_pos)
        entity_pos.append(own_goal.state.p_pos - world.ball.state.p_pos)
        entity_pos.append(own_goal.state.p_pos - agent.state.p_pos)

        # ball's relative position and velocity
        entity_pos.append(world.ball.state.p_pos - agent.state.p_pos)
        entity_vel.append(world.ball.state.p_vel)

        # Add teammates' position and velocity
        for teammate in self.get_wholeteam(agent, world):
            # agent itself is included in get_wholeteam
            entity_pos.append(teammate.state.p_pos - agent.state.p_pos)
            entity_vel.append(teammate.state.p_vel)

        # Add opponents' position and velocity
        for opponent in self.get_opponents(agent, world):
            entity_pos.append(opponent.state.p_pos - agent.state.p_pos)
            entity_vel.append(opponent.state.p_vel)

        return np.concatenate(
            entity_pos
            + entity_vel
        )
