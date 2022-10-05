import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        wall_pos = 1.3
        world.walls = [Wall(orient='H', axis_pos=-wall_pos), Wall(orient='H', axis_pos=wall_pos)]
        world.walls.append(Wall(orient='V', axis_pos=-wall_pos))
        world.walls.append(Wall(orient='V', axis_pos=wall_pos))
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.07 if agent.adversary else 0.07
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.0, 0.0, 1.0]) if not agent.adversary else np.array([1.0, 0.0, 0.0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


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
        dist_min = 2 * agent1.size + 2 * agent2.size

        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        num_collisions = 0
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    num_collisions += 1
        if num_collisions == 0:
            rew += 0
        if num_collisions == 1:
            rew += 1
        if num_collisions > 1:
            rew -= 1
        if agent.state.p_pos[0] > 0.99 or agent.state.p_pos[0] < -0.99 or agent.state.p_pos[1] < -0.99 or agent.state.p_pos[1] > 0.99:
            rew -=10
        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for adv in adversaries:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        num_collisions = 0
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        num_collisions += 1
        if num_collisions == 0:
            return rew
        if num_collisions == 1:
            return rew - 1
        if num_collisions > 1:
            return rew + 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        comm = []
        if agent in self.good_agents(world):
            entity_pos = []
            for entity in world.landmarks:
                if not entity.boundary:
                    dis = entity.state.p_pos - agent.state.p_pos
                    if abs(dis[0]) > 0.5 or abs(dis[1]) > 0.5:
                        entity_pos.append([-5.0, -5.0])
                    else:
                        entity_pos.append(entity.state.p_pos - agent.state.p_pos)

            other_pos = []
            other_vel = []
            for other in world.agents:
                if other is agent: continue
                comm.append(other.state.c)
                dis = other.state.p_pos - agent.state.p_pos
                if abs(dis[0]) > 0.5 or abs(dis[1]) > 0.5:
                    other_pos.append([-5.0, -5.0])
                else:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        else:
            entity_pos = []
            for entity in world.landmarks:
                if not entity.boundary:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            other_pos = []
            other_vel = []
            for other in world.agents:
                if other is agent: continue
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
     
