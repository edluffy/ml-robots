from isaacgym import gymapi, gymtorch
from utils.isaacgym import BaseEnv, SimState
import random
import torch
import math

class NavEnv(BaseEnv):
    def __init__(self, env_dim=1, max_cycles=1000, show_viewer=True):
        super(NavEnv, self).__init__(show_viewer=show_viewer)
        self.agents_list = ['agent']
        self.state_dims = {'agent': 5}
        self.action_dims = {'agent': 2}
        self.max_cycles = max_cycles

        opts = gymapi.AssetOptions()
        opts.fix_base_link = False
        opts.collapse_fixed_joints = True
        robot_asset = self.gym.load_asset(
            sim=self.sim,
            rootpath='hide_and_seek_robot/urdf',
            filename='robot.urdf',
            options=opts
        )
        robot_props = self.gym.get_asset_dof_properties(robot_asset)
        robot_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        robot_props["stiffness"].fill(0.0)
        robot_props["damping"].fill(1e6)

        goal_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05)

        self.agents = torch.empty(env_dim, dtype=torch.int32, device="cuda")
        self.left_wheels = torch.empty(env_dim, dtype=torch.int32, device="cuda")
        self.right_wheels = torch.empty(env_dim, dtype=torch.int32, device="cuda")
        self.goals = torch.empty(env_dim, dtype=torch.int32, device="cuda")

        shapes_per_env = self.gym.get_asset_rigid_shape_count(robot_asset) + 1
        bodies_per_env = self.gym.get_asset_rigid_body_count(robot_asset) + 1
        envs_per_row = int(math.sqrt(env_dim))
        env_spacing = 0.6
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        for n in range(env_dim):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            self.gym.begin_aggregate(env, bodies_per_env, shapes_per_env, True)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            agent_handle = self.gym.create_actor(
                env=env,
                asset=robot_asset,
                pose=pose,
                name="Agent",
                group=n,
                filter=1,
            )
            self.gym.set_actor_dof_properties(env, agent_handle, robot_props)
            self.gym.enable_actor_dof_force_sensors(env, agent_handle)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            goal_handle = self.gym.create_actor(
                env=env,
                asset=goal_asset,
                pose=pose,
                name="Goal",
                group=0,
                filter=1,
            )
            #color = gymapi.Vec3(random.random(), random.random(), random.random())
            #self.gym.set_rigid_body_color(env, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            self.agents[n] = self.gym.find_actor_index(env, "Agent", gymapi.DOMAIN_SIM)
            self.left_wheels[n] = self.gym.find_actor_dof_index(
                env, agent_handle, "left_wheel", gymapi.DOMAIN_SIM)
            self.right_wheels[n] = self.gym.find_actor_dof_index(
                env, agent_handle, "right_wheel", gymapi.DOMAIN_SIM)
            self.goals[n] = self.gym.find_actor_index(env, "Goal", gymapi.DOMAIN_SIM)

        self.simstate = SimState(self.gym, self.sim)

    def randomise(self):
        #self.simstate.pos[self.agents, :2] = 0.5*torch.randn_like(self.simstate.pos[self.agents, :2])
        #self.simstate.pos[self.goals, :2] = 0.5*torch.randn_like(self.simstate.pos[self.goals, :2])

        self.simstate.pos[self.agents, :2] = 1*torch.ones_like(self.simstate.pos[self.agents, :2])
        self.simstate.pos[self.goals, 0] = -1*torch.ones_like(self.simstate.pos[self.goals, 0])

    def set_actions(self, actions):
        wheel_radius = 0.025
        wheel_sep = 0.098
        vel = 0.5*actions['agent'][0]
        angvel = 5*actions['agent'][1]
        vel, angvel = 0.3, 0
        self.simstate.target_vels[self.left_wheels] = -((vel + angvel * wheel_sep/2) / wheel_radius)
        self.simstate.target_vels[self.right_wheels] = ((vel - angvel * wheel_sep/2) / wheel_radius)

    def get_obs(self, actions):
        agent_x, agent_y = self.simstate.pos[self.agents, :2].squeeze()
        goal_x, goal_y = self.simstate.pos[self.goals, :2].squeeze()

        q1, q2, q3, q0 = self.simstate.rots[self.agents].squeeze()
        agent_yaw = torch.atan2(2*(q3*q0 + q1*q2), -1 + 2*(q0*q0 + q1*q1))
        #goal_yaw = torch.atan2(agent_y, agent_x) - torch.atan2(goal_y, goal_x)
        #goal_yaw = torch.atan2( agent_x*goal_y - agent_y*goal_x, agent_x*goal_x + agent_y*goal_y )
        goal_yaw = torch.atan2(torch.zeros(1, device="cuda") - goal_y, torch.zeros(1, device="cuda") - goal_x)

        if actions is None:
            agent_vel, agent_angvel = 0, 0
        else:
            agent_vel, agent_angvel = actions['agent']

        return {
            'agent': [
                goal_x - agent_x,
                goal_y - agent_y,
                goal_yaw - agent_yaw,
                agent_vel,
                agent_angvel,
            ]
        }

    def get_dones(self, actions, obs):
        return {
            'agent': self.cycles >= self.max_cycles
        }

    def get_rewards(self, actions, obs, dones):
        return {
            'agent': -(obs['agent'][0]**2 + obs['agent'][1]**2)**0.5
        }
