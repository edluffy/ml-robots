from isaacgym import gymapi, gymtorch
import torch

class BaseEnv():
    def __init__(self, show_viewer=False):
        self.show_viewer = show_viewer
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        self.sim = self.gym.create_sim(
            compute_device=0,
            graphics_device=0,
            type=gymapi.SIM_PHYSX,
            params=sim_params,
        )

        if show_viewer == True:
            cam_props = gymapi.CameraProperties()
            cam_props.width = 1920
            cam_props.height = 1080 + 28
            self.viewer = self.gym.create_viewer(self.sim, cam_props)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

        self.cycles = 0

    def step(self, actions):
        self.cycles += 1
        self.set_actions(actions)
        self.simstate.update_targets()

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.simstate.refresh()

        if self.show_viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            #self.gym.sync_frame_time(self.sim)

        info = {}
        obs = self.get_obs(actions)
        dones = self.get_dones(actions, obs)
        rewards = self.get_rewards(actions, obs, dones)

        return obs, rewards, dones, info

    def reset(self):
        self.cycles = 0
        self.simstate.set_initial()
        self.randomise()
        self.simstate.update_states()

        info = {}
        return self.get_obs(None), info

    def close(self):
        if self.show_viewer:
            self.gym.destroy_viewer(self.viewer)
            self.gym.destroy_sim(self.sim)

    def render(self):
        pass

    def seed(self, seed=-1):
        pass

    def randomise(self, env_idxs):
        raise NotImplementedError

    def set_actions(self, actions):
        raise NotImplementedError

    def get_obs(self, actions):
        raise NotImplementedError

    def get_dones(self, actions, obs):
        raise NotImplementedError

    def get_rewards(self, actions, obs, dones):
        raise NotImplementedError

class SimState():
    def __init__(self, gym, sim):
        self.gym = gym
        self.sim = sim
        self.gym.prepare_sim(self.sim)

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)
        self.dof_tensor = gymtorch.wrap_tensor(self._dof_tensor)
        self.force_tensor = gymtorch.wrap_tensor(self._force_tensor)

        self.pos = self.root_tensor[:, 0:3]
        self.rots = self.root_tensor[:, 3:7]
        self.vels = self.root_tensor[:, 7:10]
        self.angvels = self.root_tensor[:, 10:13]
        self.dofs = self.dof_tensor
        self.forces = self.force_tensor

        num_dofs = self.gym.get_sim_dof_count(self.sim)
        self.target_vels = torch.zeros(num_dofs, device="cuda")
        self.target_pos = torch.zeros(num_dofs, device="cuda")

        self.refresh()
        self.initial_root_tensor = self.root_tensor.clone()
        self.initial_dof_tensor = self.dof_tensor.clone()

    def update_targets(self):
        self.gym.set_dof_position_target_tensor(self.sim,
            gymtorch.unwrap_tensor(self.target_pos))
        self.gym.set_dof_velocity_target_tensor(self.sim,
            gymtorch.unwrap_tensor(self.target_vels))

    def refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def set_initial(self):
        self.root_tensor[:] = self.initial_root_tensor
        self.dof_tensor[:] = self.initial_dof_tensor

    def update_states(self):
        self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)
        self.gym.set_dof_state_tensor(self.sim, self._dof_tensor)
