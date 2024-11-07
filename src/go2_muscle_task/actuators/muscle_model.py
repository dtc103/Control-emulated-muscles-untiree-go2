import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

#from legged_gym.utils.muscle_validation import DataStorage

torch.set_default_dtype(torch.float32)


class MuscleModel:
    def __init__(self, muscle_params, action_tensor, nenvironments, options):
        self.device = "cuda"
        self.nactioncount = 24
        self.nenvironments = nenvironments

        for k, v in muscle_params.items():
            setattr(self, k, v)

        self.phi_min = (
            torch.ones(
                (nenvironments, 12),
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            * self.phi_min
        )
        self.phi_max = (
            torch.ones(
                (nenvironments, 12),
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            * self.phi_max
        )
        self.activation_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_dot_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.force_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_1_tensor = torch.zeros(
            (self.nenvironments, 12),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_2_tensor = torch.zeros(
            (self.nenvironments, 12),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.moment, self.lce_ref = self.compute_parametrization(action_tensor)

        # TODO: remove some exceptions and automaticly set required options
        self.episode_steps = 0
        #self.eps_counter = 2
        #self.total_plot_counter = 0
        #self.storedata = options.STORE_DATA
        #if self.storedata:
            #self.datastorage = DataStorage(muscle_params, options)
        #self.writeToCSV = options.WRITE_MUSCLE_DATA_TO_CSV
        # if self.writeToCSV:
        #     if not self.storedata:
        #         raise Exception("Data storing is required in order to be able to save data into csv")
        #     self.writer = csv.writer(open("100-zeroactions.csv", "w", newline=""))
        # self.visualize_muscle_relations = options.VISUALIZATION_MUSCLE_RELATION
        # self.visualize_joint_characteristics = options.VISUALIZATION_JOINT_CHARACTERISTICS
        # self.validation_experiment_muscle_actuator_range = options.VALIDATION_EXPERIMENT_MUSCLE_ACTUATOR_RANGE
        # self.validation_experiment_muscle_action_change = options.VALIDATION_EXPERIMENT_MUSCLE_ACTION_CHANGE
        # if self.validation_experiment_muscle_actuator_range and self.validation_experiment_muscle_action_change:
        #     raise Exception("Can't use both validation experiments for muscles at the same time")
        # if self.validation_experiment_muscle_actuator_range or self.validation_experiment_muscle_action_change:
        #     if not self.storedata or not self.visualize_muscle_relations:
        #         raise Exception("Validation experiment requires data storage option and muscle relation visualization")

    def analyse_muscle(self):
        # TODO: call for analyse function can be activated using configuration file not muscle usage
        # TODO: define properly plotting for defined time step of episodes interval

        # creates plots for muscle validation analysis
        # if self.storedata:
        #     # plot each episode (24 steps -> 24 calls of analyse_muscle function)
        #     if self.episode_steps == 24:
        #         if self.visualize_muscle_relations:
        #             fig, axs = plt.subplots(7, 2, figsize=(38, 32))
        #             # fig, axs = plt.subplots(6, 2)
        #             # fig.tight_layout()
        #             self.datastorage.plot_muscle_relations(axs, self.eps_counter, self.episode_steps)

        #         if self.visualize_joint_characteristics:
        #             num_joints = 2
        #             fig, axs = plt.subplots(num_joints, 3, figsize=(16, 10))
        #             fig.suptitle("Joint characteristics", fontsize=16)
        #             self.datastorage.plot_joint_characteristics(axs, num_joints)

        #         if self.total_plot_counter == 10:
        #             self.storedata = False
        #             self.visualize_muscle_relations = False
        #             self.visualize_joint_characteristics = False

        #         self.episode_steps = 0
        #         self.total_plot_counter += 1
        #         self.eps_counter += 1
        #     self.episode_steps += 1
        pass

    def FL(self, lce: torch.Tensor) -> torch.Tensor:
        """
        Force length
        """
        length = lce
        b1 = self.bump(length, self.lmin, 1, self.lmax)
        b2 = self.bump(length, self.lmin, 0.5 * (self.lmin + 0.95), 0.95)
        bump_res = b1 + 0.15 * b2
        return bump_res

    def bump(self, length: torch.Tensor, A: float, mid: float, B: float) -> torch.Tensor:
        """
        skewed bump function: quadratic spline
        Input:
            :length: tensor of muscle lengths [Nenv, Nactuator]
            :A: scalar
            :mid: scalar
            :B: scalar

        Returns:
            :torch.Tensor: contains FV result [Nenv, Nactuator]
        """

        left = 0.5 * (A + mid)
        right = 0.5 * (mid + B)
        # Order of assignment needs to be inverse to the if-else-clause case
        bump_result = torch.ones_like(
            length,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        x = (B - length) / (B - right)
        bump_result = 0.5 * x * x

        x = (length - mid) / (right - mid)
        bump_result = torch.where(length < right, 1 - 0.5 * x * x, bump_result)

        x = (mid - length) / (mid - left)
        bump_result = torch.where(length < mid, 1 - 0.5 * x * x, bump_result)

        x = (length - A) / (left - A)
        bump_result = torch.where((length < left) & (length > A), 0.5 * x * x, bump_result)

        bump_result = torch.where(
            torch.logical_or((length <= A), (length >= B)),
            torch.tensor([0], dtype=torch.float32, device=self.device),
            bump_result,
        )
        return bump_result

    def FV(self, lce_dot: torch.Tensor) -> torch.Tensor:
        """
        Force velocity
        Input:
            :lce_dot: tensor of muscle velocities [Nenv, Nactuator]
        """
        c = self.fvmax - 1
        velocity = lce_dot

        eff_vel = torch.div(velocity, self.vmax)

        c_result = torch.zeros_like(
            eff_vel,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        c_result = torch.where(
            (eff_vel > c),
            torch.tensor([self.fvmax], dtype=torch.float32, device=self.device),
            c_result,
        )

        x = torch.sub(
            self.fvmax,
            torch.div(torch.mul(torch.sub(c, eff_vel), torch.sub(c, eff_vel)), c),
        )
        c_result = torch.where(eff_vel <= c, x, c_result)

        x = torch.mul(torch.add(eff_vel, 1), torch.add(eff_vel, 1))
        c_result = torch.where(eff_vel <= 0, x, c_result)

        c_result = torch.where(
            (eff_vel < -1),
            torch.tensor([0], dtype=torch.float32, device=self.device),
            c_result,
        )

        return c_result

    def FP(self, lce: torch.Tensor) -> torch.Tensor:
        """
        Force passive
        Inputs:
            :lce: muscle lengths [Nenv, Nactuator]
        return :fp_result: passive_force [Nenv, Nactuator]
        """
        b = 0.5 * (self.lmax + 1)

        # Order of assignment needs to be inverse to the if-else-clause case
        ## method to prevent
        cond_2_tmp = torch.div(torch.sub(lce, 1), (b - 1))
        cond_2 = torch.mul(
            torch.mul(torch.mul(cond_2_tmp, cond_2_tmp), cond_2_tmp),
            (0.25 * self.fpmax),
        )

        cond_3_tmp = torch.div(torch.sub(lce, b), (b - 1))
        cond_3 = torch.mul(torch.add(torch.mul(cond_3_tmp, 3), 1), (0.25 * self.fpmax))
        ##### copy based on condition the correct output into new tensor
        c_result = torch.zeros_like(lce, dtype=torch.float32, device=self.device, requires_grad=False)
        c_result = torch.where(lce <= b, cond_2, c_result)
        c_result = torch.where(
            lce <= 1,
            torch.tensor([0], dtype=torch.float32, device=self.device),
            c_result,
        )
        c_result = torch.where(lce > b, cond_3, c_result)

        return c_result

    def compute_parametrization(self, action_tensor):
        """
        Find parameters for muscle length computation.
        This should really only be done once...

        We compute them as one long vector now.
        """
        moment = torch.zeros_like(action_tensor)
        moment[:, : int(moment.shape[1] // 2)] = (self.lce_max - self.lce_min + self.eps) / (self.phi_max - self.phi_min + self.eps)
        moment[:, int(moment.shape[1] // 2) :] = (self.lce_max - self.lce_min + self.eps) / (self.phi_min - self.phi_max + self.eps)
        lce_ref = torch.zeros_like(action_tensor)
        lce_ref[:, : int(lce_ref.shape[1] // 2)] = self.lce_min - moment[:, : int(moment.shape[1] // 2)] * self.phi_min
        lce_ref[:, int(lce_ref.shape[1] // 2) :] = self.lce_min - moment[:, int(moment.shape[1] // 2) :] * self.phi_max
        return moment, lce_ref

    def compute_virtual_lengths(self, actuator_pos: torch.Tensor) -> None:
        """
        Compute muscle fiber lengths l_ce depending on joint angle
        Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only
        slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
        by looking at every single joint type.

        self.lce_x_tensor contains copy of given actions (they represent the actuator position)
        """
        # Repeat position tensor twice, because both muscles are computed from the same actuator position
        # the operation is NOT applied in-place to the original tensor, only the result is repeated.
        self.lce_tensor = torch.add(torch.mul(actuator_pos.repeat(1, 2), self.moment), self.lce_ref)

    def get_peak_force(self, actuator_vel):
        """
        acceleration from unit force in qpos0 is contained within self.dof_vel
        """
        ### at the moment not used because solo8/12 have the same actuators and therefore no requirement to be dynamic
        raise NotImplementedError
        force = 5000
        scale = 200

        c_result = torch.zeros_like(
            actuator_vel,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        for i, c_vel in enumerate(actuator_vel):
            if (force + 1) < 0.01:
                c_result[i] = torch.div(scale, c_vel)
            else:
                c_result[i] = force
        return c_result

    def get_vel(self, moment, actuator_vel: torch.Tensor) -> torch.Tensor:
        """
        For muscle 1, the joint angle increases if it pulls. This means
        that the joint and the muscle velocity have opposite signs. But this is already
        included in the value of the moment arm. So we don't need if/else branches here.
        Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only
        slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
        by looking at every single joint type.
        """
        return torch.mul(actuator_vel.repeat(1, 2), moment)

    def activ_dyn(self, actions: torch.Tensor) -> None:
        """
        Activity and controls have to be written inside userdata. Assume
        two virtual muscles per real mujoco actuator and let's roll.
        """
        sim_timestep = 0.005
        self.activation_tensor = 100 * (actions - self.activation_tensor) * sim_timestep + self.activation_tensor
        self.activation_tensor = torch.clip(self.activation_tensor, 0, 1)
        # self.activation_tensor = torch.mul(torch.mul(torch.sub(actions,self.activation_tensor), 100), torch.add(self.activation_tensor, sim_timestep))

    def compute_moment(self, actions, actuator_vel, lce_1, lce_2):
        """
        Joint moments are computed from muscle contractions and then returned
        """

        self.lce_dot_tensor = self.get_vel(self.moment, actuator_vel)
        lce_dot = self.lce_dot_tensor
        lce_tensor = self.lce_tensor

        FL = self.FL(lce_tensor)
        FV = self.FV(lce_dot)
        FP = self.FP(lce_tensor)

        self.force_tensor = torch.add(torch.mul(torch.mul(FL, FV), self.activation_tensor), FP)
        # peak_force = self.get_peak_force(actuator_velocity)
        F = torch.mul(self.force_tensor, self.peak_force)
        torque = F * self.moment

        # if self.storedata:
        #     self.datastorage.FL_1 = np.append(self.datastorage.FL_1, FL[:, 1].tolist()[0])
        #     self.datastorage.FL_2 = np.append(self.datastorage.FL_2, FL[:, 9].tolist()[0])
        #     self.datastorage.FV_1 = np.append(self.datastorage.FV_1, FV[:, 1].tolist()[0])
        #     self.datastorage.FV_2 = np.append(self.datastorage.FV_2, FV[:, 9].tolist()[0])
        #     self.datastorage.FP_1 = np.append(self.datastorage.FP_1, FP[:, 1].tolist()[0])
        #     self.datastorage.FP_2 = np.append(self.datastorage.FP_2, FP[:, 9].tolist()[0])

        #     self.datastorage.LCE_1 = np.append(self.datastorage.LCE_1, self.lce_tensor[:, 1].tolist()[0])
        #     self.datastorage.LCE_2 = np.append(self.datastorage.LCE_2, self.lce_tensor[:, 9].tolist()[0])
        #     self.datastorage.LCE_DOT_1 = np.append(self.datastorage.LCE_DOT_1, lce_dot[:, 1].tolist()[0])
        #     self.datastorage.LCE_DOT_2 = np.append(self.datastorage.LCE_DOT_2, lce_dot[:, 9].tolist()[0])

        #     self.datastorage.activations1 = np.append(self.datastorage.activations1, self.activation_tensor[:, 1].tolist()[0])
        #     self.datastorage.activations2 = np.append(self.datastorage.activations2, self.activation_tensor[:, 9].tolist()[0])
        #     self.datastorage.actions1 = np.append(self.datastorage.actions1, actions[:, 1].tolist()[0])
        #     self.datastorage.actions2 = np.append(self.datastorage.actions2, actions[:, 9].tolist()[0])

        #     if self.writeToCSV:
        #         self.writer.writerow(
        #             [
        #                 actuator_vel.tolist()[0][0],
        #                 FL[:, 0].tolist()[0],
        #                 FL[:, 8].tolist()[0],
        #                 FV[:, 0].tolist()[0],
        #                 FV[:, 8].tolist()[0],
        #                 FP[:, 0].tolist()[0],
        #                 FP[:, 8].tolist()[0],
        #                 self.lce_tensor[:, 0].tolist()[0],
        #                 self.lce_tensor[:, 8].tolist()[0],
        #                 lce_dot[:, 0].tolist()[0],
        #                 lce_dot[:, 8].tolist()[0],
        #             ]
        #         )

        return torch.sum(
            torch.reshape(torque, (self.nenvironments, 2, self.nactioncount // 2)),
            axis=-2,
        )

    def compute_torques(self, actuator_pos, actuator_vel, actions):
        """
        actuator_pos: Current position of actuator
        """
        # if self.validation_experiment_muscle_actuator_range:
        #     (
        #         actions,
        #         actuator_pos,
        #         actuator_vel,
        #     ) = self.datastorage.visualize_muscle_actuator_ranges(actions, actuator_pos)

        with torch.torch.no_grad():
            actions = torch.clip(actions, 0, 1)

            # activity funciton for muscle activation
            self.activ_dyn(actions)

            # update virtual lengths
            self.compute_virtual_lengths(actuator_pos)

            # compute moments
            moment = self.compute_moment(actions, actuator_vel, self.lce_1_tensor, self.lce_2_tensor)

            # if self.validation_experiment_muscle_action_change:
            #     moment = self.datastorage.visualize_max_dof_pos(moment)

            # if self.storedata:
            #     self.datastorage.joint_positions.append(actuator_pos.tolist()[0])
            #     self.datastorage.joint_velocities.append(actuator_vel.tolist()[0])
            #     self.datastorage.joint_torques.append(moment.tolist()[0])
            return moment
