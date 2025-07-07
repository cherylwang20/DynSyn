""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Huiyi Wang (huiyi.wang@mail.mcgill.ca), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
Created for imitation learning reward design
================================================= """

import collections
from myosuite.utils import gym
import numpy as np
import os
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
import pandas as pd
import scipy

class WalkEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = [
        'qpos',
        'height',
        'qvel',
        'com_vel',
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "ref_vel": 2.0,
        #"vel_rew": 1.0,
        "root_err": 5.0,
        "done": 0,
        "torso": 1,
        "ref_qpos": 5.0,
        "act_mag": -1,
        "sparse": 1,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               min_height = 0.75,
               max_rot = 1.8,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel= 1.8, #change later to match the data, 1.8m/s
               target_rot = None,
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.MAX_ERROR = 1.05
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = -target_y_vel
        self.target_rot = target_rot
        self.error_qpos = 0
        self.error_qvel = 0
        self.steps = 0
        self.set_joints = ['hip_flexion_r',	'hip_adduction_r',
                                                    'hip_rotation_r',	'knee_angle_r',	
                                                    'ankle_angle_r',   'mtp_angle_r', 'hip_flexion_l', 
                                                    'ankle_angle_r',   'mtp_angle_r', 'hip_flexion_l', 
                                                    'hip_adduction_l',	'hip_rotation_l',
                                                    'knee_angle_l',	'ankle_angle_l', 'mtp_angle_l',
                                                    'flex_extension',	'lat_bending',	'axial_rotation']
        self.ref_data = self.read_npy()
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
        obs_dict['height'] = np.array([self._get_height()]).copy()
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        #vel_reward = self._get_vel_reward()
        self.error_qpos, self.error_qvel, root_err = self._get_joint_angle_rew()
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        torso_up = abs(self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('flex_extension')]])

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('ref_vel', np.exp(- 1 * self.error_qvel)),
            ('ref_qpos', np.exp(- 200 * self.error_qpos)),
            ('root_err', np.exp(- 200 * root_err)),
            ('act_mag', act_mag[0][0]),
            ('torso', np.exp(- 5 * torso_up)),
            # Must keys
            ('sparse',  self.error_qpos <= 1),
            ('solved',    None),
            ('done',  self._get_done()),
        ))

        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def _get_joint_angle_rew(self):
        """
        Get a reward proportional to the specified joint angles 
        Reference motion w.r.t. the joint angle generated from IK from motion
        """
        indices = [
            self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(joint)]
            for joint in self.set_joints
        ]

        #set the knee angle to be negative?? No need since it was ran in myosuite IK.
        error_qpos = self.sim.data.qpos[indices].copy() - self.read_npy()[0][self.steps][indices]
        avg_error_qpos = np.linalg.norm(error_qpos)

        error_qvel = self.sim.data.qvel[indices].copy() - self.read_npy()[1][self.steps][indices]
        avg_error_vel = np.linalg.norm(error_qvel)

        error_root = np.linalg.norm(self.sim.data.qpos[:2] - self.read_npy()[0][self.steps][:2])

        return avg_error_qpos, avg_error_vel, error_root
    
    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if  self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
                qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        
        '''
        for joint in self.ref_data.columns:
            if joint in self.set_joints:  # make sure joint is in model
                idx = self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(joint)]
                qpos[idx] = np.deg2rad(self.ref_data.at[self.steps, joint])
                if "knee_angle" in joint:
                    qpos[idx] = -qpos[idx]
        '''
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]


    def _get_done(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        if self.error_qpos > self.MAX_ERROR:
            return 1
        return 0

    def _get_rot_condition(self):
        """
        MuJoCo specifies the orientation as a quaternion representing the rotation
        from the [1,0,0] vector to the orientation vector. To check if
        a body is facing in the right direction, we can check if the
        quaternion when applied to the vector [1,0,0] as a rotation
        yields a vector with a strong x component.
        """
        # quaternion of root
        quat = self.sim.data.qpos[3:7].copy()
        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0][0]

    
    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.abs(self.target_y_vel - vel[1])) + np.exp(-50*np.abs(self.target_x_vel - vel[0]))

    
    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]


    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com =  self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def read_npy(self, npy_file_path = '/reference_motion/ref_traj_Subj04_walk_18.npy'):
        
        qpos_traj = np.load(os.getcwd() + npy_file_path, allow_pickle=True).item()
        
        return qpos_traj['qpos'], qpos_traj['qvel']
