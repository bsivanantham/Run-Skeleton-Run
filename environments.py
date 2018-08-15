import numpy as np
# from osim.env import RunEnv
from osim.env import ProstheticsEnv
from gym.spaces import Box, MultiBinary


class RunEnv2(ProstheticsEnv):
    def __init__(self, visualize=False, integrator_accuracy=5e-5, model='2D', prosthetic=False, difficulty=0, skip_frame=3, reward_mult=1.):
        super(RunEnv2, self).__init__(visualize, integrator_accuracy)
        self.args = (model, prosthetic, difficulty)
        self.change_model(*self.args)
        # self.state_transform = state_transform
        # self.observation_space = Box(-1000, 1000, [state_size], dtype=np.float32)
        # self.observation_space = Box(-1000, 1000, [state_transform.state_size], dtype=np.float32)
        self.noutput = self.get_action_space_size()
        self.action_space = MultiBinary(self.noutput)
        self.skip_frame = skip_frame
        self.reward_mult = reward_mult

    def reset(self, difficulty=0, seed=None):
        self.change_model(self.args[0], self.args[1], difficulty, seed)
        d = super(RunEnv2, self).reset(False)
        s = self.dict_to_vec(d)
        # self.state_transform.reset()
        # s = self.state_transform.process(s)
        return s

    def is_done(self):  # ndrw
        # state_desc = self.get_state_desc()
        return self.osim_model.state_desc["body_pos"]["pelvis"][1] < 0.3

    def _step(self, action):
        action = np.clip(action, 0, 1)
        info = {'original_reward':0}
        reward = 0.
        for _ in range(self.skip_frame):
            s, r, t, _ = super(RunEnv2, self).step(action, False)
            pelvis = s['body_pos']['pelvis']
            r = self.x_velocity_reward(s)
            s = self.dict_to_vec(s)  # ndrw subtract pelvis_X
            # s = self.state_transform.process(s)
            info['original_reward'] += r
            reward += r
            if t:
                break
        info['pelvis'] = pelvis
        return s, reward*self.reward_mult, t, info

    def x_velocity_reward(self, state):
        penalty = -1.0
        reward = state['misc']['mass_center_vel'][0]  #  X velocity - forward/backward
        if self.model == '2D':
            # if agent is falling, return negative reward
            if state['body_pos']['pelvis'][1] < 0.75:
                return penalty + reward #  -10
            if state['body_pos']['head'][0] - state['body_pos']['pelvis'][0] < -0.35:
                return penalty + reward #  -10
            # x velocity of pelvis
            # return state['body_vel']['pelvis'][0]
            return state['misc']['mass_center_vel'][0]  #  X velocity - forward/backward
        elif self.model == '3D':
            # if agent is falling, return negative reward
            if state['body_pos']['pelvis'][1] < 0.75:  # up/down - absolute values
                return penalty + reward
            if abs(state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]) > 0.4:  # forward/backward
                return penalty + reward
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.35:  # left/right
                return penalty + reward
            # if abs(state['body_pos']['pelvis'][2]) > 0.3:  # left/right - absolute values
            #     return penalty + reward
            # if abs(state['body_pos']['head'][2]) > 0.3:  # left/right - absolute values
            #     return penalty + reward
            if self.args[1]:  # True = prosthetic
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > 0.35:  # left/right - absolute values
                    return penalty + reward
            else:
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > 0.35:  # left/right - absolute values
                    return penalty + reward
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > 0.35:  # left/right - absolute values
                return penalty + reward
            # x velocity of pelvis
            # return state['body_vel']['pelvis'][0]
            return reward  #  X velocity - forward/backward

    # @staticmethod
    def dict_to_vec(self, dict_):
        """Project a dictionary to a vector.
        Filters fiber forces in muscle dictionary as they are already
        contained in the forces dictionary.
        """
        # length without prosthesis: 443 (+ 22 redundant values)
        # length with prosthesis: 390 (+ 19 redundant values)

        # np.array([val_or_key if name != 'muscles'
        #           else list_or_dict[val_or_key]
        #           for name, subdict in dict_.items()
        #           for list_or_dict in subdict.values()
        #           for val_or_key in list_or_dict
        #           if val_or_key != 'fiber_force'])

        #         return np.array([val_or_key if name != 'muscles'
        #                 else list_or_dict[val_or_key]
        #                 for name, subdict in dict_.items()
        #                 for list_or_dict in subdict.values()
        #                 for val_or_key in list_or_dict
        #                 if val_or_key != 'fiber_force'])

        # projection = np.array([])
        # pelvis_X = dict_['body_pos']['pelvis'][0]
        # for dict_name in ['joint_pos', 'joint_vel', 'body_pos', 'body_vel', 'body_pos_rot', 'body_vel_rot', 'misc']:
        #     for dict_name_2 in dict_[dict_name]:
        #         a = dict_[dict_name][dict_name_2]
        #         if len(a) > 0:
        #             if dict_name == 'body_pos':
        #                 a[0] -= pelvis_X
        #             projection = np.concatenate((projection, np.array(a)))
        # assert len(projection) == 196

        # projection = np.array([])
        # pelvis_X = dict_['body_pos']['pelvis'][0]
        # for dict_name in ['joint_pos', 'joint_vel', 'body_pos', 'body_vel', 'body_pos_rot', 'body_vel_rot', 'misc']:
        #     for dict_name_2 in dict_[dict_name]:
        #         a = dict_[dict_name][dict_name_2]
        #         if len(a) > 0:
        #             if dict_name == 'body_pos':
        #                 a[0] -= pelvis_X
        #             projection = np.concatenate((projection, np.array(a)))
        # for dict_name_m in dict_['muscles']:
        #     for dict_name_m2 in ['activation', 'fiber_length', 'fiber_velocity']:
        #         a = [dict_['muscles'][dict_name_m][dict_name_m2]]
        #         projection = np.concatenate((projection, np.array(a)))
        # assert len(projection) == 262
        # return projection

        # projection = np.array([])
        # pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
        # for dict_name in ['body_pos', 'body_vel', 'misc']:
        #     for dict_name_2 in dict_[dict_name]:
        #         a = dict_[dict_name][dict_name_2]
        #         if len(a) > 0:
        #             if dict_name == 'body_pos':
        #                 a[0] -= pelvis_X
        #             projection = np.concatenate((projection, np.array(a)))
        # for dict_name_m in dict_['muscles']:
        #     for dict_name_m2 in ['activation']:  # , 'fiber_length', 'fiber_velocity']:
        #         a = [dict_['muscles'][dict_name_m][dict_name_m2]]
        #         projection = np.concatenate((projection, np.array(a)))
        # assert len(projection) == 106
        # return projection
        if self.model == '2D':
            pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
            projection = [dict_['body_pos']['pelvis'][1]]  # pelvis up
            for dict_name in ['body_pos', 'body_vel']:
                for dict_name_2 in ['head', 'pelvis', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'toes_r', 'toes_l']:  # dict_[dict_name]
                    if dict_name_2 == 'pelvis' and dict_name == 'body_pos':
                        continue
                    lll = dict_[dict_name][dict_name_2]
                    if len(lll) > 0:
                        for i in [0, 1]:
                            l = lll[i]
                            if dict_name == 'body_pos' and i == 0:
                                projection += [l-pelvis_X]
                            else:
                                projection += [l]
            projection += [dict_['misc']['mass_center_pos'][0] - pelvis_X]
            projection += [dict_['misc']['mass_center_pos'][1]]
            projection += dict_['misc']['mass_center_vel']

            assert len(projection) == 35
            projection = np.array(projection)
            return projection

        # elif self.model == '3D':
        #     projection_dict = {}
        #     pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
        #     pelvis_Z = dict_['body_pos']['pelvis'][2]  # X - forward, Y - up, Z - left/right
        #     for dict_name in ['body_pos', 'body_vel']:
        #         for dict_name_2 in ['head', 'pelvis', 'pros_tibia_r', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'toes_r', 'toes_l', 'pros_foot_r']:  # dict_[dict_name]
        #             if dict_name_2 in dict_[dict_name]:  # e.g. prosthetic exceptions: ['pros_tibia_r', 'pros_foot_r']
        #                 lll = dict_[dict_name][dict_name_2]
        #                 if len(lll) > 0:
        #                     for i in [0, 1, 2]:
        #                         l = lll[i]
        #                         if i == 0:  # forward/backward
        #                             if dict_name == 'body_pos' and dict_name_2 != 'pelvis':
        #                                 projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i) + '_relX'] = l-pelvis_X
        #                             elif dict_name == 'body_vel':
        #                                 projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
        #                         elif i == 1:  # up/down
        #                                 projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
        #                         elif i == 2:  # left/right
        #                             # if dict_name_2 in ['head', 'pelvis', 'talus_r', 'talus_l', 'pros_tibia_r', 'pros_foot_r']:
        #                             if dict_name == 'body_pos' and dict_name_2 != 'pelvis':
        #                                 projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i) + '_relZ'] = l - pelvis_Z
        #                             elif dict_name == 'body_vel':
        #                                 projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
        #     projection_dict['misc' + '-' + 'mass_center_pos' + '-' + str(0) + '_relX'] = dict_['misc']['mass_center_pos'][0] - pelvis_X
        #     projection_dict['misc' + '-' + 'mass_center_pos' + '-' + str(1)] = dict_['misc']['mass_center_pos'][1]
        #     projection_dict['misc' + '-' + 'mass_center_vel' + '-' + str(0)] = dict_['misc']['mass_center_vel'][0]
        #     projection_dict['misc' + '-' + 'mass_center_vel' + '-' + str(1)] = dict_['misc']['mass_center_vel'][1]
        #     if self.args[1]:
        #         assert len(projection_dict) == 44  # prosthetic
        #     else:
        #         assert len(projection_dict) == 50  # normal
        #     projection = np.array(list(projection_dict.values()))
        #     return projection

        elif self.model == '3D':
            projection_dict = {}
            pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
            pelvis_Y = dict_['body_pos']['pelvis'][1]  # X - forward, Y - up, Z - left/right
            pelvis_Z = dict_['body_pos']['pelvis'][2]  # X - forward, Y - up, Z - left/right
            for dict_name in ['body_pos', 'body_vel']:
                for dict_name_2 in ['head', 'pelvis', 'pros_tibia_r', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'pros_foot_r', 'toes_r', 'toes_l']:  # dict_[dict_name]
                    if dict_name_2 in dict_[dict_name]:  # e.g. prosthetic exceptions: ['pros_tibia_r', 'pros_foot_r']
                        lll = dict_[dict_name][dict_name_2]
                        if len(lll) > 0:
                            for i in [0, 1, 2]:
                                l = lll[i]
                                if i == 0:  # forward/backward
                                    if dict_name == 'body_pos' and dict_name_2 != 'pelvis':
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i) + '_relX'] = l-pelvis_X
                                    elif dict_name == 'body_vel' and (dict_name_2 not in ['pros_tibia_r', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']):
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
                                elif i == 1:  # up/down
                                    if dict_name == 'body_pos':
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
                                    elif dict_name == 'body_vel' and (dict_name_2 not in ['pros_tibia_r', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']):
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
                                elif i == 2:  # left/right
                                    # if dict_name_2 in ['head', 'pelvis', 'talus_r', 'talus_l', 'pros_tibia_r', 'pros_foot_r']:
                                    if dict_name == 'body_pos' and dict_name_2 != 'pelvis':
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i) + '_relZ'] = l - pelvis_Z
                                    elif dict_name == 'body_vel' and (dict_name_2 not in ['pros_tibia_r', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']):
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
            projection_dict['misc' + '-' + 'mass_center_pos' + '-' + str(0) + '_relX'] = dict_['misc']['mass_center_pos'][0] - pelvis_X
            projection_dict['misc' + '-' + 'mass_center_pos' + '-' + str(1) + '_relY'] = dict_['misc']['mass_center_pos'][1] - pelvis_Y
            projection_dict['misc' + '-' + 'mass_center_vel' + '-' + str(0)] = dict_['misc']['mass_center_vel'][0]
            projection_dict['misc' + '-' + 'mass_center_vel' + '-' + str(1)] = dict_['misc']['mass_center_vel'][1]
            if self.args[1]:
                assert len(projection_dict) == 35  # prosthetic
            else:
                assert len(projection_dict) == 38  # normal
            projection = np.array(list(projection_dict.values()))
            return projection
