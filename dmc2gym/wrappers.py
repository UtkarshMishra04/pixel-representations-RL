from gym import core, spaces
import glob
import os
import local_dm_control_suite as suite
import local_dm_control_suite_off_center as suite_off_center
from dm_env import specs
import numpy as np
import skimage.io
import cv2
from dmc2gym import natural_imgsource


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        resource_files,
        img_source,
        total_frames,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        off_center=False,
        environment_kwargs=None
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._img_source = img_source


        # create task

        if off_center:
            self._env = suite_off_center.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs
            )
        else:
            self._env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs
            )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255, shape=[3, height, width], dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._internal_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._env.physics.get_state().shape,
            dtype=np.float32
        )

        # background
        if img_source is not None:
            shape2d = (height, width)
            if img_source == "color":
                self._bg_source = natural_imgsource.RandomColorSource(shape2d)
            elif img_source == "noise":
                self._bg_source = natural_imgsource.NoiseSource(shape2d)
            elif img_source == "black":
                self._bg_source = natural_imgsource.BlackColorSource(shape2d)
            else:
                files = glob.glob(os.path.expanduser(resource_files))
                assert len(files), "Pattern {} does not match any files".format(
                    resource_files
                )
                if img_source == "images":
                    self._bg_source = natural_imgsource.RandomImageSource(shape2d, files, grayscale=True, total_frames=total_frames)
                elif img_source == "video":
                    self._bg_source = natural_imgsource.RandomVideoSource(shape2d, files, grayscale=True, total_frames=total_frames)
                else:
                    raise Exception("img_source %s not defined." % img_source)

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._img_source is not None:
                mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))  # hardcoded for dmc
                bg = self._bg_source.get_image()
                obs[mask] = bg[mask]

            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def internal_state_space(self):
        return self._internal_state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
