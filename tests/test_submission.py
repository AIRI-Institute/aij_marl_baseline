import os
import sys
import time
import warnings
from random import shuffle
from typing import Any, Dict, List, Tuple

import pytest
from aij_multiagent_rl.agents import BaseAgent, RandomAgent
from aij_multiagent_rl.env import AijMultiagentEnv
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = 'tests/test_config.yaml'


def sample_rollouts(
    n_rollouts: int,
    env: AijMultiagentEnv,
    agents: Dict[str, BaseAgent]
) -> Tuple[List[List[Dict[str, Any]]], float]:
    rollouts = []
    action_times = 0
    for _ in range(n_rollouts):
        rollout = []
        for agent in agents.values():
            agent.reset_state()
        observations, infos = env.reset()
        done = False
        while not done:
            start = time.perf_counter()
            actions = {name: agent.get_action(observation=observations[name])
                       for name, agent in agents.items() if name in env.agents}
            end = time.perf_counter()
            action_times += (end-start)
            next_observations, rewards, terminations, truncations, next_infos = env.step(actions)
            transition = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'rewards': rewards,
                'terminations': terminations,
                'truncations': truncations
            }
            observations = next_observations
            done = all(truncations.values()) or all(terminations.values())
            rollout.append(transition)
        rollouts.append(rollout)
    action_time = action_times / (sum([len(e) for e in rollouts]) * 8)
    return rollouts, action_time


class RandomLoopedPop:

    def __init__(self, options):
        self.options = options
        self._new_shuffled_options()

    def _new_shuffled_options(self):
        self.shuffled_options = self.options.copy()
        shuffle(self.shuffled_options)

    def pop(self):
        try:
            out = self.shuffled_options.pop()
        except IndexError:
            self._new_shuffled_options()
            out = self.shuffled_options.pop()
        return out


@pytest.fixture
def config() -> DictConfig:
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture
def submission_agents(config: DictConfig) -> Dict[str, BaseAgent]:
    sys.path.insert(1, config.submission_dir)
    from model import get_agent
    agents_dir = os.path.join(config.submission_dir, 'agents')
    loaded_agents = {}
    for artifact in os.listdir(agents_dir):
        if not artifact.startswith('.'):
            artifact_dir = os.path.join(agents_dir, artifact)
            agent_config = OmegaConf.load(
                os.path.join(artifact_dir, 'agent_config.yaml'))
            agent = get_agent(agent_config)
            agent.load(artifact_dir)
            loaded_agents[artifact] = agent
    return loaded_agents


@pytest.fixture
def env():
    return AijMultiagentEnv()


def test_submission_size(config: DictConfig):
    """Test submission size

    Maximum submission size should not exceed 5gb.

    Args:
        config: tests config

    Raises:
        AssertionError: If submission size exceeds 5 gb.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(config.submission_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    total_size_gb = total_size / 10**9
    if total_size_gb > 5:
        raise AssertionError(
            f'Submission directory {config.submission_dir} is larger than 5 gb. ({total_size_gb})'
        )


def test_model_file(config: DictConfig):
    """Test if `model.py` file exists

    Test if `model.py` file exists

    Args:
        config: tests config

    Raises:
        AssertionError: If `model.py` file does not exist
    """
    if 'model.py' not in os.listdir(config.submission_dir):
        raise AssertionError(
            f'File `model.py` not found in {config.submission_dir}'
        )


def test_agents_dir_exists(config: DictConfig):
    """Test if `agents` directory exists

    Test if `agents` directory exists

    Args:
        config: tests config

    Raises:
        AssertionError: If `agents` directory does not exist
    """
    agents_dir = os.path.join(config.submission_dir, 'agents')
    if not os.path.isdir(agents_dir):
        raise AssertionError(
            f'Submission {config.submission_dir} should contain `agents` directory with agents artifacts'
        )


def test_agents_subdirectories_exist(config: DictConfig):
    """Test `agents` directory

    `agents` directory should contain only subdirectories
    with agents' artifacts

    Args:
        config: tests config

    Raises:
        AssertionError: If there is something apart from
            subdirectories in `agents` directory
    """
    is_dir = []
    agents_dir = os.path.join(config.submission_dir, 'agents')
    for artifact in os.listdir(agents_dir):
        if not artifact.startswith('.'):
            is_dir.append(os.path.isdir(os.path.join(agents_dir, artifact)))
    if not all(is_dir):
        raise AssertionError(
            '`agents` directory should contain only sub-directories'
        )


def test_agent_config_yaml(config: DictConfig):
    """Test `agent_config.yaml` file

    Each subdirectory in `agents` directory should contain
    `agent_config.yaml` file for factory method

    Args:
        config: tests config

    Raises:
        AssertionError: If there is no `agent_config.yaml` file
            in any subdirectory
    """
    agents_dir = os.path.join(config.submission_dir, 'agents')
    for artifact in os.listdir(agents_dir):
        if not artifact.startswith('.'):
            artifact_dir = os.path.join(agents_dir, artifact)
            if 'agent_config.yaml' not in os.listdir(artifact_dir):
                raise AssertionError(
                    f'`agent_config.yaml` file not found in {artifact_dir}'
                )


def test_base_agent_inheritance(submission_agents: Dict[str, BaseAgent]):
    """Test BaseAgent inheritance

    Test that all user agents are inherited from
    `aij_multiagent_rl.agents.BaseAgent` class

    Args:
        submission_agents: dictionary with initialized user
            agents and their names (names are taken from
            subdirectory name)

    Raises:
        AssertionError: If there is at least one user agent not
            inherited from `aij_multiagent_rl.agents.BaseAgent`
    """
    for name, agent in submission_agents.items():
        if not issubclass(type(agent), BaseAgent):
            raise AssertionError(
                f'Agent {name} is not inherited from `aij_multiagent_rl.agents.BaseAgent`'
            )


def test_agents_selfplay(
    config: DictConfig,
    submission_agents: Dict[str, BaseAgent],
    env: AijMultiagentEnv
):
    """Test agents self-play

    Test that agents may be used for sampling
    actions from `AijMultiagentEnv` simulator and
    therefore follow required API.

    We also test here for agents sampling performance
    in order to match 100min. time constraint in testing system.
    Obviously, testing machine will have different setup from
    the one these tests will be run on (see main contest info at:
    https://dsworks.ru/champ/multiagent-ai). However, if you have
    a GPU on your local machine, you possibly should treat
    performance warning more seriously.

    Args:
        config: tests config
        submission_agents: dictionary with initialized user
            agents and their names (names are taken from
            subdirectory name)
        env: AijMultiagentEnv simulator

    Raises:
        UserWarning: If average get action time by agents exceeds
            `config.wall_time_threshold` milliseconds
    """
    # Assign agents with valid keys from environment
    rlp = RandomLoopedPop(options=list(submission_agents.keys()))
    agents = {}
    for name in env.possible_agents:
        agent_key = rlp.pop()
        agents[name] = submission_agents[agent_key]
    # Run simulation
    _, acs_time = sample_rollouts(
        n_rollouts=config.test_episodes_num,
        env=env,
        agents=agents
    )
    tt = config.wall_time_threshold
    if acs_time > tt:
        warnings.warn(
            f"""Mean `get_action()` wall time is greater than {tt} sec ({acs_time} sec), which may be too slow"""
        )


def test_agents_vs_random_play(
    config: DictConfig,
    submission_agents: Dict[str, BaseAgent],
    env: AijMultiagentEnv
):
    """Test agents vs random agents

    Test that agents may be used for sampling
    actions from `AijMultiagentEnv` simulator by playing
    together with random agents

    Args:
        config: tests config
        submission_agents: dictionary with initialized user
            agents and their names (names are taken from
            subdirectory name)
        env: AijMultiagentEnv simulator
    """
    # Assign agents with valid keys from environment
    rlp = RandomLoopedPop(options=list(submission_agents.keys()))
    agents = {}
    for i, name in enumerate(env.possible_agents):
        if i % 2 == 0:
            agent_key = rlp.pop()
            agents[name] = submission_agents[agent_key]
        else:
            agents[name] = RandomAgent(
                action_dim=env.action_space(name).n,
                seed=i
            )
    # Run simulation
    sample_rollouts(
        n_rollouts=config.test_episodes_num,
        env=env,
        agents=agents
    )
