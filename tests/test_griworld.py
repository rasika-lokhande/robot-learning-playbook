import numpy as np
import pytest
from src.envs.gridworld import GridWorld, Action

LAYOUT = ["S.O",
          "..G"]


@pytest.fixture
def env():
    return GridWorld(LAYOUT)

def test_probs_sum_to_one(env):
    assert np.allclose(env.P.sum(axis=2), 1.0)

def test_goal_is_absorbing(env):
    g = env.goal_state_id
    for a in range(env.n_actions):
        assert env.P[g, a, g] == 1.0
        assert env.R[g, a, g] == 0.0


@pytest.mark.parametrize("bad_action", [-1, 4])
def test_step_rejects_invalid_actions(env, bad_action):
    env.reset()
    with pytest.raises(ValueError):
        env.step(bad_action)