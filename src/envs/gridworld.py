from src.common.seeding import make_rng
import numpy as np

from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

MOVES = {
    Action.UP:    (-1, 0),
    Action.DOWN:  (1, 0),
    Action.LEFT:  (0, -1),
    Action.RIGHT: (0, 1),
}


class GridWorld:

    def __init__(self, layout, step_penalty = -1.0, goal_reward = 10.0, slip = 0.0, seed = 42):

        # layout: list of strings, 'O' wall, 'G' goal, 'S' start, '.' free

        self.layout = layout
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.slip = slip
        self.n_rows = len(layout)
        self.n_cols = len(layout[0])
        self.n_actions = len(Action)

        self.rng = make_rng(seed)

        self._assign_state_ids()
        self._build_model()
        self.current_state_id = None



    def _assign_state_ids(self):

        self.state_id_of_cell = {} # Dictionary mapping (row, col) -> state id
        self.cell_of_state_id = [] # array of cells(row, col) , position in list = state id

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                char = self.layout[r][c]
                if char == 'O':
                    continue                      # Obstacles are not valid states
                
                state_id = len(self.cell_of_state_id)   # next unused id
                self.state_id_of_cell[(r, c)] = state_id
                self.cell_of_state_id.append((r, c))

                if char == 'S':
                    self.start_state_id = state_id

                if char == 'G':
                    self.goal_state_id = state_id

        self.n_states = len(self.cell_of_state_id)


    def _move(self, current_state_id, action):
        """ Returns next state id after taking action from current state id"""

        row, col = self.cell_of_state_id[current_state_id]
        d_row, d_col = MOVES[Action(action)]
        target_cell = (row + d_row, col + d_col)

        next_state_id = self.state_id_of_cell.get(target_cell, current_state_id)  # fallback to current state if target state does not exist

        return next_state_id

    
    def _build_model(self):
        S, A = self.n_states, self.n_actions
        self.P = np.zeros((S, A, S))
        self.R = np.zeros((S, A, S))
        self.terminal = np.zeros(S, dtype=bool)
        self.terminal[self.goal_state_id] = True

        for state_id in range(S):
            for action in range(A):
                
                if self.terminal[state_id]:
                    self.P[state_id, action, state_id] = 1.0
                    self.R[state_id, action, state_id] = 0.0 # Already reached, no reward
                    continue 

                next_state_id = self._move(state_id, action)
                self.P[state_id, action, next_state_id] = 1.0


                if next_state_id == self.goal_state_id:
                    self.R[state_id, action, next_state_id] = self.step_penalty + self.goal_reward
                else:
                    self.R[state_id, action, next_state_id] = self.step_penalty

    # --- Transitions ----

    def reset(self):
        self.current_state_id = self.start_state_id
        return self.current_state_id

    def step(self, action):
        """Sample one transition. Returns (next_state_id, reward, done)."""
        if self.current_state_id is None:
            raise RuntimeError("Call reset() before step().")

        action = Action(action)   # ValueError if not one of the 4 defined actions

        probs = self.P[self.current_state_id, action]
        next_state_id = int(self.rng.choice(self.n_states, p=probs))
        reward = float(self.R[self.current_state_id, action, next_state_id])
        done = bool(self.terminal[next_state_id])

        self.current_state_id = next_state_id
        return next_state_id, reward, done

    

    

        



if __name__ == '__main__':
    
    LAYOUT = ["S.O", 
              "..G"]

    world = GridWorld(LAYOUT)

    # print(world.state_id_of_cell)
    # print(world.cell_of_state_id)
    # # print(world._move(2, Action.UP))
    # print(world.R[0,Action.RIGHT,1])    
    # #print(world.P)

    world.reset()
    print(world.current_state_id)
    print(world.step(Action.RIGHT))