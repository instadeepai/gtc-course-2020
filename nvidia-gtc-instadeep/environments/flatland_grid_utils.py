import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point, get_new_position
from flatland.core.grid.grid_utils import IntVector2D, IntVector2DDistance, IntVector2DArray
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.transition_map import GridTransitionMap, RailEnvTransitions
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML


def connect_straight_line_in_grid_map_(grid_map: GridTransitionMap, path: IntVector2DArray, rail_trans: RailEnvTransitions) -> IntVector2DArray:
    """
    Generates a straight rail line from start cell to end cell.
    Diagonal lines are not allowed
    :param rail_trans:
    :param grid_map:
    :param start: Cell coordinates for start of line
    :param end: Cell coordinates for end of line
    :return: A list of all cells in the path
    """
    plen=len(path)
    to_return = path.copy()
    if plen<2:
        return []

    if path[0]==path[1]:
        path = path[1:]
        to_return.remove(path[0])
    
    current_dir = get_direction(path[0], path[1])
    end_pos = path[-1]

    for index in range(len(path)-1):
        current_pos = path[index]
        new_pos = path[index+1]
        if current_pos == new_pos:
            to_return.remove(current_pos)
            continue
        new_dir = get_direction(current_pos, new_pos)
        try:
            new_trans = grid_map.grid[current_pos]
        except:
            import pdb; pdb.set_trace()
        new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
        new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
        grid_map.grid[current_pos] = new_trans

        if new_pos == end_pos:
            new_trans_e = grid_map.grid[end_pos]
            new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)
            grid_map.grid[end_pos] = new_trans_e
        current_dir = new_dir

    return to_return

def get_cells_in_path(start: IntVector2D,
                        end: IntVector2D,) -> IntVector2DArray:
    """
    Generates a straight rail line from start cell to end cell.
    Diagonal lines are not allowed
    :param rail_trans:
    :param grid_map:
    :param start: Cell coordinates for start of line
    :param end: Cell coordinates for end of line
    :return: A list of all cells in the path
    """

    if not (start[0] == end[0] or start[1] == end[1]):
        print("No straight line possible!")
        return []

    direction = direction_to_point(start, end)

    if direction is Grid4TransitionsEnum.NORTH or direction is Grid4TransitionsEnum.SOUTH:
        start_row = min(start[0], end[0])
        end_row = max(start[0], end[0]) + 1
        rows = np.arange(start_row, end_row)
        length = np.abs(end[0] - start[0]) + 1
        cols = np.repeat(start[1], length)

    else:  # Grid4TransitionsEnum.EAST or Grid4TransitionsEnum.WEST
        start_col = min(start[1], end[1])
        end_col = max(start[1], end[1]) + 1
        cols = np.arange(start_col, end_col)
        length = np.abs(end[1] - start[1]) + 1
        rows = np.repeat(start[0], length)
    
    path = list(zip(rows, cols))
    if direction in [0, 3]:
        # import pdb; pdb.set_trace()
        path = path[::-1]

    return path

def obtain_cells(path):
    """
    """
    if len(path)<2:
        return []
    all_cells = []
    for index in range(len(path)-1):
        start = path[index]
        end = path[index+1]
        cells = get_cells_in_path(start,end)
        if index == 0: 
            all_cells.extend(cells)
        else:
            all_cells.extend(cells[1:])

    return all_cells
