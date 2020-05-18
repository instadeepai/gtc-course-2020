import sys
import warnings
from typing import Callable, Tuple, Optional, Dict, List

import msgpack
import numpy as np
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.grid_utils import distance_on_rail, IntVector2DArray, IntVector2D, \
    Vec2dOperations
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes, align_cell_to_city
from environments.flatland_grid_utils import connect_straight_line_in_grid_map_, get_cells_in_path, obtain_cells


RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]


def simple_rail_generator(n_trains: int = 2, grid_mode: bool = True, max_rails_between_cities: int = 4,
                          max_rails_in_city: int = 4, seed=0) -> RailGenerator:
    """
    Generates railway networks between cities

    Parameters
    ----------
    n_trains: int
        number of trains in map. number of cities should be the same as number of
        trains
    grid_mode: bool
        distribute the cities in a grid with rows and columns
    
    Returns
    -------
    rail generator object
    """
    num_cities = n_trains
    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
        np_random: RandomState = np.random) -> RailGenerator:
        """

        Parameters
        ----------
        width: int
            Width of the environment
        height: int
            Height of the environment
        num_agents:
            Number of agents to be placed within the environment
        num_resets: int
            Count for how often the environment has been reset

        Returns
        -------
        grid_map: 

        """
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, \
            transitions=rail_trans)
        vector_field = np.zeros(shape=(height,width)) - 1.
        city_padding = 2
        city_radius = city_padding
        rails_between_cities = 1
        rails_in_city = 2
        np_random.seed(seed)

        # Calculate the max number of cities allowed
        # and reduce the number of cities to build to avoid problems
        max_feasible_cities = min(num_cities,
                                  ((height - 2) // (2 * (city_radius + 1))) * \
                                      ((width - 2) // (2 * (city_radius + 1))))

        if max_feasible_cities < num_cities:
            sys.exit(f"[ABORT] Cannot fit more than {max_feasible_cities} city in this map, no feasible environment possible! Aborting.")

        # obtain city positions
        city_positions = _generate_evenly_distr_city_positions(max_feasible_cities, \
            city_radius, width, height)

        

        # Set up connection points for all cities
        inner_connection_points, outer_connection_points, city_orientations, city_cells = \
            _generate_city_connection_points(
                city_positions, city_radius, vector_field, rails_between_cities,
                rails_in_city, np_random=np_random)
        # import pdb; pdb.set_trace()
        # connect the cities through the connection points
        inter_city_lines = _connect_cities(city_positions, outer_connection_points, city_cells,
                                           rail_trans, grid_map)

        # Build inner cities
        free_rails = _build_inner_cities(city_positions, inner_connection_points,
                                         outer_connection_points,
                                         rail_trans,
                                         grid_map)

        # Populate cities
        train_stations = _set_trainstation_positions(city_positions, city_radius, free_rails)

        # Fix all transition elements
        _fix_transitions(city_cells, inter_city_lines, grid_map, vector_field)
        
        return grid_map, {'agents_hints': {
            'num_agents': num_agents,
            'city_positions': city_positions,
            'train_stations': train_stations,
            'city_orientations': city_orientations
        }}
    

    def _generate_evenly_distr_city_positions(num_cities: int, \
        city_radius: int, width: int, height: int
        ) -> (IntVector2DArray, IntVector2DArray):
        """
        Distribute the cities in an evenly spaced grid

        Parameters
        ----------
        num_cities: int
            Max number of cities that should be placed
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        width: int
            Width of the environment
        height: int
            Height of the environment

        Returns
        -------
        Returns a list of all city positions as coordinates (x,y)

        """
        aspect_ratio = height / width

        # Compute max numbe of possible cities per row and col.
        # Respect padding at edges of environment
        # Respect padding between cities
        padding = 2
        city_size = 2 * (city_radius + 1)
        max_cities_per_row = int((height - padding) // city_size)
        max_cities_per_col = int((width - padding) // city_size)

        # Choose number of cities per row.
        # Limit if it is more then max number of possible cities

        cities_per_row = min(int(np.ceil(np.sqrt(num_cities * aspect_ratio))), max_cities_per_row)
        cities_per_col = min(int(np.ceil(num_cities / cities_per_row)), max_cities_per_col)
        num_build_cities = min(num_cities, cities_per_col * cities_per_row)
        row_positions = np.linspace(city_radius + 2, height - (city_radius + 2), cities_per_row, dtype=int)
        col_positions = np.linspace(city_radius + 2, width - (city_radius + 2), cities_per_col, dtype=int)
        city_positions = []

        for city_idx in range(num_build_cities):
            row = row_positions[city_idx % cities_per_row]
            col = col_positions[city_idx // cities_per_row]
            city_positions.append((row, col))
        return city_positions


    def _generate_city_connection_points(city_positions: IntVector2DArray, city_radius: int,
                                         vector_field: IntVector2DArray, rails_between_cities: int,
                                         rails_in_city: int = 2, np_random: RandomState = None) -> (
        List[List[List[IntVector2D]]],
        List[List[List[IntVector2D]]],
        List[np.ndarray],
        List[Grid4TransitionsEnum]):
        """
        Generate the city connection points. Internal connection points are used to generate the parallel paths
        within the city.
        External connection points are used to connect different cities together

        Parameters
        ----------
        city_positions: IntVector2DArray
            Vector that contains all the positions of the cities
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        vector_field: IntVector2DArray
            Vectorfield of the size of the environment. It is used to generate preferred orienations for each cell.
            Each cell contains the prefered orientation of cells. If no prefered orientation is present it is set to -1
        rails_between_cities: int
            Number of rails that connect out from the city
        rails_in_city: int
            Number of rails within the city

        Returns
        -------
        inner_connection_points: List of List of length number of cities
            Contains all the inner connection points for each boarder of each city.
            [North_Points, East_Poinst, South_Points, West_Points]
        outer_connection_points: List of List of length number of cities
            Contains all the outer connection points for each boarder of the city.
            [North_Points, East_Poinst, South_Points, West_Points]
        city_orientations: List of length number of cities
            Contains all the orientations of cities. This is then used to orient agents according to the rails
        city_cells: List
            List containing the coordinates of all the cells that belong to a city. This is used by other algorithms
            to avoid drawing inter-city-rails through cities.
        """
        inner_connection_points: List[List[List[IntVector2D]]] = []
        outer_connection_points: List[List[List[IntVector2D]]] = []
        city_orientations: List[Grid4TransitionsEnum] = []
        city_cells: IntVector2DArray = []

        for city_position in city_positions:

            # Chose the directions where close cities are situated
            neighb_dist = []
            for neighbour_city in city_positions:
                neighb_dist.append(Vec2dOperations.get_manhattan_distance(city_position, neighbour_city))
            closest_neighb_idx = argsort(neighb_dist)

            # Store the directions to these neighbours and orient city to face closest neighbour
            connection_sides_idx = []
            idx = 1
            if grid_mode:
                current_closest_direction = np_random.randint(4)
            else:
                current_closest_direction = direction_to_point(city_position, city_positions[closest_neighb_idx[idx]])
            connection_sides_idx.append(current_closest_direction)
            connection_sides_idx.append((current_closest_direction + 2) % 4)
            city_orientations.append(current_closest_direction)
            city_cells.extend(_get_cells_in_city(city_position, city_radius, city_orientations[-1], vector_field))
            # set the number of tracks within a city, at least 2 tracks per city
            connections_per_direction = np.zeros(4, dtype=int)
            nr_of_connection_points = np_random.randint(2, rails_in_city + 1)
            for idx in connection_sides_idx:
                connections_per_direction[idx] = nr_of_connection_points
            connection_points_coordinates_inner: List[List[IntVector2D]] = [[] for i in range(4)]
            connection_points_coordinates_outer: List[List[IntVector2D]] = [[] for i in range(4)]
            number_of_out_rails = np_random.randint(1, min(rails_between_cities, nr_of_connection_points) + 1)
            start_idx = int((nr_of_connection_points - number_of_out_rails) / 2)
            for direction in range(4):
                connection_slots = np.arange(nr_of_connection_points) - start_idx
                # Offset the rails away from the center of the city
                offset_distances = np.arange(nr_of_connection_points) - int(nr_of_connection_points / 2)
                # The clipping helps ofsetting one side more than the other to avoid switches at same locations
                # The magic number plus one is added such that all points have at least one offset
                inner_point_offset = np.abs(offset_distances) + np.clip(offset_distances, 0, 1) + 1
                for connection_idx in range(connections_per_direction[direction]):
                    if direction == 0:
                        tmp_coordinates = (
                            city_position[0] - city_radius + inner_point_offset[connection_idx],
                            city_position[1] + connection_slots[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] - city_radius, city_position[1] + connection_slots[connection_idx])
                    if direction == 1:
                        tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx],
                            city_position[1] + city_radius - inner_point_offset[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx], city_position[1] + city_radius)
                    if direction == 2:
                        tmp_coordinates = (
                            city_position[0] + city_radius - inner_point_offset[connection_idx],
                            city_position[1] + connection_slots[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] + city_radius, city_position[1] + connection_slots[connection_idx])
                    if direction == 3:
                        tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx],
                            city_position[1] - city_radius + inner_point_offset[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx], city_position[1] - city_radius)
                    connection_points_coordinates_inner[direction].append(tmp_coordinates)
                    if connection_idx in range(start_idx, start_idx + number_of_out_rails):
                        connection_points_coordinates_outer[direction].append(out_tmp_coordinates)

            inner_connection_points.append(connection_points_coordinates_inner)
            outer_connection_points.append(connection_points_coordinates_outer)
        return inner_connection_points, outer_connection_points, city_orientations, city_cells


    def _connect_cities(city_positions: IntVector2DArray, connection_points: List[List[List[IntVector2D]]],
                        city_cells: IntVector2DArray,
                        rail_trans: RailEnvTransitions, grid_map: RailEnvTransitions) -> List[IntVector2DArray]:
        """
        Connects cities together through rails. Each city connects from its outgoing connection points to the closest
        cities. This guarantees that all connection points are used.

        Parameters
        ----------
        city_positions: IntVector2DArray
            All coordinates of the cities
        connection_points: List[List[List[IntVector2D]]]
            List of coordinates of all outer connection points
        city_cells: IntVector2DArray
            Coordinates of all the cells contained in any city. This is used to avoid drawing rails through existing
            cities.
        rail_trans: RailEnvTransitions
            Railway transition objects
        grid_map: RailEnvTransitions
            The grid map containing the rails. Used to draw new rails

        Returns
        -------
        Returns a list of all the cells (Coordinates) that belong to a rail path. This can be used to access railway
        cells later.
        """
        all_paths: List[IntVector2DArray] = []
        connect_cities = []
        connected_points = []

        grid4_directions = [Grid4TransitionsEnum.NORTH, Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.SOUTH,
                            Grid4TransitionsEnum.WEST]

        for current_city_idx in np.arange(len(city_positions)):
            closest_neighbours = _closest_neighbour_in_grid4_directions(current_city_idx, city_positions)
            for out_direction in grid4_directions:

                neighbour_idx = get_closest_neighbour_for_direction(closest_neighbours, out_direction)
                if set((current_city_idx, neighbour_idx)) in connect_cities:
                    continue
                for city_out_connection_point in connection_points[current_city_idx][out_direction]:
                    city_out_connection_dir = out_direction
                    min_connection_dist = np.inf
                    for direction in grid4_directions:
                        current_points = connection_points[neighbour_idx][direction]
                        for tmp_in_connection_point in current_points:
                            tmp_dist = Vec2dOperations.get_manhattan_distance(city_out_connection_point,
                                                                              tmp_in_connection_point)
                            if tmp_dist < min_connection_dist:
                                min_connection_dist = tmp_dist
                                neighbour_connection_point = tmp_in_connection_point
                                neighbour_connection_dir = direction
                    if set((*city_out_connection_point, *neighbour_connection_point)) in connected_points:
                        continue             
                    lines = _align_start_end(city_out_connection_point, neighbour_connection_point,\
                         city_out_connection_dir, neighbour_connection_dir, grid_map, rail_trans, city_cells)
                    if len(city_positions) == 2:
                        new_line = connect_points(lines, grid_map, rail_trans)
                    else:
                        new_line = connect_rail_in_grid_map(grid_map, city_out_connection_point, neighbour_connection_point,
                                                            rail_trans, flip_start_node_trans=False,
                                                            flip_end_node_trans=False, respect_transition_validity=False,
                                                            avoid_rail=True,
                                                            forbidden_cells=city_cells)
                    all_paths.extend(new_line)
                    connect_cities.append(set((current_city_idx, neighbour_idx)))
                    connected_points.append(set((*city_out_connection_point, *neighbour_connection_point)))

        return all_paths
        

    #=================================================================================
    def connect_points(lines, grid_map, rail_trans):
        """
        """
        lines1 = []
        for m in lines:
            path = obtain_cells(m)
            line = connect_straight_line_in_grid_map_(grid_map, path, rail_trans)
            lines1.extend(line)
        
        return lines1
        
    def _align_start_end(point1, point2, dir1, dir2, grid_map, rail_trans, city_cells):
        """
        """
        h, w = grid_map.grid.shape
        # obtain x and y distances
        dist_y = abs(point1[0]-point2[0])
        dist_x = abs(point1[1]-point2[1])

        branch_length = 3

        # which is larger
        if dist_y < dist_x:
            # adjust points to be horizontal
            # direction should be horizontal
            # point1
            y1=point1[0]
            y2=point2[0]
            # y1 should equal y2
            if y1 != y2:
                # add to either
                new_y1 = max(min(y1 + (y2-y1), h-1),1)
                if (new_y1, point1[1]) in city_cells and dir1 in [0,2]:
                    new_y1 = max(min(y1 - (y2-y1), h-1),1)
                new_y2 = new_y1
            else:
                new_y1 = y1
                new_y2 = y1

            # centralize points
            x1 = point1[1]
            x2 = point2[1]
            if x1 < x2:
                next_stop = x1 + (dist_x//2 - branch_length+1)
                mid = next_stop + branch_length//2
                next_stop2 = next_stop + branch_length

                # movements
                m1 = [(y1,x1), (new_y1, x1), (new_y1, x2),(y2, x2)] 
                if new_y1+1 < h:
                    m2 = [(new_y1, x1), (new_y1, next_stop), (new_y1+1, next_stop), (new_y1+1, next_stop2), (new_y1, next_stop2), (new_y1, x2)]
                else:
                    m2 = [(new_y1, x1), (new_y1, next_stop), (new_y1-1, next_stop), (new_y1-1, next_stop2), (new_y1, next_stop2), (new_y1, x2)]
            else:
                next_stop = x2 + (dist_x//2 - branch_length+1)
                mid = next_stop + branch_length//2
                next_stop2 = next_stop + branch_length

                # movements
                m1 = [(y2,x2), (new_y2, x2), (new_y2, next_stop),(new_y2, next_stop2),(new_y2, x1),(y1, x1)]
                if new_y2+1 < h:
                    m2 = [(y2,x2), (new_y2, x2), (new_y2, next_stop),(new_y2+1, next_stop), (new_y2+1, next_stop2), (new_y2, next_stop2),(new_y2, x1),(y1, x1)]
                else:
                    m2 = [(y2,x2), (new_y2, x2), (new_y2, next_stop),(new_y2+1, next_stop), (new_y2+1, next_stop2), (new_y2, next_stop2),(new_y2, x1),(y1, x1)]

        else:
            # adjust points to be vertical
            # direction should be vertical
            # point1
            x1=point1[1]
            x2=point2[1]
            # x1 should equal x2
            if x1 != x2:
                # add to either
                new_x1 = max(min(x1+(x2-x1), w-1),1)
                if (point1[0], new_x1) in city_cells and dir1 in [1,3]:
                    new_x1 = max(min(x1-(x2-x1), w-1),1)
                new_x2 = new_x1
            else:
                new_x1 = x1
                new_x2 = x1

            # centralize points
            y1 = point1[0]
            y2 = point2[0]
            if y1 < y2:
                next_stop = y1 + (dist_y//2 - branch_length+1)
                mid = next_stop + branch_length//2
                next_stop2 = next_stop + branch_length

                # movements
                m1 = [(y1,x1), (y1, new_x1), (next_stop, new_x1),(next_stop2, new_x1), (y2,new_x1),(y2, x2) ]
                if new_x1+1 < w:
                    m2 = [(y1,x1), (y1, new_x1), (next_stop, new_x1), (next_stop, new_x1+1), (next_stop2, new_x1+1),(next_stop2, new_x1), (y2,new_x1),(y2, x2) ]
                else:
                    m2 = [(y1,x1), (y1, new_x1), (next_stop, new_x1), (next_stop, new_x1-1), (next_stop2, new_x1-1),(next_stop2, new_x1), (y2,new_x1),(y2, x2) ]
                
            else:
                next_stop = y2 + (dist_x//2 - branch_length+1)
                mid = next_stop + branch_length//2
                next_stop2 = next_stop + branch_length

                # movements
                m1 = [(y2,x2), (y2, new_x2), (next_stop, new_x2), (next_stop2, new_x2), (y1,new_x2), (y1, x1)]
                if new_x1+1 < w:
                    m2 = [(y2,x2), (y2, new_x2), (next_stop, new_x2), (next_stop, new_x2+1), (next_stop2, new_x2+1) (next_stop2, new_x2), (y1,new_x2), (y1, x1)]
                else:
                    m2 = [(y2,x2), (y2, new_x2), (next_stop, new_x2), (next_stop, new_x2-1), (next_stop2, new_x2-1) (next_stop2, new_x2), (y1,new_x2), (y1, x1)]
                
        return m1,m2
            
    #==================================================================================
        

    def get_closest_neighbour_for_direction(closest_neighbours, out_direction):
        """
        Given a list of clostest neighbours in each direction this returns the city index of the neighbor in a given
        direction. Direction is a 90 degree cone facing the desired directiont.
        Exampe:
            North: The closes neighbour in the North direction is within the cone spanned by a line going
            North-West and North-East

        Parameters
        ----------
        closest_neighbours: List
            List of length 4 containing the index of closes neighbour in the corresponfing direction:
            [North-Neighbour, East-Neighbour, South-Neighbour, West-Neighbour]
        out_direction: int
            Direction we want to get city index from
            North: 0, East: 1, South: 2, West: 3

        Returns
        -------
        Returns the index of the closest neighbour in the desired direction. If none was present the neighbor clockwise
        or counter clockwise is returned
        """

        neighbour_idx = closest_neighbours[out_direction]
        if neighbour_idx is not None:
            return neighbour_idx

        neighbour_idx = closest_neighbours[(out_direction - 1) % 4]  # counter-clockwise
        if neighbour_idx is not None:
            return neighbour_idx

        neighbour_idx = closest_neighbours[(out_direction + 1) % 4]  # clockwise
        if neighbour_idx is not None:
            return neighbour_idx

        return closest_neighbours[(out_direction + 2) % 4]  # clockwise


    def _build_inner_cities(city_positions: IntVector2DArray, inner_connection_points: List[List[List[IntVector2D]]],
                            outer_connection_points: List[List[List[IntVector2D]]], rail_trans: RailEnvTransitions,
                            grid_map: GridTransitionMap) -> (List[IntVector2DArray], List[List[List[IntVector2D]]]):
        """
        Set the parallel tracks within the city. The center track of the city is of the length of the city, the lenght
        of the tracks decrease by 2 for every parallel track away from the center
        EG:

                ---     Left Track
               -----    Center Track
                ---     Right Track

        Parameters
        ----------
        city_positions: IntVector2DArray
                        All coordinates of the cities

        inner_connection_points: List[List[List[IntVector2D]]]
            Points on city boarder that are used to generate inner city track
        outer_connection_points: List[List[List[IntVector2D]]]
            Points where the city is connected to neighboring cities
        rail_trans: RailEnvTransitions
            Railway transition objects
        grid_map: RailEnvTransitions
            The grid map containing the rails. Used to draw new rails

        Returns
        -------
        Returns a list of all the cells (Coordinates) that belong to a rail paths within the city.
        """

        free_rails: List[List[List[IntVector2D]]] = [[] for i in range(len(city_positions))]
        for current_city in range(len(city_positions)):

            # This part only works if we have keep same number of connection points for both directions
            # Also only works with two connection direction at each city
            for i in range(4):
                if len(inner_connection_points[current_city][i]) > 0:
                    boarder = i
                    break

            opposite_boarder = (boarder + 2) % 4
            nr_of_connection_points = len(inner_connection_points[current_city][boarder])
            number_of_out_rails = len(outer_connection_points[current_city][boarder])
            start_idx = int((nr_of_connection_points - number_of_out_rails) / 2)
            # Connect parallel tracks
            for track_id in range(nr_of_connection_points):
                source = inner_connection_points[current_city][boarder][track_id]
                target = inner_connection_points[current_city][opposite_boarder][track_id]
                current_track = connect_straight_line_in_grid_map(grid_map, source, target, rail_trans)
                free_rails[current_city].append(current_track)

            for track_id in range(nr_of_connection_points):
                source = inner_connection_points[current_city][boarder][track_id]
                target = inner_connection_points[current_city][opposite_boarder][track_id]

                # Connect parallel tracks with each other
                fix_inner_nodes(
                    grid_map, source, rail_trans)
                fix_inner_nodes(
                    grid_map, target, rail_trans)

                # Connect outer tracks to inner tracks
                if start_idx <= track_id < start_idx + number_of_out_rails:
                    source_outer = outer_connection_points[current_city][boarder][track_id - start_idx]
                    target_outer = outer_connection_points[current_city][opposite_boarder][track_id - start_idx]
                    connect_straight_line_in_grid_map(grid_map, source, source_outer, rail_trans)
                    connect_straight_line_in_grid_map(grid_map, target, target_outer, rail_trans)
        return free_rails


    def _set_trainstation_positions(city_positions: IntVector2DArray, city_radius: int,
                                    free_rails: List[List[List[IntVector2D]]]) -> List[List[Tuple[IntVector2D, int]]]:
        """
        Populate the cities with possible start and end positions. Trainstations are set on the center of each paralell
        track. Each trainstation gets a coordinate as well as number indicating what track it is on

        Parameters
        ----------
        city_positions: IntVector2DArray
                        All coordinates of the cities
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        free_rails: List[List[List[IntVector2D]]]
            Cells that allow for trainstations to be placed

        Returns
        -------
        Returns a List[List[Tuple[IntVector2D, int]]] containing the coordinates of trainstations as well as their
        track number within the city
        """
        num_cities = len(city_positions)
        train_stations = [[] for i in range(num_cities)]
        for current_city in range(len(city_positions)):
            for track_nbr in range(len(free_rails[current_city])):
                possible_location = free_rails[current_city][track_nbr][
                    int(len(free_rails[current_city][track_nbr]) / 2)]
                train_stations[current_city].append((possible_location, track_nbr))
        return train_stations


    def _fix_transitions(city_cells: IntVector2DArray, inter_city_lines: List[IntVector2DArray],
                         grid_map: GridTransitionMap, vector_field):
        """
        Check and fix transitions of all the cells that were modified. This is necessary because we ignore validity
        while drawing the rails.

        Parameters
        ----------
        city_cells: IntVector2DArray
            Cells within cities. All of these might have changed and are thus checked
        inter_city_lines: List[IntVector2DArray]
            All cells within rails drawn between cities
        vector_field: IntVector2DArray
            Vectorfield of the size of the environment. It is used to generate preferred orienations for each cell.
            Each cell contains the prefered orientation of cells. If no prefered orientation is present it is set to -1
        grid_map: RailEnvTransitions
            The grid map containing the rails. Used to draw new rails

        """

        # Fix all cities with illegal transition maps
        rails_to_fix = np.zeros(3 * grid_map.height * grid_map.width * 2, dtype='int')
        rails_to_fix_cnt = 0
        cells_to_fix = city_cells + inter_city_lines
        for cell in cells_to_fix:
            try:
                cell_valid = grid_map.cell_neighbours_valid(cell, True)
            except:
                import pdb; pdb.set_trace()

            if not cell_valid:
                rails_to_fix[3 * rails_to_fix_cnt] = cell[0]
                rails_to_fix[3 * rails_to_fix_cnt + 1] = cell[1]
                rails_to_fix[3 * rails_to_fix_cnt + 2] = vector_field[cell]

                rails_to_fix_cnt += 1
        # Fix all other cells
        for cell in range(rails_to_fix_cnt):
            grid_map.fix_transitions((rails_to_fix[3 * cell], rails_to_fix[3 * cell + 1]), rails_to_fix[3 * cell + 2])


    def _closest_neighbour_in_grid4_directions(current_city_idx: int, city_positions: IntVector2DArray) -> List[int]:
        """
        Finds the closest city in each direction of the current city
        Parameters
        ----------
        current_city_idx: int
            Index of current city
        city_positions: IntVector2DArray
            Vector containing the coordinates of all cities

        Returns
        -------
        Returns indices of closest neighbour in every direction NESW
        """

        city_distances = []
        closest_neighbour: List[int] = [None for i in range(4)]

        # compute distance to all other cities
        for city_idx in range(len(city_positions)):
            city_distances.append(
                Vec2dOperations.get_manhattan_distance(city_positions[current_city_idx], city_positions[city_idx]))
        sorted_neighbours = np.argsort(city_distances)

        for neighbour in sorted_neighbours[1:]:  # do not include city itself
            direction_to_neighbour = direction_to_point(city_positions[current_city_idx], city_positions[neighbour])
            if closest_neighbour[direction_to_neighbour] is None:
                closest_neighbour[direction_to_neighbour] = neighbour

            # early return once all 4 directions have a closest neighbour
            if None not in closest_neighbour:
                return closest_neighbour

        return closest_neighbour


    def argsort(seq):
        """
        Same as Numpy sort but for lists
        Parameters
        ----------
        seq: List
            list that we would like to sort from smallest to largest

        Returns
        -------
        Returns the sorted list

        """
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__)


    def _get_cells_in_city(center: IntVector2D, radius: int, city_orientation: int,
                           vector_field: IntVector2DArray) -> IntVector2DArray:
        """
        Function the collect cells of a city. It also populates the vector field accoring to the orientation of the
        city.

        Example: City oriented north with a radius of 5, the vectorfield in the city will be as follows:
            |S|S|S|S|S|
            |S|S|S|S|S|
            |S|S|S|S|S|  <-- City center
            |N|N|N|N|N|
            |N|N|N|N|N|

        This is used to later orient the switches to avoid infeasible maps.

        Parameters
        ----------
        center: IntVector2D
            center coordinates of city
        radius: int
            radius of city (it is a square)
        city_orientation: int
            Orientation of city
        Returns
        -------
        flat list of all cell coordinates in the city

        """
        x_range = np.arange(center[0] - radius, center[0] + radius + 1)
        y_range = np.arange(center[1] - radius, center[1] + radius + 1)
        x_values = np.repeat(x_range, len(y_range))
        y_values = np.tile(y_range, len(x_range))
        city_cells = list(zip(x_values, y_values))
        for cell in city_cells:
            vector_field[cell] = align_cell_to_city(center, city_orientation, cell)
        return city_cells

    return generator