from collections import namedtuple
import numpy as np


"""
    The co-ordinate system is (row, column) of the track map:
                    --------- COLUMN ------->
       |    [ (0, 0),           (0, 1),         ...,    (0, size-1) ]
      ROW   [ (1, 0),           (1, 1),         ...,    (1, size-1) ]
       |    [  ...,               ...,          ...,         ...    ]
       v    [ (size-1, 0),    (size-1, 1),      ...,    (size-1, size-1) ]

    The headings are:
        (1, 0):         increase in row number          DOWN
        (0, 1):         increase in column number       RIGHT
        (-1, 0):        decrease in row number          UP
        (0, -1):        decrease in column number       LEFT
    """


class Heading:
    def __init__(self, row, column):
        if row != 0 and column != 0:
            raise ValueError(f'Tried to create a heading with both row and column non-zero: {(row, column)}')

        self._row = np.sign(row)
        self._column = np.sign(column)

    @property
    def row(self):
        return self._row

    @property
    def column(self):
        return self._column

    def __hash__(self):
        return hash(('heading', self.row, self.column))

    def __eq__(self, other):
        return type(other) == type(self) and self.row == other.row and self.column == other.column

    def __str__(self):
        return f'Heading(row={self.row}, column={self.column})'

    def get_left_heading(self):
        current_index = heading_left_turn_cycle.index(self)
        new_heading = heading_left_turn_cycle[(current_index + 1) % 4]
        return Heading(new_heading.row, new_heading.column)  # don't want to pass out the reference

    def get_right_heading(self):
        current_index = heading_left_turn_cycle.index(self)
        new_heading = heading_left_turn_cycle[(current_index - 1) % 4]
        return Heading(new_heading.row, new_heading.column)  # don't want to pass out the reference

    def get_reverse_heading(self):
        return Heading(-self.row, -self.column)

    def over_rows(self):
        return self.row != 0

    def rotate_from_track_to_car(self, track_left, track_right, track_up, track_down):
        # Rotate into car heading from track co-ordinates
        if 1 == self.row:  # (1, 0) -> facing down the rows (to larger row number)
            ahead, left, behind, right = track_down, track_right, track_up, track_left
        elif 1 == self.column:  # (0, 1) -> facing right across columns (to larger column number)
            ahead, left, behind, right = track_right, track_up, track_left, track_down
        elif -1 == self.row:  # (-1, 0) -> facing up the rows (to smaller row number)
            ahead, left, behind, right = track_up, track_left, track_down, track_right
        else:  # (0, -1) -> facing left across columns (to smaller column number)
            ahead, left, behind, right = track_left, track_down, track_right, track_up
        return ahead, left, behind, right

    def rotate_from_car_to_track(self, track_state):
        distances = [track_state.distance_ahead, track_state.distance_left, track_state.distance_behind,
                     track_state.distance_right]
        if 1 == self.row:  # (1, 0) -> facing down the rows (to larger row number)
            track_down, track_right, track_up, track_left = distances
        elif 1 == self.column:  # (0, 1) -> facing right across columns (to larger column number)
            track_right, track_up, track_left, track_down = distances
        elif -1 == self.row:  # (-1, 0) -> facing up the rows (to smaller row number)
            track_up, track_left, track_down, track_right = distances
        else:  # (0, -1) -> facing left across columns (to smaller column number)
            track_left, track_down, track_right, track_up = distances
        return track_left, track_right, track_up, track_down

    @staticmethod
    def get_all_headings():
        return heading_left_turn_cycle

    def to_json_compatible(self):
        return {'row': int(self.row), 'column': int(self.column)}

    @staticmethod
    def from_json_compatible(json_dict):
        return Heading(row=json_dict['row'], column=json_dict['column'])


heading_left_turn_cycle = [Heading(1, 0), Heading(0, 1), Heading(-1, 0), Heading(0, -1)]


class Position:
    def __init__(self, row, column):
        if row < 0 or column < 0:
            raise ValueError(f"Position can't have a negative row or column: {(row, column)}")

        self._position = (int(row), int(column))

    def __str__(self):
        return f'Position(row={self.row}, column={self.column})'

    def __hash__(self):
        return hash(('position', self.row, self.column))

    def __eq__(self, other):
        return type(other) == type(self) and self.row == other.row and self.column == other.column

    @property
    def row(self):
        return self._position[0]

    @property
    def column(self):
        return self._position[1]

    def get_new_position(self, heading: Heading, distance: int):
        if heading not in heading_left_turn_cycle:
            raise ValueError(f'Unknown heading {heading}. Must be one of {heading_left_turn_cycle}')
        if distance < 0:
            raise ValueError(f'Distance must be greater than 0 but is {distance}')
        return Position(self.row + distance * heading.row, self.column + distance * heading.column)

    def distance_to(self, other):
        # Simple Euclidean distance
        return np.sqrt((self.row - other.row)**2 + (self.column - other.column)**2)

    def copy(self):
        return Position(self.row, self.column)

    def to_json_compatible(self):
        return {'row': self.row, 'column': self.column}

    @staticmethod
    def from_json_compatible(json_dict):
        return Position(row=json_dict['row'], column=json_dict['column'])
