from enum import Enum

import numpy

class IntermediateCorner(Enum):
    NE = 0
    NW = 1
    SW = 2
    SE = 3
       
def corner_directions_border_outward_by_lte(raster: numpy.ndarray, cell_location: tuple[int, int], lte: float) -> list[IntermediateCorner]:
    """Check which corner direction for a cell are bordered by cell values less than or equal to a specified value

    :param raster: elevation raster (bottom)
    :param cell_location: Target cell location in Y,X order
    :type cell_location: tuple[int, int]
    :param lte: less than or equal value to compare
    :type lte: float
    :return: _description_
    :rtype: IntermediateCorner
    """
    # Corners to check in form of cell locations to check and the corner direction
    corners_to_check: list[tuple[list[tuple[int, int]], IntermediateCorner]] = []
    
    # corners that border outward with lte cells
    corners_match: list[IntermediateCorner] = []
    
    # Create corner tuples
    NE_check_locations = [(cell_location[0]-1, cell_location[1]), (cell_location[0]-1, cell_location[1]+1), (cell_location[0], cell_location[1]+1)]
    corners_to_check.append((NE_check_locations, IntermediateCorner.NE))
    
    NW_check_locations = [(cell_location[0]-1, cell_location[1]), (cell_location[0]-1, cell_location[1]-1), (cell_location[0], cell_location[1]-1)]
    corners_to_check.append((NW_check_locations, IntermediateCorner.NW))
    
    SW_check_locations = [(cell_location[0]+1, cell_location[1]), (cell_location[0]+1, cell_location[1]-1), (cell_location[0], cell_location[1]-1)]
    corners_to_check.append((SW_check_locations, IntermediateCorner.SW))
    
    SE_check_locations = [(cell_location[0]+1, cell_location[1]), (cell_location[0]+1, cell_location[1]+1), (cell_location[0], cell_location[1]+1)]
    corners_to_check.append((SE_check_locations, IntermediateCorner.SE))
    
    for ctc in corners_to_check:
        all_corners_lte = True
        for cl in ctc[0]:
            if cl[0] >= 0 and cl[0] < raster.shape[0] and cl[1] >= 0 and cl[1] < raster.shape[1]:
                if raster[cl] > lte:
                    all_corners_lte = False
                
        if all_corners_lte:
            corners_match.append(ctc[1])
    
    return corners_match

def z0_nudge_corners_from_source_raster(
    raster: numpy.ndarray,
    cell_location: tuple[int, int],
    zero_threshold: float = 0,
) -> list[IntermediateCorner]:
    """Return Z0 nudge corners from source-cell contributor windows.

    The target cell must be at or below ``zero_threshold``. The cell is
    considered only when at least one corner's 2x2 contributor window sees a
    value above that threshold. A corner needs nudging when its own contributor
    window does not see a value above the threshold.
    """
    row, col = cell_location
    if row < 0 or row >= raster.shape[0] or col < 0 or col >= raster.shape[1]:
        return []

    current_value = raster[row, col]
    if numpy.isnan(current_value) or current_value > zero_threshold:
        return []

    corner_windows = [
        (
            IntermediateCorner.NW,
            [(row - 1, col - 1), (row - 1, col), (row, col - 1), (row, col)],
        ),
        (
            IntermediateCorner.NE,
            [(row - 1, col), (row - 1, col + 1), (row, col), (row, col + 1)],
        ),
        (
            IntermediateCorner.SW,
            [(row, col - 1), (row, col), (row + 1, col - 1), (row + 1, col)],
        ),
        (
            IntermediateCorner.SE,
            [(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)],
        ),
    ]

    window_has_positive: dict[IntermediateCorner, bool] = {}
    for corner, window in corner_windows:
        has_positive = False
        for check_row, check_col in window:
            if (
                check_row < 0
                or check_row >= raster.shape[0]
                or check_col < 0
                or check_col >= raster.shape[1]
            ):
                continue
            value = raster[check_row, check_col]
            if not numpy.isnan(value) and value > zero_threshold:
                has_positive = True
                break
        window_has_positive[corner] = has_positive

    if not any(window_has_positive.values()):
        return []

    return [
        corner
        for corner, has_positive in window_has_positive.items()
        if not has_positive
    ]

def find_middle_corner(corners: list[IntermediateCorner]) -> IntermediateCorner:
    """Return the spatially middle corner of 3 corners

    :param corners: 3 corners that border outward with values that are Z=0
    :type corners: list[IntermediateCorner]
    :return: The middle corner
    :rtype: IntermediateCorner
    """
    if len(corners) != len(set(corners)):
        raise ValueError(f"Expected 3 corners, got {len(corners)} unique corners")
    
    if len(corners) != 3:
        raise ValueError(f"Expected 3 corners, got {len(corners)} corners")
    
    corners = sorted(corners, key=lambda p:p.value)
    
    # If sorted order is NE,0 SW,2 SE,3
    if corners[1].value - corners[0].value != 1:
        return corners[2]
    
    # If sorted order is NE,0 NW,1 SE,3
    if corners[0] == IntermediateCorner.NE and corners[2].value - corners[1].value != 1:
        return corners[0]
    
    return corners[1]
