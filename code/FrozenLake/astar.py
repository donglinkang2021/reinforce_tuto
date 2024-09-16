
import heapq
from typing import Tuple, List, Dict

__all__ = ["astar", "path2table"]

class Node:
    def __init__(self, x:int, y:int, parent=None, target=None, alpha=0.45):
        self.x = x
        self.y = y
        # the cost of the path from the start node to the current node
        steps_done = parent.steps_done + 1 if parent else 0  
        # the heuristic cost from the current node to the target node
        steps_rest = abs(x - target.x) + abs(y - target.y) if target else 0 # manhattan distance
        self.steps_done = steps_done
        self.f = steps_done * alpha + steps_rest * (1 - alpha)
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def backtrace(node:Node):
    return backtrace(node.parent) + [(node.x, node.y)] if node else []

def astar(
    board:List[str],
    start:Tuple[int, int],
    end:Tuple[int, int],
):
    n_rows, m_cols = len(board), len(board[0])
    open_list = []
    closed_list = set()

    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])

    def is_end(node:Node) -> bool:
        return node.x == end_node.x and node.y == end_node.y
    
    def is_valid(x:int, y:int) -> bool:
        return 0 <= x < n_rows and \
            0 <= y < m_cols and \
            board[x][y] != "H" and \
            (x, y) not in closed_list
    
    def not_in_open_list(node:Node) -> bool:
        return (node.x, node.y) not in [(n.x, n.y) for n in open_list]

    heapq.heappush(open_list, start_node)
    while open_list:
        current_node = heapq.heappop(open_list)
        if is_end(current_node): break
        closed_list.add((current_node.x, current_node.y))
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in neighbors:
            neighbor_x = current_node.x + dx
            neighbor_y = current_node.y + dy
            if is_valid(neighbor_x, neighbor_y):
                neighbor_node = Node(
                    neighbor_x, 
                    neighbor_y, 
                    current_node, 
                    end_node
                )
                if not_in_open_list(neighbor_node):
                    heapq.heappush(
                        open_list, 
                        neighbor_node
                    )
    return backtrace(current_node) if is_end(current_node) else None

def path2table(path:List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """
    Convert a path to a dictionary of actions.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    """
    table = {}
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        if x1 > x0:
            table[(x0, y0)] = 1
        elif x1 < x0:
            table[(x0, y0)] = 3
        elif y1 > y0:
            table[(x0, y0)] = 2
        else:
            table[(x0, y0)] = 0
    return table

if __name__ == "__main__":
    from utils import generate_random_map
    size = 8
    p = 0.8
    seed = 42
    board = generate_random_map(size, p, seed)
    print(board)
    path = astar(board, (0, 0), (size-1, size-1))
    print(path)
    table = path2table(path)
    print(table)

# python code/FrozenLake/astar.py