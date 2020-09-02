class Agent:
    def __init__(self):
        self.algorithm = ''
        self.board_size = None
        self.start_grid = None
        self.goal_grid = None
        self.number_grids = 0
        self.grids_dict = {}
        self.graph = {}

    def input_data(self, input_path):
        """
        Parse the input txt correctly
        :param input_path: file path
        :return:
        """
        with open(input_path, 'r') as file:
            self.algorithm = file.readline().rstrip("\n")
            self.board_size = tuple(map(int, file.readline().rstrip("\n").split(" ")))
            self.start_grid = tuple(map(int, file.readline().rstrip("\n").split(" ")))
            self.goal_grid = tuple(map(int, file.readline().rstrip("\n").split(" ")))
            self.number_grids = int(file.readline().rstrip("\n"))
            # Record all grids with actions in a dict eg {(1, 3, 1): [5], (1, 3, 2): [1, 11], ...}
            for line in file:
                line_temp = tuple(map(int, line.rstrip("\n").split(" ")))
                self.grids_dict[(line_temp[0], line_temp[1], line_temp[2])] = list(line_temp[3:])
        return

    def check_valid_grid(self, grid):
        """
        Check if grid is valid and not out of bounds
        :param grid: tuple (0, 0, 0)
        :return: Valid or not
        """
        if 0 < grid[0] < self.board_size[0] and 0 < grid[1] < self.board_size[1] and 0 < grid[2] < self.board_size[2]:
            return True
        return False

    def create_graph(self):
        """
        Create the graph for the grid
        eg for Example 1
        {(1, 3, 1): {(1, 3, 2): 1}, (1, 3, 2): {(2, 3, 2): 1, (2, 3, 3): 1}, (2, 3, 2): {(2, 3, 3): 1, (3, 3, 3): 1},
         (2, 3, 3): {(2, 4, 2): 1}, (2, 4, 2): {(3, 4, 3): 1}, (3, 4, 3): {}, ... }
        :return:
        """
        code_dict = {1: (1, 0, 0), 2: (-1, 0, 0), 3: (0, 1, 0), 4: (0, -1, 0), 5: (0, 0, 1),
                     6: (0, 0, -1), 7: (1, 1, 0), 8: (1, -1, 0), 9: (-1, 1, 0), 10: (-1, -1, 0),
                     11: (1, 0, 1), 12: (1, 0, -1), 13: (-1, 0, 1), 14: (-1, 0, -1), 15: (0, 1, 1),
                     16: (0, 1, -1), 17: (0, -1, 1), 18: (0, -1, -1)}
        for grid, actions in self.grids_dict.items():
            # Add a node for each lines if not exist, also check validity
            if grid not in self.graph and self.check_valid_grid(grid):
                self.graph[grid] = {}
            # For every action, add a node for destination if not exist
            for action in actions:
                new_grid_loc = tuple(map(sum, zip(grid, code_dict[action])))
                if self.check_valid_grid(grid):
                    if new_grid_loc not in self.graph:
                        self.graph[new_grid_loc] = {}

                # Assign weight and create an edge
                if self.algorithm == "BFS":
                    self.graph[grid][new_grid_loc] = 1
        return


if __name__ == '__main__':
    input_path = "input.txt"
    agent = Agent()
    agent.input_data(input_path)
    agent.create_graph()
    print()