from collections import deque
import queue


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
            assert len(self.grids_dict) == self.number_grids
        return

    def check_valid_grid(self, grid):
        """
        Check if grid is valid and not out of bounds
        :param grid: tuple (0, 0, 0)
        :return: Valid or not
        """
        if 0 <= grid[0] <= self.board_size[0] and 0 <= grid[1] <= self.board_size[1] and 0 <= grid[2] <= self.board_size[2]:
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

            # For every action, find the destination, check if present in input grid list, add a node for destination if not exist
            for action in actions:
                new_grid_loc = tuple(map(sum, zip(grid, code_dict[action])))

                # If destination not in grid list, it has no action and will not be right answer, so do nothing
                if new_grid_loc in self.grids_dict:
                    if new_grid_loc not in self.graph:
                        self.graph[new_grid_loc] = {}

                    # Assign weight and create an edge
                    if self.algorithm == "BFS":
                        self.graph[grid][new_grid_loc] = 1
                    else:
                        if 1 <= action <= 6:
                            self.graph[grid][new_grid_loc] = 10
                        else:
                            self.graph[grid][new_grid_loc] = 14

        assert len(self.graph) == self.number_grids
        return

    def search(self):
        """
        What search algorithm to use?
        :return: result path
        """
        if self.algorithm == 'BFS':
            return self.bfs()
        else:
            return self.ucs()

    def bfs(self):
        """
        Do BFS
        :return: Answer and total cost. Empty list and 0 for no solution
        """
        visited = set()
        queue = deque()
        parent = {}

        # If entrance/goal grid not in graph, I should return FAIL
        if self.start_grid not in self.graph or self.goal_grid not in self.graph:
            return [], 0  # Empty list means FAIL
        # If entrance is goal
        if self.start_grid == self.goal_grid:
            return [(self.start_grid, 0)], 0

        queue.append(self.start_grid)
        visited.add(self.start_grid)
        while queue:
            s = queue.popleft()
            for t in self.graph[s]:
                if t not in visited:
                    queue.append(t)
                    visited.add(t)
                    parent[t] = s

                    # Goal test
                    if t == self.goal_grid:
                        path = self.backtrack(parent)
                        result_cost, total_cost = self.compute_cost(path)
                        return result_cost, total_cost
        return [], 0

    @staticmethod
    def heuristic(current_node, goal_node):
        """
        Compute straight line distance sqrt(a^2 + b^2 + c^2)
        :param current_node:
        :param goal_node:
        :return:
        """
        result = 0
        for i in range(0, 3):
            result += (current_node[i] - goal_node[i]) ** 2
        return result ** 0.5 * 10

    def ucs(self):
        """
        Djikstra algorithm for UCS or A*
        :return:
        """
        # If entrance/goal grid not in graph, I should return FAIL
        if self.start_grid not in self.graph or self.goal_grid not in self.graph:
            return [], 0  # Empty list means FAIL
        # If entrance is goal
        if self.start_grid == self.goal_grid:
            return [(self.start_grid, 0)], 0

        dist = {}
        MAX_DIST = 2 ** 31
        pq = queue.PriorityQueue()
        for node in self.graph:
            if node != self.start_grid:
                dist[node] = MAX_DIST
                pq.put((dist[node], node))
        parent = {}
        dist[self.start_grid] = 0
        pq.put((dist[self.start_grid], self.start_grid))

        while not pq.empty():
            source = pq.get()
            s = source[1]
            for t, cost in self.graph[s].items():
                new_dist = cost + dist[s]
                if new_dist < dist[t]:
                    dist[t] = new_dist
                    parent[t] = s

                    # Remove and replace queue with lower cost priority
                    try:
                        index_del = [x[1] for x in pq.queue].index(t)
                        del pq.queue[index_del]
                    except ValueError:
                        pass
                    pq.put((dist[t], t))

                # Goal test, also need to check if cost to t is less than MAX_DIST
                # early exit should be OK because two steps will not be better than one step
                if t == self.goal_grid and dist[t] < MAX_DIST:
                    path = self.backtrack(parent)
                    result_cost, total_cost = self.compute_cost(path)
                    return result_cost, total_cost
        return [], 0

    def backtrack(self, parent):
        """
        If goal test is positive, backtrack thru parent dict and find the route
        :param self:
        :param parent:
        :return: Answer
        """
        result = []
        current_node = self.goal_grid
        result.append(current_node)
        while current_node != self.start_grid:
            parent_node = parent[current_node]
            current_node = parent_node
            result.insert(0, current_node)
        return result

    def compute_cost(self, result):
        """
        Compute cost of each step and save to kv pair with the path, also total cost
        :param result: Result path
        :return: Result path with costs, total cost
        """
        if len(result) == 0:  # If FAIL
            return []
        total_cost = 0
        result_cost = [(result[0], 0)]  # Start with start grid with 0 cost
        for i in range(1, len(result)):
            if self.algorithm == "BFS":
                result_cost.append((result[i], 1))  # All steps has 1 cost
                total_cost += 1
            else:
                # Calculate cost. Either 10 or 14.
                cost = 10
                if abs(result[i][0] - result[i - 1][0]) + abs(result[i][1] - result[i - 1][1]) + abs(result[i][2] - result[i - 1][2]) > 1:
                    cost = 14
                result_cost.append((result[i], cost))
                total_cost += cost
        return result_cost, total_cost

    @staticmethod
    def output_to_file(result_cost, total_cost):
        """
        Output results
        :param result_cost:
        :param total_cost:
        :return:
        """
        if len(result_cost) == 0:
            with open("output.txt", "w+") as file:
                file.write("FAIL")
            return
        with open("output.txt", "w+") as file:
            file.write(str(total_cost))
            file.write("\n")
            file.write(str(len(result_cost)))
            file.write("\n")
            for node in result_cost:
                file.write(" ".join([str(node[0][0]), str(node[0][1]), str(node[0][2]), str(node[1])]))
                file.write("\n")


if __name__ == '__main__':
    input_path = "input.txt"
    agent = Agent()
    agent.input_data(input_path)
    agent.create_graph()
    result_cost, total_cost = agent.search()
    agent.output_to_file(result_cost, total_cost)

