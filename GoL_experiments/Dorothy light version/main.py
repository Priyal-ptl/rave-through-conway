import collections

from dorothy import Dorothy

dot = Dorothy(width=384, height=384, title="Feedback")

class Sketch:

    alive_cells: set[tuple[int, int]]
    counter: int = 0

    def setup(self):
        self.alive_cells = set([tuple(cell) for cell in [[2, 1], [2, 2], [2, 3]]])
        print(self.alive_cells)

    def on_key(self, key, action, mods):
        pass

    def draw(self):
        dot.background((0, 0, 0))

        feedback_amount = dot.mouse_y / dot.height
        
        if self.counter % 100 == 0: 
            self.evolve()
            print(self.alive_cells)
        self.counter += 1

    def evolve(self):
        neighbors = (
            (-1, -1),  # Above left
            (-1, 0),  # Above
            (-1, 1),  # Above right
            (0, -1),  # Left
            (0, 1),  # Right
            (1, -1),  # Below left
            (1, 0),  # Below
            (1, 1),  # Below right
        )
        num_neighbors = collections.defaultdict(int)
        for row, col in self.alive_cells:
            for drow, dcol in neighbors:
                num_neighbors[(row + drow, col + dcol)] += 1

        stay_alive = {
            cell for cell, num in num_neighbors.items() if num in {2, 3}
        } & self.alive_cells
        come_alive = {
            cell for cell, num in num_neighbors.items() if num == 3
        } - self.alive_cells

        self.alive_cells = stay_alive | come_alive

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
