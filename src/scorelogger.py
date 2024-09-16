from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np


class ScoreLogger:
    def __init__(self, name, slices, slices_x):
        self.scores = deque(maxlen=None)
        self.name = name
        self.csv = f'./scores/{name}_scores.csv'
        self.png = f'./scores/{name}_scores.png'
        self.slices = slices
        self.slices_x = slices_x
        self.xrange_csv = f'./scores/{name}_xrange.csv'
        self.xrange_png = f'./scores/{name}_xrange.png'
        self.x_positions = deque(maxlen=None)

        if os.path.exists(self.png):
            os.remove(self.png)
        if os.path.exists(self.csv):
            os.remove(self.csv)
        if os.path.exists(self.xrange_png):
            os.remove(self.xrange_png)
        if os.path.exists(self.xrange_csv):
            os.remove(self.xrange_csv)
        
    def add_final_position(self, position, episode):
        self.final_positions.append(position)
        self._save_csv(self.positions_csv, position)
        if episode % self.slices == 0 and episode != 0:
            self._save_png(input_path=self.positions_csv,
                        output_path=self.positions_png,
                        x_label="episodes",
                        y_label="final x positions",
                        show_legend=True)
            
    def add_x_positions(self, x_positions, episode):
        min_x = min(x_positions)
        max_x = max(x_positions)
        self.x_positions.append((min_x, max_x))
        
        if episode % self.slices_x == 0 and episode != 0:
            self._save_csv_ranges(self.xrange_csv, [episode, min_x, max_x])
            self._save_xrange_png()

    def _save_xrange_png(self):
        episodes = []
        min_xs = []
        max_xs = []

        with open(self.xrange_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3:
                    episodes.append(int(row[0]))
                    min_xs.append(float(row[1]))
                    max_xs.append(float(row[2]))

        plt.subplots()
        plt.plot(episodes, min_xs, label="min x position", color="blue")
        plt.plot(episodes, max_xs, label="max x position", color="red")

        plt.title(f"{self.name} - X Position Range per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("X Position")
        plt.legend(loc="upper left")

        plt.savefig(self.xrange_png, bbox_inches="tight")
        plt.close()


    def add_score(self, score, episode, epsilon):
        self._save_csv(self.csv, score)
        if episode % self.slices == 0 and episode != 0:
            self._save_png(input_path=self.csv,
                        output_path=self.png,
                        x_label="episodes",
                        y_label="scores",
                        show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        #print(f'Episode: {episode} ==> [score: {score}, epsilon: {epsilon}, min: {min(self.scores)}, avg: {mean_score}, max: {max(self.scores)}] \n')

    def _save_png(self, input_path, output_path, x_label, y_label, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            input_lst = []
            for value in data:
                if(len(value)>0):
                    input_lst.append(value[0])
                
            output_lst = self.calculate_means(input_lst, self.slices)
            #print(output_lst)
            for i in range(0, len(output_lst)):
                x.append(int(i * self.slices))
                y.append(float(output_lst[i]))

        plt.subplots()
        plt.plot(x, y, label="score per episode")

        plt.plot(x, [np.mean(y)] * len(y), linestyle="--", label="average")

        plt.title(self.name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

    def _save_csv_ranges(self, path, data):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        with open(path, "a", newline='') as scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(data)


    def calculate_means(self, original_list, window_size=100):
        means_list = []
        num_chunks = len(original_list) // window_size

        for i in range(num_chunks):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            chunk = original_list[start_idx:end_idx]

            # Convert chunk elements to integers (or floats if necessary)
            chunk = [float(x) for x in chunk]  # Use float(x) if the values are floats

            mean = sum(chunk) / len(chunk)
            means_list.append(mean)

        return means_list
