from crossing_test_algorithm import Point, doIntersect, onSegment, orientation
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from viz_utils.utils import set_seed


class CrossingDataset:
    def __init__(self, num_arcs, name, balance=True, num_starting_node=5000):
        self.num_arcs = num_arcs
        self.global_dict_vector = []
        self.dataset_name = name
        self.DEBUG = False
        self.balance = balance
        self.counter_no = 0
        self.counter_yes = 0
        self.counter_starting_node = 0
        self.num_starting_node = num_starting_node

    def build(self):
        while True:
            p1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
            q1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
            p2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
            q2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

            if self.counter_starting_node < self.num_starting_node:

                node = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

                if self.counter_starting_node < self.num_starting_node // 4:
                    p1 = p2 = node
                    q1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
                    q2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

                elif (self.counter_starting_node > self.num_starting_node // 4) and \
                        (self.counter_starting_node < self.num_starting_node * (2 / 4)):
                    p1 = q2 = node
                    p2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
                    q1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

                elif (self.counter_starting_node > self.num_starting_node * (2 / 4)) and \
                        self.counter_starting_node < self.num_starting_node * (3 / 4):
                    q1 = p2 = node
                    p1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
                    q2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

                elif self.counter_starting_node > self.num_starting_node * (3 / 4):
                    q1 = q2 = node
                    p1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
                    p2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

                self.counter_starting_node += 1

            intersect_flag = doIntersect(p1, q1, p2, q2)
            el_dict = {"p1_x": p1.x, "p1_y": p1.y, "q1_x": q1.x, "q1_y": q1.y,
                       "p2_x": p2.x, "p2_y": p2.y, "q2_x": q2.x, "q2_y": q2.y, "intersect_target": 1 if intersect_flag
                                                                                                        is True else 0}
            if self.DEBUG:
                print(intersect_flag)
                plt.plot([p1.x, q1.x], [p1.y, q1.y], color='b', linestyle='-', linewidth=2)
                plt.plot([p2.x, q2.x], [p2.y, q2.y], color='k', linestyle='-', linewidth=2)
                plt.show()

            if intersect_flag:
                self.counter_yes += 1

            else:
                if self.counter_yes < self.counter_no - 10:  # if unbalanced, skip the not intersect
                    continue
                else:
                    self.counter_no += 1

            self.global_dict_vector.append(el_dict)
            if self.counter_yes + self.counter_no == self.num_arcs:
                break

    def save_to_disk(self, path):
        df = pd.DataFrame(self.global_dict_vector)
        df.to_csv(os.path.join(path, f"{self.dataset_name}.csv"))


if __name__ == "__main__":

    folder_save = "data"
    os.makedirs(folder_save, exist_ok=True)

    suffix = "starting_node"

    set_seed()

    az = CrossingDataset(num_arcs=100000, name=f"{suffix}_training_set", num_starting_node=5000)
    az.build()
    az.save_to_disk(folder_save)
    print("intersect origin: ", az.counter_starting_node)

    val = CrossingDataset(num_arcs=20000, name=f"{suffix}_validation_set", num_starting_node=1000)
    val.build()
    val.save_to_disk(folder_save)
    print("intersect origin: ", val.counter_starting_node)
    test = CrossingDataset(num_arcs=50000, name=f"{suffix}_test_set", num_starting_node=2500)
    test.build()
    test.save_to_disk(folder_save)
    print("intersect origin: ", test.counter_starting_node)
