import os
import random
from sklearn.model_selection import train_test_split
import pickle
from viz_utils.utils import set_seed

set_seed(1234)

list_to_avoid = ["grafo6353.39.graphml", "grafo7223.40.graphml", "grafo7312.41.graphml"]
path_to_rome = os.path.join("data", "rome")

graph_files = os.listdir(path_to_rome)

graph_files = random.sample(graph_files, 100)

print("not_connected.txt" in graph_files)

for i in graph_files:
    if ".graphml" not in i:
        print(f"Removing {i}")
        graph_files.remove(i)

    if i in list_to_avoid:
        print(list_to_avoid)
        print(f"Removing {i}")
        graph_files.remove(i)

try:
    graph_files.remove("not_connected.txt")
    graph_files.remove("training_list")
    graph_files.remove("test_list")
    graph_files.remove("validation_list")
    graph_files.remove("training_list100")
    graph_files.remove("test_list100")
    graph_files.remove("validation_list100")
    graph_files.remove("training_list1000")
    graph_files.remove("test_list1000")
    graph_files.remove("validation_list1000")
    print("Conn removed!")
except:
    print("cannot find Not connected ")

print(len(graph_files))
print("not_connected.txt" in graph_files)

# training_set = random.sample(graph_files, 10000)
# remaining_set = [x for x in graph_files if x not in training_set]
#
# test_set = random.sample(remaining_set, 1000)
# validation = [x for x in remaining_set if x not in test_set]
#
# print(len(training_set))
# print(len(test_set))
# print(len(validation))
#
# print("Check if there are elem in common")
#
# common_tr_test = list(set(training_set).intersection(test_set))
# common_tr_val = list(set(training_set).intersection(validation))
# common_valr_test = list(set(validation).intersection(test_set))

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

X_train, X_test = train_test_split(graph_files, test_size=1 - train_ratio)
X_test, X_validate = train_test_split(X_test, test_size=test_ratio / (test_ratio + validation_ratio))

print(
    f"Len train: {len(X_train)}, len test: {len(X_test)}, len valid:{len(X_validate)}; total length: {len(X_train) + len(X_validate) + len(X_test)}")

with open(os.path.join(path_to_rome, "training_list100"), 'wb') as fp:
    pickle.dump(X_train, fp)
with open(os.path.join(path_to_rome, "validation_list100"), 'wb') as fp:
    pickle.dump(X_validate, fp)
with open(os.path.join(path_to_rome, "test_list100"), 'wb') as fp:
    pickle.dump(X_test, fp)
