import itertools
import random
import time
import warnings
from multiprocessing import Pool

import mlflow
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

warnings.filterwarnings('ignore')


def generate_dm(df, show=True):
    temp = df[[0, 1]].to_numpy()
    dm = distance_matrix(temp, temp)

    temp = df[2].to_numpy() // 2
    temp = temp * np.ones((200, 200))
    dm = dm + temp + temp.T
    dm = dm // 1

    for i in range(200):
        dm[i][i] = np.inf

    if show:
        df_dist = pd.DataFrame(dm)
        # display(df_dist)
    return dm


def get_random_solution():
    return random.sample([i for i in range(0, 200)], 100)


def calulate_total_cost(path, dm):
    total = 0
    nr = len(path)
    for idx, node in enumerate(path):
        total += dm[node][path[(idx + 1) % nr]]
    return total


def generate_inter_moves(path, dm):
    not_included_ids = list(set([i for i in range(200)]) - set(path))
    inter_moves = []
    for idx, node in enumerate(path):
        for new_node in not_included_ids:
            new_path = path[:idx] + [new_node] + path[idx + 1:]

            new_cost = dm[path[idx - 1]][new_node] + dm[path[(idx + 1) % 100]][new_node]  # two new edges
            new_cost -= (dm[path[idx - 1]][node] + dm[path[(idx + 1) % 100]][node])  # two old edges
            # we want to min new_cost

            inter_moves.append((new_path, new_cost))
    return inter_moves


def generate_intra_moves(path, type_of_change, dm):
    intra_moves = []
    if type_of_change == 'node':
        for idx1, idx2 in all_possible_comb:
            new_path = path[:idx1] + [path[idx2]] + path[idx1 + 1:idx2] + [path[idx1]] + path[idx2 + 1:]
            new_cost = dm[path[idx1]][path[(idx2 + 1) % 100]] + dm[path[idx1]][path[idx2 - 1]]  # new edges for node1
            new_cost += dm[path[idx2]][path[(idx1 + 1) % 100]] + dm[path[idx2]][path[idx1 - 1]]  # new edges for node2
            new_cost -= (dm[path[idx1]][path[idx1 - 1]] + dm[path[idx1]][path[(idx1 + 1) % 100]])  # old edges for node1
            new_cost -= (dm[path[idx2]][path[idx2 - 1]] + dm[path[idx2]][path[(idx2 + 1) % 100]])  # old edges for node2
            intra_moves.append((new_path, new_cost))
    else:  # edge

        for idx1, idx2 in all_possible_comb:
            if (idx2 + 1) % 100 != idx1:
                new_path = path[:idx1] + path[idx1:idx2 + 1][::-1] + path[idx2 + 1:]  # %100 is const.
                #                 new_path = path[:idx1] + path[idx1:idx2][::-1] + path[idx2:] # %100 is const.
                new_cost = dm[path[idx1]][path[(idx2 + 1) % 100]] + dm[path[idx1 - 1]][path[idx2]]  # new edges
                new_cost -= (dm[path[idx1 - 1]][path[idx1]] + dm[path[idx2]][path[(idx2 + 1) % 100]])  # old edges
                intra_moves.append((new_path, new_cost))

    return intra_moves


def main(type_of_change, greedy, path, dm):
    change = True
    max_id = dm.shape[0]

    while change:
        change = False
        # possible_moves [(new_path, delta)]
        # inter: change set of selected nodes(one in ine out)
        # intra: change order of nodes
        # a) exchange 2 nodes
        # b) exchange 2 edges
        possible_moves = generate_inter_moves(path, dm) + generate_intra_moves(path, type_of_change, dm)
        if greedy:
            random.shuffle(possible_moves)
            for new_path, new_cost in possible_moves:
                if new_cost < 0:
                    path = new_path
                    change = True
                    break
        else:
            possible_moves.sort(key=lambda x: x[1])
            if possible_moves[0][1] < 0:
                path = possible_moves[0][0]
                change = True

    return path


def f(start_node_idx, dm, data, weight_regret=1, weight_cost=0, return_cycle=False):
    data_indexes = data.index.to_list()
    if start_node_idx is None:
        start_node_idx = random.choice(data_indexes)

    node = data_indexes[start_node_idx]
    next_node = dm[node].argmin()

    data_indexes.remove(node)
    data_indexes.remove(next_node)

    edge1 = str(node) + '_' + str(next_node)
    edge2 = str(next_node) + '_' + str(node)
    cycle = [edge1, edge2]

    # for, sort1, sort2, total
    while len(cycle) < data.shape[0] // 2:
        store = []

        for node in data_indexes:
            options = []
            for edge in cycle:
                old_node1, old_node2 = [int(i) for i in edge.split('_')]
                change = dm[old_node1][node] + dm[old_node2][node] - dm[old_node1][old_node2]
                options.append((edge, change))
            options.sort(key=lambda x: x[1])  # here
            regret = options[0][1] - options[1][1]
            value = weight_regret * regret + weight_cost * options[0][1]
            store.append((node, options[0][0], value))

        store.sort(key=lambda x: x[2])

        best = store[0]
        cycle.remove(best[1])
        data_indexes.remove(best[0])
        old_node1, old_node2 = [int(i) for i in best[1].split('_')]
        cycle.append(str(old_node1) + '_' + str(best[0]))
        cycle.append(str(best[0]) + '_' + str(old_node2))

    total = 0
    for edge in cycle:
        node1, node2 = [int(i) for i in edge.split('_')]
        total += dm[node1][node2]

    if return_cycle:
        return total, cycle
    else:
        return total


def run(type_of_change, greedy, path, dm, file_name, start_node_idx=None):
    mlflow.set_tracking_uri('https://mlflow.dev.eosc.pcss.pl')
    mlflow.set_experiment('studia')

    with mlflow.start_run():
        start_time = time.time()
        final_path = main(type_of_change, greedy, path, dm)
        final_cost = calulate_total_cost(final_path, dm)

        mlflow.log_param("type_of_change", type_of_change)
        mlflow.log_param("greedy", greedy)
        mlflow.log_param("file_name", file_name)
        mlflow.log_metric("time", time.time() - start_time)
        mlflow.log_metric("cost", final_cost)
        mlflow.log_param("lab", 3)
        if start_node_idx is not None:
            mlflow.log_param("start_node_idx", start_node_idx)


def old_run(id, dm, tsp, file_name):
    mlflow.set_tracking_uri('hidden')
    mlflow.set_experiment('studia')

    with mlflow.start_run():
        start_time = time.time()
        score, cycle = f(id, dm, tsp, 1, 0, True)

        mlflow.log_metric("time", time.time() - start_time)
        mlflow.log_metric("cost", score)
        mlflow.log_param("lab", 2)
        mlflow.log_param("start_node_idx", id)
        mlflow.log_param("file_name", file_name)
    path = [int(x.split('_')[0]) for x in cycle]
    return path


all_possible_comb = list(itertools.combinations([i for i in range(100)], 2))
if __name__ == '__main__':

    t = time.time()

    # dm = generate_dm(tsp_c)
    for file_name in ['TSPC.csv', 'TSPD.csv']:
        tsp = pd.read_csv(file_name, sep=';', header=None)
        dm = generate_dm(tsp)

        for path_type in ['random', 'old']:
            for type_of_change in ['node', 'edge']:

                for greedy in [True, False]:

                    print(file_name, type_of_change, path_type, greedy)
                    pool = Pool(processes=6)
                    for i in range(200):
                        if path_type == 'random':
                            path = get_random_solution()
                            pool.apply_async(run, args=(type_of_change, greedy, path, dm, file_name))
                            # print(res.get())
                        else:
                            path = old_run(i, dm, tsp, file_name)
                            pool.apply_async(run, args=(type_of_change, greedy, path, dm, file_name, i,))
                            # print(res.get())
                    pool.close()
                    pool.join()

    print(time.time() - t)
