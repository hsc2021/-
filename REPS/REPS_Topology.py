from qns.entity.node.app import Application
from qns.network.topology import Topology
from qns.utils.rnd import get_rand
from typing import Dict, List, Tuple
from qns.entity.node import QNode
from qns.entity.qchannel import QuantumChannel
import itertools
import math


def get_destenation(node1_x, node1_y, node2_x, node2_y):
    return math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)


class SessionInfo:
    """
    SessionInfo is a class to store the information of SD pair.
    """

    def __init__(self, id: int, sd: Tuple[QNode, QNode], W: int):
        self.id = id
        self.sd = sd
        self.W = W


class REPS_Topo(Topology):
    """
    RepsTopology is a class that generates a Waxman random graph.includes
    nodes_number Qnodes
    size(float): the size of the network
    """

    def __init__(self, nodes_number: int, size: float,
                 nodes_apps: List[Application] = [], qchannel_args: Dict = {},
                 cchannel_args: Dict = {}, qmemory_args: Dict = {},) -> None:
        super().__init__(nodes_number, nodes_apps, qchannel_args, cchannel_args, qmemory_args)
        # set_seed(int(time.time()))
        self.size = size

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []
        location_table = {}
        distance_table = [[]]

        for i in range(self.nodes_number):
            pos_x = get_rand(0, 100)  # Assuming size is 100 for the example
            pos_y = get_rand(0, 100)
            node = QNode(name=f"n{i+1}")
            location_table[node] = [pos_x, pos_y]
            nl.append(node)

        # Waxman link pob
        # link = REPS_QuantumChannel(src=nl[0], dest=nl[1], EL_succ_prob=0.9)
        # ll.append(link)
        iter_indexs = list(itertools.combinations(range(self.nodes_number), 2))
        # print("iter", iter_indexs)
        L = 0
        distance_table = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
        for i, j in iter_indexs:
            distance = get_destenation(location_table[nl[i]][0], location_table[nl[i]][1], location_table[nl[j]][0], location_table[nl[j]][1])
            distance_table[i][j] = distance_table[j][i] = distance
            if distance > L:
                L = distance
        # print("L", L)
        # print("distance_table", distance_table)
        # print(distance_table)
        alpah, bata = 0.9, 10
        # print(iter_indexs)
        for i, j in iter_indexs:
            # exit(1)
            tmp = get_rand(0, 1)
            # print("tmp", tmp, distance_table[i][j] / (bata * L), alpah * math.exp(-distance_table[i][j] / (bata * L)))
            if tmp < alpah * math.exp(-distance_table[i][j] / (bata * L)):
                # print("i, j", i, j)
                # print("i, j", i, j, nl[i].name, nl[j].name)
                node_list = [nl[i], nl[j]]
                # print("node_list", node_list)
                link = QuantumChannel(name=f"l{i+1},{j+1}", node_list=node_list, bandwidth=4)
                nl[i].add_qchannel(link)
                nl[j].add_qchannel(link)
                ll.append(link)
        # print("Topology is built", ll)
        return nl, ll
