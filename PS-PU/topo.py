from typing import Tuple, List, Dict, Optional
from qns.network.topology import Topology
from qns.entity.node import QNode, Application
from qns.entity.qchannel import QuantumChannel
from qns.entity.memory import QuantumMemory
from qns.utils.rnd import get_rand

try:
    import dpnp as np
except ImportError:
    import numpy as np
import itertools


class WaxmanTopo(Topology):
    def __init__(self, nodes_number: int, size: float,
                 alpha: float, beta: float,
                 nodes_apps: List[Application] = [],
                 qchannel_args: Dict = {}, cchannel_args: Dict = {},
                 memory_args: Optional[List[Dict]] = {}):

        super().__init__(nodes_number, nodes_apps, qchannel_args,
                         cchannel_args, memory_args)
        self.size = size
        self.alpha = alpha
        self.beta = beta

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []

        location_table: Dict[QNode, Tuple[float, float]] = {}
        distance_table: Dict[Tuple[QNode, QNode], float] = {}

        for i in range(self.nodes_number):
            n = QNode(f"n{i+1}")
            nl.append(n)
            x = get_rand() * self.size
            y = get_rand() * self.size
            location_table[n] = (x, y)

        L = 0
        cb = list(itertools.combinations(nl, 2))
        for n1, n2 in cb:
            tmp_l = np.sqrt((location_table[n1][0] - location_table[n2][0]) ** 2
                            + (location_table[n1][1] - location_table[n2][1]) ** 2)
            distance_table[(n1, n2)] = tmp_l
            if tmp_l > L:
                L = tmp_l

        for n1, n2 in cb:
            if n1 == n2:
                continue
            d = distance_table[(n1, n2)]
            p = self.alpha * np.exp(-d / (self.beta * L))
            if get_rand() < p:

                ql = QuantumChannel(name=f"{n1.name}-{n2.name}", length=d, **self.qchannel_args)
                m1 = QuantumMemory(name=f"{n1.name}-{n2.name}", node=n1, **self.memory_args)
                m2 = QuantumMemory(name=f"{n2.name}-{n1.name}", node=n2, **self.memory_args)

                n1.add_memory(m1)
                n2.add_memory(m2)

                ll.append(ql)
                n1.add_qchannel(ql)
                n2.add_qchannel(ql)
        return nl, ll


class XTopo(Topology):
    def __init__(self, nodes_number: int,
                 nodes_apps: List[Application] = [],
                 qchannel_args: Dict = {}, cchannel_args: Dict = {},
                 memory_args: Optional[List[Dict]] = {}):

        self.nodes_number = 7
        super().__init__(self.nodes_number, nodes_apps, qchannel_args,
                         cchannel_args, memory_args)

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []

        for i in range(self.nodes_number):
            n = QNode(f"n{i+1}")
            nl.append(n)

        cb = [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)]

        for (n1_idx, n2_idx) in cb:
            n1 = nl[n1_idx]
            n2 = nl[n2_idx]
            ql = QuantumChannel(name=f"{n1.name}-{n2.name}", length=10, **self.qchannel_args)
            m1 = QuantumMemory(name=f"{n1.name}-{n2.name}", node=n1, **self.memory_args)
            m2 = QuantumMemory(name=f"{n2.name}-{n1.name}", node=n2, **self.memory_args)

            n1.add_memory(m1)
            n2.add_memory(m2)

            ll.append(ql)
            n1.add_qchannel(ql)
            n2.add_qchannel(ql)
        return nl, ll


class LineTopo(Topology):
    def __init__(self, nodes_number: int,
                 nodes_apps: List[Application] = [],
                 qchannel_args: Dict = {}, cchannel_args: Dict = {},
                 memory_args: Optional[List[Dict]] = {}):

        self.nodes_number = nodes_number
        super().__init__(self.nodes_number, nodes_apps, qchannel_args,
                         cchannel_args, memory_args)

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []

        for i in range(self.nodes_number):
            n = QNode(f"n{i+1}")
            nl.append(n)

        for n1_idx in range(self.nodes_number-1):
            n2_idx = n1_idx + 1
            n1 = nl[n1_idx]
            n2 = nl[n2_idx]
            ql = QuantumChannel(name=f"{n1.name}-{n2.name}", length=10, **self.qchannel_args)
            m1 = QuantumMemory(name=f"{n1.name}-{n2.name}", node=n1, **self.memory_args)
            m2 = QuantumMemory(name=f"{n2.name}-{n1.name}", node=n2, **self.memory_args)

            n1.add_memory(m1)
            n2.add_memory(m2)

            ll.append(ql)
            n1.add_qchannel(ql)
            n2.add_qchannel(ql)
        return nl, ll
