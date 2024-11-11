from qns.network.route import DijkstraRouteAlgorithm
from qns.network.topology.topo import ClassicTopology
from qns.simulator.simulator import Simulator
from qns.network.network import QuantumNetwork
from qns.utils.rnd import set_seed
from REPS_App import REPS_APP
import time
from REPS_Topology import REPS_Topo
from qns.network.topology.randomtopo import RandomTopology


def main():
    end_sim_time = 10
    start_sim_time = 0
    accuracy = 1000000
    size = 10000
    set_seed(int(time.time()))
    # set_seed(1)
    request_num = 6
    nodes_number = 50
    link_prob = 0.8
    node_prob = 0.9
    edge_capa = 6
    s = Simulator(start_sim_time, end_sim_time, accuracy=accuracy)
    # topo = REPS_Topo(nodes_number, size)
    topo = RandomTopology(nodes_number=nodes_number, lines_number=nodes_number/10)
    net = QuantumNetwork(name="network", topo=topo, route=DijkstraRouteAlgorithm(), classic_topo=ClassicTopology.All)

    classic_route = DijkstraRouteAlgorithm()
    classic_route.build(net.nodes, net.cchannels)
    net.build_route()

    node = net.get_node("n1")
    # print(node)
    node.add_apps(REPS_APP(net, node, request_num=request_num, link_prob=link_prob, node_prob=node_prob, edge_capa=edge_capa))
    net.install(s)
    s.run()

    result = node.apps[-1].ECs
    print("result:", result)


if __name__ == '__main__':
    main()

'''
    REPS
这篇文章面临的主要问题是量子通信中建立纠缠连接的困难,尤其是在节点之间距离较远时,成功创建纠缠链接的概率较低。此外,量子网络的吞吐量受到
限制，因为每个建立的纠缠连接只能传输一个量子比特，这导致了低吞吐量的问题。

为了解决这些问题,文章提出了一种名为REPS(冗余纠缠资源配置与选择)的新方法。REPS的核心思路是通过引入冗余外部链接来提高故障容忍能力,
从而优化量子网络的吞吐量。

REPS(冗余纠缠资源配置与选择)算法的基本思想是通过引入冗余资源来提高量子网络中纠缠连接的成功率,从而最大化网络的吞吐量。
该算法的设计考虑了量子网络的独特特性,旨在优化资源的使用和连接的建立。具体步骤如下：
冗余资源配置：
在第一步,REPS算法首先根据网络拓扑和资源情况,配置尽可能多的外部链接资源。这些链接是冗余的,目的是在某些链接创建失败的情况下仍然能够
建立纠缠连接.
全局信息利用：
在第二步,REPS利用全局知识,即所有成功创建的外部链接的信息,来选择合适的链接进行连接.这一过程确保了在建立纠缠连接时,能够充分利用已成
功创建的资源
优化选择：
REPS算法通过设计高效的选择算法,确定哪些外部链接应该被用来建立纠缠连接,以实现网络的整体优化.这些算法基于全局信息,能够在多个源-目的对
之间进行合理的资源分配.
'''
