import numpy as np
import logging
import qns.utils.log as log
from typing import List, Optional
from qns.network import QuantumNetwork, Request
from qns.network.topology.topo import ClassicTopology
from qns.network.route import DijkstraRouteAlgorithm
from qns.network.protocol import ClassicPacketForwardApp
from qns.simulator.simulator import Simulator
from qns.utils.rnd import set_seed
from qns.utils.multiprocess import MPSimulations
from qns.models.delay import NormalDelayModel
from link_protocol import LinkEPRRecvApp, LinkEPRSendApp
from transport.transport_protocol import FIFOTransportApp
from transport.center_ps import PSControllerTransportApp
from transport.center_pu import ControllerTransportApp, SlaveTransportApp
from topo import WaxmanTopo


req_count_1 = 0
req_count_2 = 0


def setup_exp(TransportApp="FIFO", logger_level=logging.INFO, topo_seed: Optional[int] = 0, request_seed: Optional[int] = 0, game_seed: Optional[int] = 1):

    log.logger.setLevel(logger_level)

    # topology parameters
    nodes_number = 50
    sessions_num = 10
    size = 100_000
    # 20-0.8-0.3 998 45
    # 50-0.5-0.1 340 414
    alpha = 0.5
    beta = 0.1

    start_time = 0
    end_time = 10
    simulation_end_time = 11

    # link layer parameters
    init_fidelity = 0.99
    send_rate = 1000
    drop_rate = 0.9
    delay = 0.01
    delay_sed = 0.004
    decoherence_rate = 1

    # memory parameters
    memory_cap = 100
    memory_decoherent_time = 1

    # build topology
    if topo_seed is not None:
        set_seed(topo_seed)
    topo = WaxmanTopo(nodes_number=nodes_number, alpha=alpha, beta=beta, size=size,
                      memory_args={"capacity": memory_cap, "decoherence_rate": memory_decoherent_time},
                      qchannel_args={"delay": NormalDelayModel(delay, delay_sed), "drop_rate": 0.9, "decoherence_rate": decoherence_rate},
                      cchannel_args={"delay": NormalDelayModel(delay, delay_sed)})

    net = QuantumNetwork(name="network", topo=topo, route=DijkstraRouteAlgorithm(), classic_topo=ClassicTopology.Follow)

    classic_route = DijkstraRouteAlgorithm()
    classic_route.build(net.nodes, net.cchannels)
    net.build_route()

    # add classic packet forward app (enable forward classic packet ability)
    for n in net.nodes:
        n.add_apps(ClassicPacketForwardApp(route=classic_route))

    for ql in net.qchannels:
        src = ql.node_list[0]
        dest = ql.node_list[1]

        src_memory_idx = -1
        dest_memory_idx = -1
        for idx, m in enumerate(src.memories):
            if m.name == f"{src.name}-{dest.name}":
                src_memory_idx = idx
                break

        for idx, m in enumerate(dest.memories):
            if m.name == f"{dest.name}-{src.name}":
                dest_memory_idx = idx
                break
        assert (src_memory_idx >= 0)
        assert (dest_memory_idx >= 0)

        src_app = LinkEPRSendApp(dest, memory_port=src_memory_idx,
                                 send_rate=send_rate, init_fidelity=init_fidelity,
                                 ack_timeout=6*delay)

        dest_app = LinkEPRRecvApp(src, memory_port=dest_memory_idx)

        src.add_apps(src_app)
        dest.add_apps(dest_app)

    # add transport layer app and select controller

    n1 = net.get_node("n1")
    controller = n1
    controller_app = None

    if TransportApp == "FIFO":
        for n in net.nodes:
            n.add_apps(FIFOTransportApp(node=n, net=net))
    elif TransportApp == "PS":
        for n in net.nodes:
            n.add_apps(SlaveTransportApp(net=net, controller=controller))
        controller_app = PSControllerTransportApp(net=net)
        controller.add_apps(controller_app)
    elif TransportApp == "PU":
        for n in net.nodes:
            n.add_apps(SlaveTransportApp(net=net, controller=controller))
        controller_app = ControllerTransportApp(net=net)
        controller.add_apps(controller_app)

    # initial the simulator
    s = Simulator(0, simulation_end_time, 100000)
    for n in net.nodes:
        n.install(s)

    # initial requests
    if request_seed is not None:
        set_seed(request_seed)

    net.random_requests(sessions_num, allow_overlay=True)
    raw_requests: List[Request] = net.requests

    requests = []
    path_index_counter = {}
    requests_path_len = []

    for req in raw_requests:
        src = req.src
        dest = req.dest
        path_idx = 0

        if f"{src.name}-{dest.name}" in path_index_counter:
            path_idx = path_index_counter[f"{src.name}-{dest.name}"]
            path_index_counter[f"{src.name}-{dest.name}"] += 1
        else:
            path_index_counter[f"{src.name}-{dest.name}"] = 0
        requests.append((src, dest, path_idx))
    connected_requests = []  # the src and dest are connected
    canceled_requests = []  # the src and dest are not connected

    for idx, req in enumerate(requests):
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        name = f"req-{idx}"

        if src == dest:
            # src should not be dest
            canceled_requests.append(req)
            continue

        if path_idx >= 1:
            # DijkstraRouteAlgorithm can not handle multiple routing
            canceled_requests.append(req)
            continue

        if len(net.query_route(src, dest)) <= path_idx:
            canceled_requests.append(req)
            continue

        if controller_app is not None:
            if src != controller and len(net.query_route(src, controller)) <= 0:
                canceled_requests.append(req)
                continue
        connected_requests.append(req)

        if controller_app is None:
            src.apps[-1].init_session(dest=dest, name=name, order=path_idx, start_time=start_time, end_time=end_time)
        else:
            controller_app.init_session(src=src, dest=dest, order=path_idx, name=name, start_time=start_time, end_time=end_time)
        requests_path_len.append(len(net.query_route(src, dest)[path_idx][2]))

    # find good topo seed
    # log.info(f"[simulator] generate request for topo seed {topo_seed}: {requests}")
    # log.info(f"[simulator] connected: {len(connected_requests)} average length: {sum(requests_path_len)/sessions_num}")
    # # log.info(f"[simulator] network topology: {net.nodes} {net.qchannels}")
    # for req in connected_requests:
    #     src = req[0]
    #     dest = req[1]
    #     path_idx = req[2]
    #     idx = requests.index(req)
    #     name = f"req-{idx}"
    #     log.info(f"[session {name}] {src.name}-> {dest.name}: {net.query_route(src, dest)[path_idx]}")
    # return

    # set monitor to get current state

    # start simulation
    if game_seed is not None:
        set_seed(game_seed)
    log.install(s)
    log.info("[simulator] simulation started")
    s.run()

    # get result
    results = {}

    for req in connected_requests:
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        idx = requests.index(req)
        name = f"req-{idx}"

        if controller_app is None:
            for s in src.apps[-1].src_sessions:
                if s.id == name:
                    results[name] = s.success_list
                    break
        else:
            results[name] = controller_app.success_dict[name]

    for req in canceled_requests:
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        idx = requests.index(req)
        name = f"req-{idx}"
        results[name] = []

    total = sum([len(req) for req in results.values()]) / end_time
    try:
        fair = sum([len(r) for r in results.values()])**2 / (sum([len(r)**2 for r in results.values()]) * sessions_num)
    except ZeroDivisionError as e:
        print(f"[warning] no requests get in {TransportApp} ({topo_seed}-{game_seed}): {results}")
        fair = 0
    average = total / sessions_num

    return_msg = {
        "total": total,
        "fair": fair,
        "avg": average,
        "len": sum(requests_path_len)/sessions_num
    }

    for k, v in results.items():
        return_msg[k] = len(v) / end_time
    for k, v in results.items():
        return_msg[f"fidelity_{k}"] = np.mean(v) if len(v) >= 1 else 0

    log.info(f"[simulator] network topology: {net.nodes} {net.qchannels}")
    for req in connected_requests:
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        idx = requests.index(req)
        name = f"req-{idx}"
        log.info(f"[session {name}] {src.name}-> {dest.name}: {net.query_route(src, dest)[path_idx]}")
    print(return_msg)
    return return_msg


class SIMU(MPSimulations):
    def run(self, setting):
        print(f"start simulation {setting}")
        topo_seed = 414
        game_seed = setting["_repeat"]
        ret = setup_exp(TransportApp=setting["transport_app"], logger_level=logging.ERROR, topo_seed=topo_seed, game_seed=game_seed)
        print(f"finish simulation {setting}: {ret}")
        return ret


if __name__ == "__main__":

    ss = SIMU(settings={
        # "transport_app": ["EDTCP-1"]
        "transport_app": ["FIFO", "FAIR", "QTCP3", "EDTCP-1", "EDTCP-1.5", "PS", "PU"],
        # "transport_app": ["FIFO", "FAIR", "QTCP", "EDTCP-1", "PS", "PU"],

    },  aggregate=True, iter_count=50)

    ss.start()

    print(ss.get_data())
    ss.get_data().to_csv("result/random_50node_memory100_rate1000_delay_10_seed414_50times_dephase1_round_3.csv")
    ss.get_raw_data().to_csv("result/random_50node_memory100_rate1000_delay_10_seed414_50times_dephase1_round_3_raw.csv")

    # print(setup_exp(TransportApp="FIFO", logger_level= logging.INFO))
    # print(setup_exp(TransportApp="EDTCP-1", logger_level=logging.INFO, topo_seed=414, game_seed=0))

'''
作者提出了三种调度算法,以优化量子网络中纠缠生成的效率和资源利用。这三种调度方案分别是PU(优先调度)、PS(公平调度)和PF(公平流调度)。
1. PU(优先调度)
思路：
PU调度方案的核心思想是优先考虑吞吐量的最大化。该方案通过优先分配网络中可用的边缘容量，以实现尽可能多的纠缠生成。 
在调度过程中,PU会选择那些能够提供最大流量的路径,从而确保在每个时间窗口内，尽可能多的连接请求得到满足.
优点
该方案能够显著提高网络的吞吐量，充分利用网络的边缘容量。 
在高负载情况下,PU调度能够有效地处理大量的连接请求,确保网络的高效运行。 
缺点：
由于其优先考虑吞吐量,PU调度可能导致资源分配的不均衡,某些路径可能会承载过多的流量,而其他路径则可能处于闲置状态,从而造成公平性问题.
2. PS(公平调度)
思路：
PS调度方案的设计目标是实现资源的公平分配。该方案通过均匀分配网络中的流量，确保每个连接请求都能获得相对公平的服务。 
在调度过程中，PS会尽量避免某些路径的过载，确保所有请求都能得到满足，而不会让某些请求因资源不足而被延迟或拒绝。 
优点：
PS调度能够有效地降低网络中边缘容量利用的方差，使得流量分布更加均匀，从而提高了网络的稳定性。 
该方案在处理多个连接请求时，能够确保每个请求都能获得合理的服务，避免了因资源分配不均而导致的性能下降。 
缺点：
尽管PS调度在公平性上表现良好，但其吞吐量通常低于PU调度，可能导致在高负载情况下，整体网络性能的下降【T5】。 
3. PF(公平流调度)
思路：
PF调度方案结合了优先调度和公平调度的特点，旨在实现对请求和路径的完全公平性。 
在调度过程中，PF会根据每个请求的优先级和路径的可用性，进行动态调整，以确保所有请求都能得到公平的处理。 
优点：
PF调度在公平性和吞吐量之间取得了良好的平衡，能够在保证公平的同时，尽量提高网络的整体性能。 
该方案能够有效地处理不同优先级的请求，确保高优先级请求在资源有限的情况下仍能得到满足。 
缺点：
PF调度的吞吐量通常低于PU调度，且在某些情况下，可能会导致整体网络效率的下降，尤其是在高负载情况下【T5】。 



三种调度策略的实现方式各有不同，具体如下：
比例分享调度算法 (Proportional Share, PU)：
实现方式：该算法基于边缘特定的容量分配。每条边的容量 Cij​ 会根据使用该边的所有路径的需求进行比例分配。具体来说，算法只依赖于每
条边的局部信息，而不需要全局网络状态。这种方法允许在每个时间窗口内动态调整流量，以最大化网络的吞吐量。 
步骤： 收集每条边的当前流量和容量信息。 
根据每条边的使用情况和请求的流量需求，计算每条边的可用容量。 
按照比例分配可用容量给所有使用该边的路径。

均匀调度算法 (Fair Share, PF)：
实现方式：该算法实现了最大-最小公平性，确保所有路径在流量分配上得到均等对待。流量从零开始，逐步增加，直到达到边缘的饱和状态。每
次流量增加时，所有路径都以相同的增量进行流量分配，直到没有更多的流量可以分配。 
步骤： 初始化所有路径的流量为零。 
在每个时间窗口内，均匀地增加所有路径的流量，直到某些边缘达到饱和。 
一旦某条边缘饱和，使用该边缘的路径将停止增加流量。 

渐进填充调度算法 (Progressive Filling, PS)：
实现方式：该算法结合了局部信息和逐步填充的策略，确保在流量分配时保持一定的公平性。与PF类似，PS算法也从零开始增加流量，但它在每个
时间窗口内会根据当前的流量状态进行动态调整，以确保流量的均匀分配。 
步骤： 初始化所有路径的流量为零。 
在每个时间窗口内，逐步增加所有路径的流量，确保每条边的流量增加是均匀的。 
监测每条边的饱和状态，并在达到饱和时停止对该边的流量增加。 
'''