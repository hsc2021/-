try:
    import dpnp as np
except ImportError:
    import numpy as np

import logging
import qns.utils.log as log

from qns.network import QuantumNetwork
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
from topo import LineTopo


req_count_1 = 0
req_count_2 = 0


def setup_exp(TransportApp="FIFO", nodes_number: int = 8, logger_level=logging.INFO):

    # topology parameters
    nodes_number = nodes_number

    start_time = 0
    end_time = 10
    simulation_end_time = 11

    # link layer parameters
    init_fidelity = 0.99
    send_rate = 1000
    delay = 0.01
    delay_sed = 0.004
    decoherence_rate = 1

    # memory parameters
    memory_cap = 100
    memory_decoherent_time = 1

    # build topology
    topo = LineTopo(nodes_number=nodes_number, memory_args={"capacity": memory_cap, "decoherence_rate": memory_decoherent_time},
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

        src_app = LinkEPRSendApp(dest, memory_port=src_memory_idx, send_rate=send_rate, init_fidelity=init_fidelity, ack_timeout=6*delay)

        dest_app = LinkEPRRecvApp(src, memory_port=dest_memory_idx)

        src.add_apps(src_app)
        dest.add_apps(dest_app)

    src = net.get_node("n1")
    dest = net.get_node(f"n{nodes_number}")
    controller = net.get_node(f"n{round(nodes_number/2)}")
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

    requests = [(src, dest, 0)]
    connected_requests = []  # the src and dest are connected
    canceled_requests = []  # the src and dest are not connected

    for idx, req in enumerate(requests):
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        name = "req-{}".format(idx)

        if len(net.query_route(src, dest)) <= path_idx:
            canceled_requests.append(req)
            continue
        connected_requests.append(req)

        if controller_app is None:
            src.apps[-1].init_session(dest=dest, name=name, order=path_idx, start_time=start_time, end_time=end_time)
        else:
            controller_app.init_session(src=src, dest=dest, order=path_idx, name=name, start_time=start_time, end_time=end_time)

    # start simulation
    log.logger.setLevel(logger_level)
    log.install(s)
    s.run()

    # get result
    results = {}

    for req in connected_requests:
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        idx = requests.index(req)
        name = "req-{}".format(idx)

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
        name = "req-{}".format(idx)
        results[name] = []

    total = sum([len(req) for req in results.values()]) / end_time
    try:
        fair = sum([len(r) for r in results.values()])**2 / (sum([len(r)**2 for r in results.values()]) * len(results))
    except:
        fair = 0

    return_msg = {
        "total": total,
        "fair": fair,
    }

    for k, v in results.items():
        return_msg[k] = len(v) / end_time

    for k, v in results.items():
        return_msg[f"f_{k}"] = np.mean(v) if len(v) >= 1 else 0

    return return_msg


class SIMU(MPSimulations):
    def run(self, setting):
        print(f"start simulation {setting}")

        nodes_number = setting["nodes_number"]
        iter_count = setting["_repeat"]
        set_seed(iter_count+10)
        ret = setup_exp(TransportApp=setting["transport_app"], nodes_number=nodes_number, logger_level=logging.ERROR)
        print(f"finish simulation {setting}: {ret}")
        return ret


if __name__ == "__main__":

    ss = SIMU(settings={
        "transport_app": ["PU"],
        "nodes_number": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    },  aggregate=True, iter_count=5)

    ss.start()

    print(ss.get_data())
    ss.get_data().to_csv("result/result_line_delay50_100games_a0.7_round3_small.csv")
    ss.get_raw_data().to_csv("result/result_line_delay50_100games_a0.7_round3_small_raw.csv")
    # print(setup_exp(TransportApp="EDTCP-1", nodes_number=15, logger_level= logging.INFO))
