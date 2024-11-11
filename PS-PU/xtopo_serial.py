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
from qns.entity.monitor import Monitor
from qns.models.delay import NormalDelayModel
from link_protocol import LinkEPRRecvApp, LinkEPRSendApp
from transport.transport_protocol import FIFOTransportApp
from transport.center_ps import PSControllerTransportApp
from transport.center_pu import ControllerTransportApp, SlaveTransportApp
from topo import XTopo

req_count_1 = 0
req_count_2 = 0

G = {}


def setup_exp(TransportApp="FIFO", logger_level=logging.INFO):

    # topology parameters
    nodes_number = 8

    start_time = 0
    end_time = 30
    simulation_end_time = 31

    # link layer parameters
    init_fidelity = 0.99
    send_rate = 1000
    delay = 0.05
    delay_sed = 0.02
    decoherence_rate = 2

    # memory parameters
    memory_cap = 100
    memory_decoherent_time = 2

    # build topology
    topo = XTopo(nodes_number=nodes_number,
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
                                 ack_timeout=6 * delay)

        dest_app = LinkEPRRecvApp(src, memory_port=dest_memory_idx)

        src.add_apps(src_app)
        dest.add_apps(dest_app)

    # add transport layer app

    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    n3 = net.get_node("n3")
    n4 = net.get_node("n4")
    n5 = net.get_node("n5")
    n6 = net.get_node("n6")
    n7 = net.get_node("n7")
    controller = n3
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

    requests = [(n1, n5, 0, None, None), (n2, n7, 0, 10, None), (n4, n6, 0, None, None), (n4, n7, 0, None, None)]
    # requests = [(n1, n5, 0, None, None)]
    connected_requests = []  # the src and dest are connected
    canceled_requests = []  # the src and dest are not connected

    for idx, req in enumerate(requests):
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        name = "req-{}".format(idx)

        s_st = req[3] if req[3] is not None else start_time
        s_et = req[4] if req[4] is not None else end_time

        if len(net.query_route(src, dest)) <= path_idx:
            canceled_requests.append(req)
            continue
        connected_requests.append(req)

        if controller_app is None:
            src.apps[-1].init_session(dest=dest, name=name, order=path_idx, start_time=s_st, end_time=s_et)
        else:
            controller_app.init_session(src=src, dest=dest, order=path_idx, name=name, start_time=s_st, end_time=s_et)

    # setup monitor

    m = Monitor("monitor-0", network=net)

    def monitor_count(src, req_id: str, controller_app=None):
        def stat(s, n, e):
            global G
            if controller_app is None:
                current_session = None
                for s in src.apps[-1].sessions:
                    if s.id == req_id:
                        current_session = s
                assert (current_session is not None)
                new_count = len(current_session.success_list)
            else:
                new_count = len(controller_app.success_dict[name])
            last_count = G.get(f"last-{req_id}", 0)
            G[f"last-{req_id}"] = new_count
            return new_count-last_count
        return stat

    def monitor_fid(src, req_id: str, controller_app=None):
        def fid(s, n, e):
            global G

            last_count = G.get(f"lastfid-{req_id}", 0)
            if controller_app is None:
                current_session = None
                for s in src.apps[-1].sessions:
                    if s.id == req_id:
                        current_session = s
                assert (current_session is not None)
                new_count = len(current_session.success_list)
                new_eprs = current_session.success_list[last_count:]
            else:
                new_count = len(controller_app.success_dict[name])
                new_eprs = controller_app.success_dict[name][last_count:]
            G[f"lastfid-{req_id}"] = new_count
            return np.mean(new_eprs) if len(new_eprs) > 0 else 0
        return fid

    def monitor_win(src, req_id: str, controller_app=None):
        def win(s, n, e):
            current_session = None
            for s in src.apps[-1].sessions:
                if s.id == req_id:
                    current_session = s
            assert (current_session is not None)
            return current_session.last_win
        return win

    def monitor_gen(src, req_id: str, controller_app=None):
        def win(s, n, e):
            current_session = None
            for s in src.apps[-1].sessions:
                if s.id == req_id:
                    current_session = s
            assert (current_session is not None)
            return current_session.last_gen
        return win

    for idx, req in enumerate(requests):
        src = req[0]
        dest = req[1]
        path_idx = req[2]
        name = "req-{}".format(idx)
        m.add_attribution(f"edr-{name}", monitor_count(src=src, req_id=name, controller_app=controller_app))
        m.add_attribution(f"fid-{name}", monitor_fid(src=src, req_id=name, controller_app=controller_app))
        if TransportApp[:2] in ["ED", "QT"]:
            m.add_attribution(f"win-{name}", monitor_win(src=src, req_id=name, controller_app=controller_app))
            m.add_attribution(f"gen-{name}", monitor_gen(src=src, req_id=name, controller_app=controller_app))
    m.at_period(0.01)
    m.install(s)

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

    return_msg = {
        "total": total,
    }

    for k, v in results.items():
        return_msg[k] = len(v) / end_time

    for k, v in results.items():
        return_msg[f"f_{k}"] = np.mean(v) if len(v) >= 1 else 0

    return return_msg, m.get_date()


if __name__ == "__main__":
    # setup_exp(TransportApp="QTCP3", logger_level= logging.INFO)
    for app in ["EDTCP-1", "QTCP3"]:
        ret, stat = setup_exp(TransportApp=app, logger_level=logging.ERROR)
        for req in ["req-0", "req-1", "req-2", "req-3"]:
            stat_tmp = stat[(stat[f"edr-{req}"] > 0)][["time", f"win-{req}"]]
            stat_tmp.to_csv(f"result/serial/result_serial-delay50-{app}-{req}.csv")
        print(ret)

    # ret, stat = setup_exp(TransportApp="QTCP3", logger_level= logging.INFO)
    # print(ret, stat)
    # for app in ["EDTCP-1", "QTCP3"]:
    #     ret, stat = setup_exp(TransportApp=app, logger_level= logging.ERROR)
    #     stat = stat[ (stat["edr-req-0"]>0)][["time","edr-req-0","win-req-0"]]
    #     stat.to_csv(f"result/serial/result_serial-delay50-one-{app}.csv")
    #     print(ret)
