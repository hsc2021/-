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
from transport.center_pu import SlaveTransportApp, ControllerTransportApp
from transport.center_ps import PSControllerTransportApp
from topo import XTopo

req_count_1 = 0
req_count_2 = 0
nodes_number = 8
init_fidelity = 0.99
send_rate = 1000
memory_cap = 100
memory_decoherent_time = 0.5
delay = 0.05
delay_sed = 0.02
decoherence_rate = 0.2

topo = XTopo(nodes_number=nodes_number,
             memory_args={"capacity": memory_cap, "decoherence_rate": memory_decoherent_time},
             qchannel_args={"delay": NormalDelayModel(delay, delay_sed), "drop_rate": 0.9, "decoherence_rate": decoherence_rate},
             cchannel_args={"delay": NormalDelayModel(delay, delay_sed)})

net = QuantumNetwork(name="network", topo=topo, route=DijkstraRouteAlgorithm(), classic_topo=ClassicTopology.Follow)
net.build_route()

classic_route = DijkstraRouteAlgorithm()
classic_route.build(net.nodes, net.cchannels)

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

    src_app = LinkEPRSendApp(dest, memory_port=src_memory_idx,
                             send_rate=send_rate, init_fidelity=init_fidelity,
                             ack_timeout=6*delay)

    dest_app = LinkEPRRecvApp(src, memory_port=dest_memory_idx)

    src.add_apps(src_app)
    dest.add_apps(dest_app)

n1 = net.get_node("n1")
n3 = net.get_node("n3")
n5 = net.get_node("n5")
n2 = net.get_node("n2")
n4 = net.get_node("n4")
n6 = net.get_node("n6")
n7 = net.get_node("n7")
for n in net.nodes:
    n.add_apps(SlaveTransportApp(net=net, controller=n3))
controllerApp = PSControllerTransportApp(net=net)
# controllerApp = ControllerTransportApp(net=net)
n3.add_apps(controllerApp)
s = Simulator(0, 12, 100000)
for n in net.nodes:
    n.install(s)
n3.apps[-1].init_session(n1, n5, name="req1", end_time=10)
n3.apps[-1].init_session(n2, n7, name="req2", end_time=10)
n3.apps[-1].init_session(n4, n6, name="req3", end_time=10)
n3.apps[-1].init_session(n4, n7, name="req4", end_time=10)
s.run()

req1 = controllerApp.success_dict["req1"]
req2 = controllerApp.success_dict["req2"]
req3 = controllerApp.success_dict["req3"]
req4 = controllerApp.success_dict["req4"]
# return {"req3": len(req3), "req4": len(req4)}
reqs = [req1, req2, req3, req4]
fair = (sum([len(r) for r in reqs])/len(reqs))**2 / (sum([len(r)**2 for r in reqs])/len(reqs))

return {
    "total": len(req1+req2+req3+req4),
    "fair": fair,
    "req1": len(req1),
    "req2": len(req2),
    "req3": len(req3),
    "req4": len(req4),
    "fid1": np.mean(req1) if len(req1) >= 1 else 0,
    "fid2": np.mean(req2) if len(req2) >= 1 else 0,
    "fid3": np.mean(req3) if len(req3) >= 1 else 0,
    "fid4": np.mean(req4) if len(req3) >= 1 else 0}
