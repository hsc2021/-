from qns.models.qubit import QubitFactory, H, CNOT
from qns.models.qubit.const import QUBIT_STATE_0
from qns.models.qubit.decoherence import DephaseStorageErrorModel, DepolarOperateErrorModel
from qns.entity.node import Application, QNode
from qns.entity.memory import QuantumMemory
from qns.entity.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.monitor import Monitor
from qns.entity.qchannel import QuantumChannel, RecvQubitPacket
from qns.simulator.simulator import Simulator, Time
from qns.simulator.event import func_to_event, Event

import qns.utils.log as log

import logging
try:
    import dpnp as np
except ImportError:
    import numpy as np
from scipy.linalg import sqrtm

cut_off_fidelity = -1
decoherent_time = 1

name_idx = 1

Qubit = QubitFactory(store_error_model=DephaseStorageErrorModel, operate_error_model=DepolarOperateErrorModel, operate_decoherence_rate=0.01)


def calculate_fidelity(rho_1: np.ndarray):
    rho_2 = np.array([
        [0.5, 0, 0, 0.5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0.5, 0, 0, 0.5]
    ])
    rho_1_2 = sqrtm(rho_1)
    x = np.dot(rho_1_2, np.dot(rho_2, rho_1_2))
    x = sqrtm(x)
    x = np.trace(x)
    return float(np.real(x))


class LinkEPRRecvApp(Application):
    def __init__(self, src: QNode,  memory_port: int):
        super().__init__()
        self.src = src
        self.memory_port = memory_port
        self.temp_list = []

    def install(self, node, simulator: Simulator):
        super().install(node, simulator)
        self.memory: QuantumMemory = self._node.memories[self.memory_port]
        self.qchannel: QuantumChannel = self._node.get_qchannel(self.src)
        self.cchannel: ClassicChannel = self._node.get_cchannel(self.src)
        self.add_handler(self.handle_recv_qubit, [RecvQubitPacket])

    def handle_recv_qubit(self, node, event: Event):

        if event.qchannel != self.qchannel:
            return False

        # remove_eprs = []
        # tc = self._simulator.tc
        # for idx, e in enumerate(self.memory._storage):
        #     if e is not None:
        #         store_time = self.memory._store_time[idx]
        #         diff_time = tc.sec - store_time.sec
        #         w = e.w * np.exp(-decoherent_time * diff_time)
        #         evaluate_fidelity = (w * 3 + 1) / 4 
        #         if evaluate_fidelity < cut_off_fidelity:
        #             remove_eprs.append(e)
        # for e in remove_eprs:
        #     self.memory.read(e)
        #     #log.info(f"[link] node {self._node}-{self.memory.name} cut-off epr {e.name}")

        recv_epr = event.qubit
        self.temp_list.append(recv_epr)
        # ret = self.memory.write(recv_epr)
        if True:
            # log.info(f"[link] node {self._node.name}-{self.memory.name} write epr {recv_epr.name} success")
            packet = ClassicPacket(msg={"cmd": "link", "name": recv_epr.name}, src=self._node, dest=self.src)
            self.cchannel.send(packet, next_hop=self.src)
            # log.info(f"[link] node {self._node.name} send ack {recv_epr.name}")
        else:
            pass
            # log.info(f"[link] node {self._node.name}-{self.memory.name} write epr {recv_epr.name} failed")
        return True

    def __repr__(self) -> str:
        return f"<LinkEPRRecvAPP src: {self.src} dest: {self._node}>"


class LinkEPRSendApp(Application):

    def __init__(self, dest: QNode,
                 memory_port: int, send_rate: int = 10,
                 init_fidelity: float = 0.99, ack_timeout: float = 1):
        super().__init__()
        self.dest = dest
        self.memory_port = memory_port
        self.send_rate = send_rate
        self.init_fidelity = init_fidelity
        self.ack_timeout = ack_timeout

        self.temp_list = {}
        self.temp_storage = []

    def install(self, node, simulator: Simulator):
        super().install(node, simulator)
        self.memory: QuantumMemory = self._node.memories[self.memory_port]
        self.qchannel: QuantumChannel = self.get_node().get_qchannel(self.dest)
        self.cchannel: ClassicChannel = self._node.get_cchannel(self.dest)

        self.add_handler(self.handle_ack, [RecvClassicPacket], [self.cchannel])

        t = self._simulator.ts
        event = func_to_event(t=t, fn=self.gen_epr, by=self._node)
        self._simulator.add_event(event)

    def gen_epr(self):

        next_t = self._simulator.tc + Time(sec=1 / self.send_rate)
        event = func_to_event(t=next_t, fn=self.gen_epr)
        self._simulator.add_event(event)

        self.clear_state()
        # if len(self.temp_list) >= self.memory.capacity - self.memory.count:
        #     # print(234, self.temp_list, self.memory.capacity, self.memory.count)
        #     return

        global name_idx
        # e1 = WernerStateEntanglement(fidelity=self.init_fidelity,
        #                              name=str(name_idx))
        q1 = Qubit(state=QUBIT_STATE_0, name= f"{name_idx}")
        q2 = Qubit(state=QUBIT_STATE_0, name= f"{name_idx}")
        H(q1)
        CNOT(q1, q2)
        name_idx += 1


        # tc = self._simulator.tc
        # remove_eprs = []
        # for idx, e in enumerate(self.memory._storage):
        #     if e is not None:
        #         store_time = self.memory._store_time[idx]
        #         diff_time = tc.sec - store_time.sec
        #         w = e.w * np.exp(-decoherent_time * diff_time)
        #         evaluate_fidelity = (w * 3 + 1) / 4 
        #         if evaluate_fidelity < cut_off_fidelity:
        #             remove_eprs.append(e)
        # for e in remove_eprs:
        #     #log.info(f"node {self._node}-{self.memory.name} cut-off epr {e.name}")
        #     self.memory.read(e)

        # ret = self.memory.write(e1)
        self.temp_list[q1.name] = (q1, self._simulator.tc.sec)
        #log.info(f"[link] node {self._node} send epr {q2.name}")
        self.qchannel.send(q2, self.dest)

    def handle_ack(self, node, event: RecvClassicPacket):

        msg = event.packet.get()
        if msg.get("cmd") != "link":
            return False

        if event.cchannel != self.cchannel:
            return False

        msg = msg.get("name")

        #log.info(f"[link] node {self._node} recv ack {msg}")
        try:
            epr, _ = self.temp_list.pop(msg, (None,None))
        except KeyError:
            print(12341234)
            return False

        recvApps = self.dest.get_apps(LinkEPRRecvApp)
        recvApp = None
        for app in recvApps:
            if app.qchannel == self.qchannel:
                recvApp = app

        epr2 = None
        for e in recvApp.temp_list:
            if e.name == msg:
                epr2 = e
        
        assert(epr2 is not None)

        self.clear_state()

        ret1 = self.memory.write(epr)
        ret2 = recvApp.memory.write(epr2)
        if not ret1:
            #log.info(f"[link] generate epr {msg} failed: {self._node.name} memory full")
            recvApp.memory.read(epr2)
        if not ret2:
            #log.info(f"[link] generate epr {msg} failed: {self.dest.name} memory full")
            self.memory.read(epr)
        return True

    def clear_state(self):
        tc = self._simulator.tc.sec

        remove_list = []
        for k, (_, v) in self.temp_list.items():
            if tc >= v + self.ack_timeout:
                remove_list.append(k)

        for e in remove_list:
            #log.info(f"[link] node {self._node} clear drop qubit {e}")
            self.temp_list.pop(e, None)

        # remove_eprs = [epr for epr in self.memory._storage if epr is not None and calculate_fidelity(epr.state.rho) < 0.8]
        # for epr in remove_eprs:
        #     self.memory.read(epr)

        # remove_eprs = []
        # for idx, e in enumerate(self.memory._storage):
        #     if e is not None:
        #         store_time = self.memory._store_time[idx]
        #         diff_time = tc.sec - store_time.sec
        #         w = e.w * np.exp(-decoherent_time * diff_time)
        #         evaluate_fidelity = (w * 3 + 1) / 4 
        #         if evaluate_fidelity < cut_off_fidelity:
        #             remove_eprs.append(e)
        # for e in remove_eprs:
        #     #log.info(f"[link] node {self._node}-{self.memory.name} cut-off epr {e.name}")
        #     self.memory.read(e)

    def __repr__(self) -> str:
        return f"<LinkEPRSendAPP src: {self._node} dest: {self.dest}>"


if __name__ == "__main__":
    
    log.logger.setLevel(logging.INFO)
    n1 = QNode(name="n1")
    n2 = QNode(name="n2")

    m1 = QuantumMemory(name="m1", capacity=10000000, decoherence_rate=decoherent_time)
    m2 = QuantumMemory(name="m2", capacity=10000000, decoherence_rate=decoherent_time)

    c1 = QuantumChannel(name="c1", delay = 0.05, drop_rate=0.95)
    cc = ClassicChannel(name="cc", delay = 0.05)

    n1.add_memory(m1)
    n2.add_memory(m2)
    n1.add_qchannel(c1)
    n2.add_qchannel(c1)
    n1.add_cchannel(cc)
    n2.add_cchannel(cc)

    send_app = LinkEPRSendApp(dest=n2, memory_port=0, send_rate=1000, ack_timeout=0.2)
    recv_app = LinkEPRRecvApp(src=n1, memory_port=0)

    n1.add_apps(send_app)
    n2.add_apps(recv_app)

    m = Monitor(name="monitor-1")

    def epr_num_1(s, n, e):
        count = 0
        for e in m1._storage:
            if e is not None:
                count += 1
        print(s.tc, count,sep="\t", end="\t")
        return count

    def epr_num_2(s, n, e):
        count = 0
        for e in m2._storage:
            if e is not None:
                count += 1
        print(count)
        return count

    m.add_attribution("epr_num_1", calculate_func= epr_num_1)
    m.add_attribution("epr_num_2", calculate_func= epr_num_2)

    m.at_period(1)

    s = Simulator(0, 15, 100000)

    n1.install(s)
    n2.install(s)
    log.install(s)
    m.install(s)
    s.run()
    data = m.get_date()
    data.to_csv("link_stats_1000hz_0.95_65cap_5000rnd.csv",index=False)




