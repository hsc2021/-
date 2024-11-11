from dataclasses import dataclass, field
from enum import Enum
from inspect import currentframe
from typing import Any, Dict, List, Optional
import uuid
try:
    import dpnp as np
except ImportError:
    import numpy as np
from scipy.linalg import sqrtm

from qns.entity.node import Application, QNode
from qns.entity.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.memory import QuantumMemory
from qns.models import QuantumModel
from qns.models.qubit import CNOT, H, X, Z, Qubit
from qns.simulator.simulator import Simulator
from qns.simulator.event import Event, Time, func_to_event
from qns.network import QuantumNetwork
import qns.utils.log as log


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


class SessionState(Enum):
    NOT_START = 0
    INIT = 1        # start to build a connection
    SYN = 2         # build a connection
    READY = 3       # connection established
    WIN_PROBE = 4   # forward backword
    SWAP = 5        # entanglement swapping
    FIN = 6         # finished


@dataclass
class Session():
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    src: QNode = None
    dest: QNode = None
    _node_idx: Optional[int] = None

    start_time: Optional[Time] = field(repr=False,
                                       default_factory=lambda: Time(time_slot=0))
    end_time: Optional[Time] = field(repr=False, default_factory=lambda: None)

    path: List[QNode] = field(default_factory=lambda: [])  # the session path
    state: SessionState = field(default=SessionState.INIT)  # the state
    # the fidelity of all successful distributed eprs
    success_list: List[Any] = field(repr=False, default_factory=lambda: [])

    # occupied eprs of this session in the upper link
    from_eprs: List[Any] = field(repr=False, default_factory=lambda: [])
    # occupied eps of the session in the downstream link
    to_eprs: List[Any] = field(repr=False, default_factory=lambda: [])
    # the number of sessions that share a same part of the path
    common_sessions_num: List[int] = field(
        repr=False, default_factory=lambda: [])

    # store the swapping result from the repeaters
    src_results: Dict[int, Any] = field(repr=False, default_factory=lambda: {})

    # the requested windows from the last transmit
    last_win: int = 0

    # the distributed eprs in the last transmit
    last_gen: int = 0

    # ec flag used in Tele-QTP
    ec: int = 0

    # SS flag used in Tele-QTP
    ss: int = 1
    
    def __hash__(self) -> int:
        return hash(self.id)

    def get_next_hop(self, node: QNode):
        """
        may raise IndexError or ValueError Exeption
        """
        idx = self.path.index(node)
        idx += 1
        return self.path[idx]

    def get_prev_hop(self, node: QNode):
        """
        may raise IndexError or ValueError Exeption
        """
        idx = self.path.index(node)
        idx -= 1
        return self.path[idx]

    def get_index(self, node: QNode):
        if self._node_idx is not None:
            return self._node_idx
        self._node_idx = self.path.index(node)
        return self._node_idx


class FIFOTransportApp(Application):

    name = "FIFO"

    def __init__(self, node: QNode, net: QuantumNetwork):
        super().__init__()

        self.node = node
        self.net = net

        self.sessions: List[Session] = []
        self.src_sessions: List[Session] = []
        self.dest_sessions: List[Session] = []

    def install(self, node, simulator: Simulator):
        super().install(node, simulator)

        self.add_handler(self.handle_syn, [RecvClassicPacket])
        self.add_handler(self.handle_ack, [RecvClassicPacket])
        self.add_handler(self.handle_fw, [RecvClassicPacket])
        self.add_handler(self.handle_bw, [RecvClassicPacket])
        self.add_handler(self.success, [RecvClassicPacket])

    def init_session(self, dest: QNode, order: int = 0, start_time: Optional[float] = 0, end_time: Optional[float] = None, name=None):
        src = self.node
        dest = dest

        start_time = self._simulator.time(sec=start_time)
        if end_time is not None:
            end_time = self._simulator.time(sec=end_time)

        path = self.net.query_route(src, dest)[order][2]
        if name is None:
            s = Session(src=src, dest=dest, path=path,
                        start_time=start_time, end_time=end_time)
        else:
            s = Session(src=src, dest=dest, path=path,
                        start_time=start_time, end_time=end_time, id=name)
        self.add_session(s)

        if s.start_time is not None:
            def s_start():
                s.state = SessionState.INIT
                self.update_Session_state(s)

            event = func_to_event(s.start_time, fn=s_start, by=self.node)
            self._simulator.add_event(event)

        if s.end_time is not None:
            def s_fin():
                s.state = SessionState.FIN
            event = func_to_event(s.end_time, fn=s_fin, by=self.node)
            self._simulator.add_event(event)

    def add_session(self, session: Session):
        self.sessions.append(session)
        if self.node == session.src:
            self.src_sessions.append(session)
        if self.node == session.dest:
            self.dest_sessions.append(session)

    def update_Session_state(self, s: Session):

        if s.end_time is not None and self._simulator.tc >= s.end_time:
            s.state = SessionState.FIN
            for n in s.path:
                for ns in n.apps[-1].sessions:
                    if ns.id == s.id:
                        ns.state = SessionState.FIN
        elif s.state == SessionState.INIT:
            self.start_syn(s)
        elif s.state == SessionState.READY:
            self.start_fw(s)

    def start_syn(self, s: Session):
        next_hop = s.get_next_hop(self.node)
        cchannel: ClassicChannel = self.node.get_cchannel(next_hop)

        packet = ClassicPacket(msg=self._parse_session(
            s), src=self.node, dest=next_hop)
        cchannel.send(packet=packet, next_hop=next_hop)
        s.state = SessionState.SYN

    def handle_syn(self, node, event: RecvClassicPacket):
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "syn":
            return False

        s = self._load_session(msg)
        if s.dest != self.node:
            next_hop = s.get_next_hop(self.node)
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            packet = ClassicPacket(msg=msg, src=self.node, dest=next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
            s.state = SessionState.SYN

            self.sessions.append(s)
        else:
            s.state = SessionState.READY
            self.sessions.append(s)
            self.dest_sessions.append(s)

            next_hop = s.get_prev_hop(self.node)

            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            msg = {"cmd": "ack", "id": s.id}
            packet = ClassicPacket(msg=msg, src=self.node, dest=next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
        return True

    def handle_ack(self, node, event: RecvClassicPacket):
        msg: Dict = event.packet.get()

        cmd = msg.get("cmd", "")
        if cmd != "ack":
            return False

        sid = msg.get("id")
        current_session = None
        for s in self.sessions:
            if s.id == sid:
                current_session = s
        assert(current_session is not None)
        if current_session.state == SessionState.FIN:
            return True

        current_session.state = SessionState.READY

        if current_session.src != self.node:
            next_hop = current_session.get_prev_hop(self.node)
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            packet = ClassicPacket(msg=msg, src=self.node, dest=next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
        else:
            self.update_Session_state(current_session)
        return True

    def start_fw(self, s: Session):
        s.state = SessionState.WIN_PROBE
        s.src_results.clear()

        next_hop = s.get_next_hop(self.node)
        cchannel: ClassicChannel = self.node.get_cchannel(next_hop)

        send_list = self._src_win_strategy(s)

        msg = {"cmd": "fw", "id": s.id, "eprs": [send_list]}

        log.info(f"[transport] {self.node.name}->{next_hop.name}: {msg}")
        packet = ClassicPacket(msg=msg, src=self.node, dest=next_hop)
        cchannel.send(packet=packet, next_hop=next_hop)

    def handle_fw(self, node, event: RecvClassicPacket):
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "fw":
            return False
        log.info(f"[transport] {self.node.name} recv: {msg}")
        sid = msg.get("id")
        current_session = None
        for s in self.sessions:
            if s.id == sid:
                current_session = s
        assert(current_session is not None)
        if current_session.state == SessionState.FIN:
            return True
        recv_eprs_list: List[List[QuantumModel]] = msg.get("eprs", [])

        if self.node != current_session.dest:
            current_session.state = SessionState.WIN_PROBE
            new_eprs_list = self._repeater_win_strategy(
                current_session, recv_eprs_list)

            next_hop = current_session.get_next_hop(self.node)
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            msg["eprs"] = new_eprs_list
            log.info(f"[transport] {self.node.name}->{next_hop.name}: {msg}")
            packet = ClassicPacket(msg=msg, src=self.node, dest=next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
        else:
            current_session.state = SessionState.SWAP
            ret_eprs_list = self._dest_win_strategy(
                current_session, recv_eprs_list)
            response_msg = {"id": current_session.id,
                            "cmd": "bw", "eprs": ret_eprs_list}
            next_hop = current_session.get_prev_hop(self.node)
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            log.info(
                f"[transport] {self.node.name}->{next_hop.name}: {response_msg}")
            packet = ClassicPacket(
                msg=response_msg, src=self.node, dest=next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
        return True

    def handle_bw(self, node, event: RecvClassicPacket) -> List[QuantumModel]:
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "bw":
            return False
        log.info(f"[transport] {self.node.name} recv: {msg}")
        sid = msg.get("id")
        current_session = None
        for s in self.sessions:
            if s.id == sid:
                current_session = s
        assert(current_session is not None)
        if current_session.state == SessionState.FIN:
            return True
        recv_eprs_list: List[List[QuantumModel]] = msg["eprs"]

        if current_session.src != self.node:
            current_session.state = SessionState.SWAP
            next_hop = current_session.get_prev_hop(self.node)
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            log.info(f"[transport] {self.node.name}->{next_hop.name}: {msg}")
            packet = ClassicPacket(msg=msg, src=self.node, dest=next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)

            self.swap(current_session, recv_eprs_list)
            current_session.state = SessionState.READY
        else:
            if len(current_session.path) == 2:
                # for two adjancent nodes
                epr_lists = recv_eprs_list[-1]
                dest = current_session.dest

                src_memory = self.node.get_memory(
                    f"{self.node.name}-{dest.name}")
                dest_memory = dest.get_memory(f"{dest.name}-{self.node.name}")
                dest_session = None
                for tmp_s in dest.apps[-1].sessions:
                    if tmp_s.id == current_session.id:
                        dest_session = tmp_s
                        break
                assert(dest_session is not None)

                for epr in epr_lists:
                    src_qubit = src_memory.read(epr)
                    dest_qubit = dest_memory.read(epr)
                    if src_qubit is None:
                        log.info(f"[{self.node.name}]: {current_session.id} src qubit {epr} is not in memory")
                        continue
                    if dest_qubit is None:
                        log.info(f"[{self.node.name}]:  {current_session.id} dest qubit {epr} is not in memory")
                        continue
                    fidelity = calculate_fidelity(src_qubit.state.rho)
                    log.info(
                        f"[transport] {self.node.name} swap successful, fidelity: {fidelity}")
                    current_session.success_list.append(fidelity)
                current_session.last_gen = len(epr_lists)
                current_session.to_eprs = []
                dest_session.from_eprs = []
                current_session.state = SessionState.READY
                dest_session.state = SessionState.READY
                self.update_Session_state(current_session)
            else:
                current_session.state = SessionState.SWAP
        return True

    def swap(self, s: Session, recv_eprs_list: List[List[QuantumModel]]):
        final_win = len(recv_eprs_list[-1])

        prev_node = s.get_prev_hop(self.node)
        next_node = s.get_next_hop(self.node)
        prev_memory: QuantumMemory = self.node.get_memory(
            f"{self.node.name}-{prev_node.name}")
        next_memory: QuantumMemory = self.node.get_memory(
            f"{self.node.name}-{next_node.name}")

        path_idx = s.get_index(self.node)
        prev_idx = path_idx - 1
        next_idx = path_idx + 1

        results = []

        # BSM
        for i in range(final_win):
            prev_qubit_name = recv_eprs_list[prev_idx][i]
            next_qubit_name = recv_eprs_list[path_idx][i]
            prev_qubit: Qubit = prev_memory.read(prev_qubit_name)
            next_qubit: Qubit = next_memory.read(next_qubit_name)

            if prev_qubit is None:
                log.info(f"{self.node.name}: session {s.id} no prev qubit named {prev_qubit_name}")
                continue
            if next_qubit is None:
                log.info(f"{self.node.name}: session {s.id} no next qubit named {next_qubit_name}")
                continue

            CNOT(next_qubit, prev_qubit)
            H(next_qubit)

            c0 = prev_qubit.measure()
            c1 = next_qubit.measure()
            x = 1 if c0 == 1 else 0
            z = 1 if c1 == 1 else 0
            results.append({"idx": i, "q1": prev_qubit_name,
                           "q2": next_qubit_name, "x": x, "z": z})

        cchannel: ClassicChannel = self.node.get_cchannel(prev_node)
        msg = {"cmd": "swap", "id": s.id,
               "node_idx": path_idx, "result": results}
        log.info(f"[transport] {self.node.name}->{s.src.name}: {msg}")
        packet = ClassicPacket(msg=msg, src=self.node, dest=s.src)
        cchannel.send(packet=packet, next_hop=prev_node)

        # clear occupy
        s.from_eprs = []
        s.to_eprs = []

    def success(self, node, event: RecvClassicPacket):
        msg: Dict = event.packet.get()
        if event.dest != self.node:
            log.info("error")
            return False

        cmd = msg.get("cmd", "")
        if cmd != "swap":
            return False
        log.info(
            f"[transport] {self.node.name} recv: {msg} from {event.packet.src.name}")
        sid = msg.get("id")
        current_session = None
        for s in self.sessions:
            if s.id == sid:
                current_session = s
        assert(current_session is not None)

        results = msg["result"]
        send_node_idx = msg["node_idx"]

        current_session.src_results[send_node_idx] = results

        if len(current_session.src_results) < len(current_session.path) - 2:
            log.debug("not ready to succ")
            return True

        # retrive qubit status
        next_hop = current_session.get_next_hop(self.node)
        next_memory = self.node.get_memory(f"{self.node.name}-{next_hop.name}")

        next_result = current_session.src_results[1]
        dest_result = current_session.src_results[len(current_session.path)-2]

        dest_node = current_session.dest
        dest_transport_app: FIFOTransportApp = dest_node.apps[-1]
        dest_session = None
        for s in dest_transport_app.sessions:
            if s.id == sid:
                dest_session = s
        assert(dest_session is not None)

        before_dest_node = dest_session.get_prev_hop(dest_node)
        dest_memory = dest_node.get_memory(
            f"{dest_node.name}-{before_dest_node.name}")

        actually_win = min([ len(current_session.src_results[i]) for i in range(1, len(current_session.path)-1)])
        # max i is the current window
        for i in range(actually_win):
            src_qubit_name = next_result[i]["q1"]
            dest_qubit_name = dest_result[i]["q2"]
            src_qubit = next_memory.read(src_qubit_name)
            dest_qubit = dest_memory.read(dest_qubit_name)

            if src_qubit is None:
                log.info(f"{self.node.name}: session {current_session.id} no src qubit named {src_qubit_name}")
                continue
            if dest_qubit is None:
                log.info(f"{self.node.name}: session {current_session.id} no dest qubit named {dest_qubit_name}")
                continue

            for n in range(1, len(current_session.path) - 1):
                result = current_session.src_results[n]
                x = result[i]["x"]
                z = result[i]["z"]
                if x == 1:
                    X(src_qubit)
                if z == 1:
                    Z(src_qubit)

            fidelity = calculate_fidelity(src_qubit.state.rho)
            current_session.success_list.append(fidelity)
            log.info(
                f"[transport] {self.node.name} distribution successful, fidelity: {fidelity} eprs: {src_qubit_name}-{dest_qubit_name}")

        current_session.last_gen = len(results)
        # clean qubits and status
        current_session.from_eprs = []
        current_session.to_eprs = []
        dest_session.from_eprs = []
        dest_session.from_eprs = []
        current_session.src_results = {}

        dest_session.state = SessionState.READY
        current_session.state = SessionState.READY
        self.update_Session_state(current_session)
        return True

    def _src_win_strategy(self, s: Session):
        next_hop = s.get_next_hop(self.node)
        next_memory: QuantumMemory = self.node.get_memory(
            f"{self.node.name}-{next_hop.name}")

        assert(next_memory is not None)
        epr_list = []
        for e in next_memory._storage:
            if e is not None:
                occupied = False
                for tmp_s in self.sessions:
                    if e.name in tmp_s.from_eprs or e.name in tmp_s.to_eprs:
                        occupied = True
                        break
                if not occupied:
                    epr_list.append(e.name)

        for e in epr_list:
            s.to_eprs.append(e)
        return epr_list

    def _repeater_win_strategy(self, s: Session, recv_eprs_list: List[List[QuantumModel]]) -> List[List[QuantumModel]]:
        prev_node = s.get_prev_hop(self.node)
        next_node = s.get_next_hop(self.node)
        prev_memory: QuantumMemory = self.node.get_memory(
            f"{self.node.name}-{prev_node.name}")
        next_memory: QuantumMemory = self.node.get_memory(
            f"{self.node.name}-{next_node.name}")

        actual_prev_eprs = []
        for e_name in recv_eprs_list[-1]:
            if not prev_memory.get(e_name):
                continue

            occupied = False
            for tmp_s in self.sessions:
                if e_name in tmp_s.from_eprs or e_name in tmp_s.to_eprs:
                    occupied = True
                    break
            if not occupied:
                actual_prev_eprs.append(e_name)
        actual_prev_eprs_len = len(actual_prev_eprs)

        first_next_eprs = [
            e.name for e in next_memory._storage if e is not None]
        try_next_eprs = []
        for e_name in first_next_eprs:
            occupied = False
            for tmp_s in self.sessions:
                if e_name in tmp_s.from_eprs or e_name in tmp_s.to_eprs:
                    occupied = True
                    break
            if not occupied:
                try_next_eprs.append(e_name)
        try_next_eprs_len = len(try_next_eprs)
    
        total_win = min(try_next_eprs_len, actual_prev_eprs_len)


        for e in actual_prev_eprs[:total_win]:
            s.from_eprs.append(e)

        node_id = int(self.node.name[1:])
        next_id = int(next_node.name[1:])

        if total_win == 0:
            try_next_eprs = []
        elif node_id < next_id:
            try_next_eprs = try_next_eprs[:total_win]
        else:
            try_next_eprs = try_next_eprs[-total_win:]

        for e in try_next_eprs[:total_win]:
            s.to_eprs.append(e)

        ret_eprs_list = [epr_list[:total_win] for epr_list in recv_eprs_list[:-1]]
        ret_eprs_list.append(actual_prev_eprs[:total_win])
        ret_eprs_list.append(try_next_eprs)

        return ret_eprs_list

    def _dest_win_strategy(self, s: Session, recv_eprs_list: List[List[QuantumModel]]) -> List[List[QuantumModel]]:
        prev_node = s.get_prev_hop(self.node)
        prev_memory: QuantumMemory = self.node.get_memory(
            f"{self.node.name}-{prev_node.name}")

        actual_prev_eprs = []
        for e_name in recv_eprs_list[-1]:
            if not prev_memory.get(e_name):
                continue

            occupied = False
            for tmp_s in self.sessions:
                if e_name in tmp_s.from_eprs or e_name in tmp_s.to_eprs:
                    occupied = True
                    break
            if not occupied:
                actual_prev_eprs.append(e_name)
        actual_prev_eprs_len = len(actual_prev_eprs)

        total_win = actual_prev_eprs_len

        actual_prev_eprs = actual_prev_eprs[:total_win]

        for e in actual_prev_eprs:
            s.from_eprs.append(e)

        ret_eprs_list = [epr_list[:total_win] for epr_list in recv_eprs_list[:-1]]
        ret_eprs_list.append(actual_prev_eprs)
        return ret_eprs_list

    def _parse_session(self, s: Session):
        src_name = s.src.name
        dest_name = s.dest.name
        sid = s.id

        path_name = []
        for n in s.path:
            path_name.append(n.name)

        return {"cmd": "syn", "src": src_name, "dest": dest_name, "id": sid, "path": path_name}

    def _load_session(self, session_dict):
        sid = session_dict["id"]
        src = self.net.get_node(session_dict["src"])
        assert(src is not None)

        dest = self.net.get_node(session_dict["dest"])
        assert(dest is not None)

        path = []

        for n_name in session_dict["path"]:
            n = self.net.get_node(n_name)
            assert(n is not None)
            path.append(n)

        s = Session(id=sid, src=src, dest=dest,
                    path=path, state=SessionState.INIT)
        return s
