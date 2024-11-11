from enum import Enum
from typing import Any, Optional, List, Dict, Tuple
from qns.entity import QNode, Application
from qns.entity import ClassicChannel, ClassicPacket, RecvClassicPacket, QuantumMemory, QuantumChannel
from qns.network import QuantumNetwork
from qns.simulator import Simulator, func_to_event
import qns.utils.log as log
from qns.models.qubit import Qubit, X, CNOT, H, Z
from transport.transport_protocol import Session, calculate_fidelity


class ControllerState(Enum):
    BEFORE = -1
    INIT = 0
    INFO = 1
    SWAP = 2


class ControllerTransportApp(Application):
    def __init__(self, net: QuantumNetwork):
        super().__init__()
        self.net = net
        self.sessions: List[Session] = []
        self.state: ControllerState = ControllerState.BEFORE

        self.time_slot = 0
        self.current_sessions: List[Session] = []

        self.collect_info_nodes: List[QNode] = []
        self.collected_info_nodes: List[QNode] = []
        self.collected_eprs: Dict[str, Dict[str, str]] = {}

        self.swap_nodes: List[QNode] = []
        self.successful_sessions: List[Session] = []

        self.swap_strategy: Dict[str, Any] = {}

        self.success_dict: Dict[str, List[float]] = {}

    def install(self, node, simulator: Simulator):
        super().install(node, simulator)
        self.node: QNode = self._node

        event = func_to_event(self._simulator.ts, fn=self.start_collect_info, by=self.node)
        self._simulator.add_event(event)

        event2 = func_to_event(self._simulator.ts + self._simulator.time(sec=0.1), fn=self._lazy_start, by=self.node)
        self._simulator.add_event(event2)

        self.add_handler(self.handle_collect_result, [RecvClassicPacket])
        self.add_handler(self.handle_done, [RecvClassicPacket])

    def _lazy_start(self):
        if self.state == ControllerState.BEFORE:
            self.state = ControllerState.INFO
            self.start_collect_info()

    def init_session(self, src: QNode, dest: QNode, order: int = 0, start_time: Optional[float] = 0, end_time: Optional[float] = None, name=None):
        src = src
        dest = dest

        start_time = self._simulator.time(sec=start_time)
        if end_time is not None:
            end_time = self._simulator.time(sec=end_time)

        path = self.net.query_route(src, dest)[order][2]
        if name is None:
            s = Session(src=src, dest=dest, path=path, start_time=start_time, end_time=end_time)
        else:
            s = Session(src=src, dest=dest, path=path, start_time=start_time, end_time=end_time, id=name)
        self.success_dict[s.id] = []

        if s.start_time is not None:
            def s_start():
                self.sessions.append(s)

                # add session to the src
                for app in src.apps:
                    if isinstance(app, SlaveTransportApp):
                        app.src_sessions.append(s)
                        break

                for n in s.path:
                    for app in n.apps:
                        if isinstance(app, SlaveTransportApp):
                            app.sessions.append(s)

                if self.state == ControllerState.BEFORE:
                    self.state = ControllerState.INFO
                    self.start_collect_info()

            event = func_to_event(s.start_time, fn=s_start, by=self.node)
            self._simulator.add_event(event)

        if s.end_time is not None:
            def s_fin():
                self.sessions.remove(s)
            event = func_to_event(s.end_time, fn=s_fin, by=self.node)
            self._simulator.add_event(event)

    def start_collect_info(self):
        self.state = ControllerState.INFO

        log.info(f"[{self.node}] start new time slot {self.time_slot}")

        self.current_sessions.clear()
        for s in self.sessions:
            self.current_sessions.append(s)

        related_nodes = []
        for s in self.current_sessions:
            for n in s.path:
                if n not in related_nodes:
                    related_nodes.append(n)

        msg = {"cmd": "collect", "ts": self.time_slot}
        for n in related_nodes:
            self.collect_info_nodes.append(n.name)
            log.info("[{}] send msg to {}: {}".format(self.node, n, msg))
            self._tell_node(n, msg)

        if len(self.current_sessions) == 0:
            event2 = func_to_event(self._simulator.tc + self._simulator.time(sec=0.2), fn=self._next_time_slot, by=self.node)
            self._simulator.add_event(event2)

    def handle_collect_result(self, node, event: RecvClassicPacket) -> Optional[bool]:
        """
        collect the epr information of the network
        """
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "collect_result":
            return False
        log.info(f"[{self.node.name}] receive {msg}")
        node_name = msg.get("node")
        ts = msg.get("ts")
        result = msg.get("result")
        assert (ts == self.time_slot)
        self.collected_info_nodes.append(node_name)
        self.collected_eprs[node_name] = result

        log.info("[{}] collect info: {}/{}".format(self.node, len(self.collected_info_nodes), len(self.collect_info_nodes)))

        if len(self.collected_info_nodes) != len(self.collect_info_nodes):
            return True

        self.start_epr_distribution()

    def start_epr_distribution(self):
        """
        calculate the swapping results
        """

        # step 1: get the eprs of each link
        related_links = {}
        for s in self.current_sessions:
            for current_node in s.path[:-1]:
                next_node = s.get_next_hop(current_node)
                link: QuantumChannel = current_node.get_qchannel(next_node)

                if link.name not in related_links:
                    related_links[link.name] = []
                else:
                    continue

                eprs_list_from_current_node = self.collected_eprs[current_node.name][f"{current_node.name}-{next_node.name}"]
                eprs_list_from_next_node = self.collected_eprs[next_node.name][f"{next_node.name}-{current_node.name}"]

                for epr_name in eprs_list_from_current_node:
                    if epr_name in eprs_list_from_next_node:
                        related_links[link.name].append(epr_name)

        # step 2: get the entanglement distribution strategy
        # INPUT: related_links = {link_name: [epr_name, ...]}
        # OUTPUT: result = {key: node_name, value: [{key: src_node, value: [(epr_name_1, epr_name_2), ...]}]}

        result = self.core_strategy(related_links)
        self.swap_strategy = result

        for dest_node_name, strategy in result.items():
            msg = {"cmd": "swap", "ts": self.time_slot, "strategy": strategy}
            log.info("[{}] send msg to {}: {}".format(self.node, dest_node_name, msg))
            self._tell_node(dest_node_name, msg)

        self.swap_strategy = result
        self.state = ControllerState.SWAP

    def handle_done(self, node, event: RecvClassicPacket) -> Optional[bool]:

        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "done":
            return False
        log.info(f"[{self.node.name}] receive {msg}")

        sid = msg.get("id")
        result = msg.get("result")

        ts = msg.get("ts")
        assert (ts == self.time_slot)

        current_session = [s for s in self.current_sessions if s.id == sid][0]

        for f in result:
            self.success_dict[sid].append(f)

        self.successful_sessions.append(current_session)

        log.info("[{}] {}/{} sessions are done".format(self.node, len(self.successful_sessions), len(self.current_sessions)))

        if len(self.successful_sessions) != len(self.current_sessions):
            return True

        log.info("[{}] all sessions are done".format(self.node))
        self.state = ControllerState.INIT
        self._next_time_slot()

    def core_strategy(self, related_links: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        core entanglement distribution strategy here
        """

        # PU algorithm in ``Effective routing design for remote entanglement  generation on quantum network''

        # step 0: calcuate related tables
        session_links = {}  # {"req1": ["n1-n3", "n3-n4", "n4-n5"], "req2": ["n2-n3", "n3-n4", "n4-n6"]}

        for s in self.current_sessions:
            session_links[s.id] = []
            for current_node in s.path[:-1]:
                next_node = s.get_next_hop(current_node)
                link: QuantumChannel = current_node.get_qchannel(next_node)
                session_links[s.id].append(link.name)

        link_sessions = {}  # {"link1": ["req1", "req2"], "link2": ["req1", "req2"]}
        for link_name in related_links.keys():
            link_sessions[link_name] = []
            for s in self.current_sessions:
                if link_name in session_links[s.id]:
                    link_sessions[link_name].append(s.id)

        link_cap = {}  # {"n1-n3": 2, "n2-n3": 4, ...}
        for link_name, link_eprs in related_links.items():
            link_cap[link_name] = len(link_eprs)

        # step 1: fill the capacity of each link
        win = 0
        link_try = {}

        while True:
            # print(win)
            win = win + 1
            link_try.clear()

            for link_name in related_links.keys():
                link_try[link_name] = 0

            for s in self.current_sessions:
                for link_name in session_links[s.id]:
                    link_try[link_name] += win

            skip = True
            for link_name, link_win in link_try.items():
                if link_win < link_cap[link_name]:
                    skip = False
            if skip:
                break

        congested_links = []
        for link in link_sessions.keys():
            if link_try[link] > link_cap[link]:
                congested_links.append(link)

        session_win = {}  # {"req1": 1, "req2": 2}
        for s in self.current_sessions:
            session_win[s.id] = win

        max_win = win  # the max windows

        # step 2: avoid congestion:
        for congested_link in congested_links:
            # print(233, congested_link)
            for win in range(max_win, -1, -1):
                for s_id in link_sessions[congested_link]:
                    if session_win[s_id] > win:
                        session_win[s_id] = win

                new_cap = sum([session_win[s_id] for s_id in link_sessions[congested_link]])
                if new_cap <= link_cap[congested_link]:
                    break

        # step 3: alloc entanglements

        occupied_eprs_set = set()

        result = {}

        for s in self.current_sessions:
            if len(s.path) == 2:
                repeater = s.path[1]
                if repeater.name in result.keys():
                    continue
                result[repeater.name] = []
            else:
                for n in s.path[1:-1]:
                    if n.name in result.keys():
                        continue
                    result[n.name] = []

        for s in self.current_sessions:
            src_name = s.src.name
            win = session_win[s.id]  # windows for this session

            if len(s.path) == 2:
                # without swap (for two adjacent nodes)
                dest_node_name = s.path[1].name

                link = s.src.get_qchannel(s.dest)
                link_name = link.name
                available_eprs = [epr for epr in related_links[link_name] if epr not in occupied_eprs_set]

                assign_eprs = available_eprs[: min(len(available_eprs), win)]
                for epr in assign_eprs:
                    occupied_eprs_set.add(epr)

                session_result = {"src": src_name}
                session_result["eprs"] = assign_eprs
                session_result["type"] = "direct"
                session_result["id"] = s.id
                session_result["node_idx"] = 1
                result[dest_node_name].append(session_result)
            else:
                # swap mode (for remote entanglement distribution)

                link_eprs = {}
                for link_name in session_links[s.id]:
                    available_eprs = [epr for epr in related_links[link_name] if epr not in occupied_eprs_set]
                    assign_eprs = available_eprs[: min(len(available_eprs), win)]
                    for epr in assign_eprs:
                        occupied_eprs_set.add(epr)
                    link_eprs[link_name] = assign_eprs

                for n in s.path[1:-1]:
                    prev_node = s.get_prev_hop(n)
                    next_node = s.get_next_hop(n)

                    prev_channel = n.get_qchannel(prev_node)
                    next_channel = n.get_qchannel(next_node)

                    prev_eprs = link_eprs[prev_channel.name]
                    next_eprs = link_eprs[next_channel.name]

                    actual_win = min(len(prev_eprs), len(next_eprs), win)

                    path_idx = s.path.index(n)
                    session_result = {"src": src_name, "id": s.id, "type": "swap", "eprs": [], "node_idx": path_idx}

                    for idx in range(actual_win):
                        session_result["eprs"].append((prev_eprs[idx], next_eprs[idx]))

                    result[n.name].append(session_result)

        # print("debug point 1", related_links)
        # print(link_cap)
        # print(session_win)
        # print(result)
        log.info(f"[{self.node.name}] generate strategy: {session_win}")
        return result

    def _tell_node(self, dest: QNode, msg: Any):
        """
        send message to any nodes, or send message to myself
        """
        if isinstance(dest, str):
            dest = self.net.get_node(dest)

        packet = ClassicPacket(msg=msg, src=self.node, dest=dest)
        if dest != self.node:
            next_hop: QNode = self.net.query_route(self.node, dest)[0][1]
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
        else:
            event = RecvClassicPacket(t=self._simulator.tc, packet=packet, dest=self.node, by=self.node)
            self._simulator.add_event(event)

    def _next_time_slot(self):
        self.collect_info_nodes.clear()
        self.collected_eprs.clear()
        self.collected_info_nodes.clear()

        self.swap_nodes.clear()
        self.swap_strategy.clear()
        self.successful_sessions.clear()

        self.state = ControllerState.INIT
        self.time_slot += 1
        self.current_sessions.clear()
        self.start_collect_info()
        log.info(f"[{self.node.name}] start time slot {self.time_slot}")


class SlaveTransportApp(Application):
    def __init__(self, net: QuantumNetwork, controller: QNode):
        super().__init__()
        self.net = net
        self.controller = controller

        self.src_sessions: List[Session] = []
        self.sessions: List[Session] = []

        # store all swap informations for source node
        self.swap_messages: Dict[str, Any] = {}

        self.ts = 0

    def install(self, node, simulator: Simulator):
        super().install(node, simulator)
        self.node: QNode = node

        self.add_handler(self.handle_collect_info, [RecvClassicPacket])
        self.add_handler(self.handle_swap, [RecvClassicPacket])
        self.add_handler(self.handle_result, [RecvClassicPacket])

    def handle_collect_info(self, node, event: RecvClassicPacket) -> Optional[bool]:
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "collect":
            return False

        log.info(f"[{self.node.name}] receive {msg}")

        ts = msg.get("ts")
        self.ts = ts

        # clean state
        self.swap_messages.clear()
        for s in self.sessions:
            s.src_results.clear()

        epr_results = {}

        for m in self.node.memories:
            if isinstance(m, QuantumMemory):
                epr_results[m.name] = [epr.name for epr in m._storage if epr is not None]

        msg = {"cmd": "collect_result", "ts": self.ts, "result": epr_results, "node": self.node.name}

        log.info("[{}] send msg to {}: {}".format(self.node, self.controller, msg))
        self._tell_node(self.controller, msg)
        return True

    def handle_swap(self, node, event: RecvClassicPacket) -> Optional[bool]:
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "swap":
            return False

        log.info(f"[{self.node.name}] receive {msg}")
        ts = msg.get("ts")
        if ts != self.ts:
            assert ("time slot not correct")

        strategy = msg.get("strategy")

        for session_strategy in strategy:
            src_node_name = session_strategy["src"]
            type_name = session_strategy["type"]
            eprs = session_strategy["eprs"]
            sid = session_strategy["id"]
            node_idx = session_strategy["node_idx"]

            if type_name == "direct":
                to_src_message = {"cmd": "result", "id": sid, "ts": self.ts, "node_idx": node_idx, "type": type_name, "result": eprs}
                log.info("[{}] send msg to {}: {}".format(self.node, src_node_name, to_src_message))
                self._tell_node(src_node_name, to_src_message)
            else:
                swap_results = []

                current_session = [s for s in self.sessions if s.id == sid][0]
                prev_node = current_session.get_prev_hop(self.node)
                next_node = current_session.get_next_hop(self.node)

                prev_memory = self.node.get_memory(f"{self.node.name}-{prev_node.name}")
                next_memory = self.node.get_memory(f"{self.node.name}-{next_node.name}")

                for idx, epr_pair in enumerate(eprs):
                    prev_qubit_name = epr_pair[0]
                    next_qubit_name = epr_pair[1]

                    prev_qubit: Qubit = prev_memory.read(prev_qubit_name)
                    next_qubit: Qubit = next_memory.read(next_qubit_name)

                    if prev_qubit is None:
                        log.info(f"{self.node.name}: session {sid} no prev qubit named {prev_qubit_name}")
                        continue
                    if next_qubit is None:
                        log.info(f"{self.node.name}: session {sid} no next qubit named {next_qubit_name}")
                        continue

                    CNOT(next_qubit, prev_qubit)
                    H(next_qubit)

                    c0 = prev_qubit.measure()
                    c1 = next_qubit.measure()
                    x = 1 if c0 == 1 else 0
                    z = 1 if c1 == 1 else 0
                    swap_results.append({"idx": idx, "q1": prev_qubit_name, "q2": next_qubit_name, "x": x, "z": z})
                to_src_message = {"cmd": "result", "id": sid, "ts": self.ts, "node_idx": node_idx, "type": type_name, "result": swap_results}
                log.info("[{}] send msg to {}: {}".format(self.node, src_node_name, to_src_message))
                self._tell_node(src_node_name, to_src_message)
        return True

    def handle_result(self, node, event: RecvClassicPacket) -> Optional[bool]:
        msg: Dict = event.packet.get()
        cmd = msg.get("cmd", "")
        if cmd != "result":
            return False

        log.info(f"[{self.node.name}] receive {msg}")

        sid = msg.get("id")
        node_idx = msg.get("node_idx")
        result = msg.get("result")
        result_type = msg.get("type")
        current_session = [s for s in self.sessions if s.id == sid][0]

        current_session.src_results[node_idx] = result

        if result_type == "direct":
            # direct handle (no swapping)
            swap_result = []
            epr_lists = result

            dest = current_session.dest

            src_memory = self.node.get_memory(
                f"{self.node.name}-{dest.name}")
            dest_memory = dest.get_memory(f"{dest.name}-{self.node.name}")
            dest_session = None

            dest_slave_app = [app for app in dest.apps if isinstance(app, SlaveTransportApp)][0]
            for tmp_s in dest_slave_app.sessions:
                if tmp_s.id == current_session.id:
                    dest_session = tmp_s
                    break
            assert (dest_session is not None)

            for epr in epr_lists:
                src_qubit = src_memory.read(epr)
                dest_qubit = dest_memory.read(epr)
                if src_qubit is None:
                    log.info(f"[{self.node.name}] no src qubit named {epr}")
                    continue
                if dest_qubit is None:
                    log.info(f"[{self.node.name}] no dest qubit named {epr}")
                    continue
                fidelity = calculate_fidelity(src_qubit.state.rho)
                log.info(
                    f"[transport] {self.node.name} distribution successful, fidelity: {fidelity}")
                current_session.success_list.append(fidelity)
                swap_result.append(fidelity)
            msg = {"cmd": "done", "ts": self.ts, "id": current_session.id, "result": swap_result}

            log.info("[{}] send msg to {}: {}".format(self.node, self.controller, msg))
            self._tell_node(self.controller, msg)
            return True

        if len(current_session.src_results) != len(current_session.path) - 2:
            log.info(f"[{self.node.name}] not all results received")
            return True

        # retrive qubit status
        next_hop = current_session.get_next_hop(self.node)
        next_memory = self.node.get_memory(f"{self.node.name}-{next_hop.name}")

        next_result = current_session.src_results[1]
        dest_result = current_session.src_results[len(current_session.path)-2]

        dest_node = current_session.dest
        dest_transport_app = [app for app in dest_node.apps if isinstance(app, SlaveTransportApp)][0]
        dest_session = None
        for s in dest_transport_app.sessions:
            if s.id == sid:
                dest_session = s
        assert (dest_session is not None)

        before_dest_node = dest_session.get_prev_hop(dest_node)
        dest_memory = dest_node.get_memory(
            f"{dest_node.name}-{before_dest_node.name}")

        actual_win = min(len(next_result), len(dest_result))

        success_list = []
        # max i is the current window
        for i in range(actual_win):
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
            success_list.append(fidelity)
            current_session.success_list.append(fidelity)
            log.info(
                f"[transport] {self.node.name} distribution successful, fidelity: {fidelity} eprs: {src_qubit_name}-{dest_qubit_name}")

        msg = {"cmd": "done", "id": current_session.id, "ts": self.ts, "result": success_list}

        log.info("[{}] send msg to {}: {}".format(self.node, self.controller, msg))
        self._tell_node(self.controller, msg)
        return True

    def _tell_node(self, dest: QNode, msg: Any):
        """
        send message to any nodes, or send message to myself
        """
        if isinstance(dest, str):
            dest = self.net.get_node(dest)

        packet = ClassicPacket(msg=msg, src=self.node, dest=dest)
        if dest != self.node:
            next_hop: QNode = self.net.query_route(self.node, dest)[0][1]
            cchannel: ClassicChannel = self.node.get_cchannel(next_hop)
            cchannel.send(packet=packet, next_hop=next_hop)
        else:
            event = RecvClassicPacket(t=self._simulator.tc, packet=packet, dest=self.node, by=self.node)
            self._simulator.add_event(event)
