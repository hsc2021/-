from transport.center_pu import ControllerTransportApp
from qns.entity import QuantumChannel
import qns.utils.log as log
from typing import Dict, List, Tuple


class PSControllerTransportApp(ControllerTransportApp):
    def core_strategy(self, related_links: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        core entanglement distribution strategy here
        """

        # PU algorithm in ``Effective routing design for remote entanglement  generation on quantum network''

        # step 0: calcuate related tables
        session_links = {}  # {"req1": ["n1-n3", "n3-n4", "n4-n5"], "req2": ["n2-n3", "n3-n4", "n4-n6"]}
        # print(self.current_sessions)
        for s in self.current_sessions:
            session_links[s.id] = []
            for current_node in s.path[:-1]:
                next_node = s.get_next_hop(current_node)
                link: QuantumChannel = current_node.get_qchannel(next_node)
                session_links[s.id].append(link.name)
        # print(session_links)
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
        session_win = {}

        for s in session_links.keys():
            session_win[s] = 999999999999

        # print(self.sessions)
        # print(link_cap)
        # print(link_sessions)
        for link_name, cap in link_cap.items():
            link_session = link_sessions[link_name]

            link_win = int(cap/len(link_session))

            for s in link_session:
                if link_win < session_win[s]:
                    session_win[s] = link_win
        # print(session_win)
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

        log.info(f"[{self.node.name}] generate strategy: {session_win}")
        return result
