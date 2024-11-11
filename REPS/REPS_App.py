from qns.entity.node.app import Application
from qns.simulator.simulator import Simulator
from qns.simulator.event import func_to_event
from qns.network import QuantumNetwork
from qns.utils.rnd import get_rand, get_randint
from qns.network.requests import Request
from REPS_Topology import SessionInfo
from qns.entity.node.node import QNode
import heapq
import pulp
import math


def get_index(n, net):
    i = 0
    for node in net.nodes:
        # print(i, node)
        # exit(1)
        if node.name == n.name:
            return i
        i += 1
    return None


class REPS_APP(Application):
    def __init__(self, net: QuantumNetwork, node: QNode, request_num=5, edge_capa=5, link_prob=0.9, node_prob=0.7):
        super().__init__()
        self.node = node
        self.net = net
        self.request_num = request_num
        self.nodes_number = len(self.net.nodes)
        self.session_paths = {}
        self.session_flow = {}
        self.session_flow_int = {}
        self.nodes_capa = []
        self.nodes_prob = []
        self.links_prob = [[]]
        self.links_capa = [[]]
        # self.session = {0: SessionInfo(0, (0, 1), 1), 1: SessionInfo(1, (1, 2), 1)}
        self.session = {}
        self.session_sd = {}
        self.session_time = {}

        self.edge_capa = edge_capa
        self.link_prob = link_prob
        self.node_prob = node_prob

        # info need to record
        self.f_hat = {}
        self.Eles = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
        self.f = {}
        self.candidate_paths = {}
        self.ECs = {}
        self.selected_path = {}

        self.temp = 0

    def install(self, node: QNode, s: Simulator):
        super().install(node, s)

        t = s.ts
        event = func_to_event(t, self.REPS, by=self)
        self._simulator.add_event(event)

    def Init_Topology(self):
        # print("In Init_Topology")
        # 初始化网络拓扑，包括节点容量、节点交换概率、链路容量、链路交换概率
        self.nodes_capa = [0 for i in range(self.nodes_number)]
        self.nodes_prob = [0 for i in range(self.nodes_number)]
        self.links_prob = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
        self.links_capa = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]

        for i in range(self.nodes_number):
            self.nodes_capa[i] = 6
            self.nodes_prob[i] = self.node_prob

        # print("nodes_capa:", self.nodes_capa, "nodes_prob:", self.nodes_prob)
        # print(self.net.qchannels)
        # exit(1)
        for link in self.net.qchannels:
            # print("link.node_list", link.node_list)
            i = get_index(link.node_list[0], self.net)
            j = get_index(link.node_list[1], self.net)
            # k = get_index(link.node_list[2], self.net)
            # print(i, j, k)
            # exit(1)
            self.links_capa[i][j] = self.links_capa[j][i] = self.edge_capa
            self.links_prob[i][j] = self.links_prob[j][i] = self.link_prob

    def create_request(self):
        # print("In create_request")
        # 创建请求，共有request_num个请求
        pass

    def create_session(self):
        # print("In create_session")
        # 创建会话，每个请求对应一个会话
        reqs_list = []
        for i in range(self.request_num):
            src = get_randint(0, self.nodes_number - 1)
            dst = get_randint(0, self.nodes_number - 1)
            while src == dst:
                dst = get_randint(0, self.nodes_number - 1)
            reqs_list.append(Request(src, dst))
        for i in range(self.request_num):
            new_session = SessionInfo(i, (reqs_list[i].src, reqs_list[i].dest), 1)
            # session = {0: SessionInfo(0, (0, 1), 1), 1: SessionInfo(1, (1, 2), 1)}
            # session_id = {0: (0, 1), 1: (1, 2)}
            self.session[i] = new_session
            self.session_sd[i] = new_session.sd
            # TODO: session_time改成浮点型，而不是列表
            if i not in self.session_time:
                self.session_time[i] = []
            self.session_time[i].append(self._simulator.current_time.sec)
            # print(f"session {i}:", self.session[i].sd, self.session_time[i])
            # exit(1)
        for id in self.session_sd:
            # print(id, self.session_sd[id])
            # exit(1)
            # f_hat = {0: [[0, 0], [0, 0]], 1: [[0, 0], [0, 0]]}
            self.f_hat[id] = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]

    def Dijkstra(self, auxi_links_capa, sd):
        '''
        Dijsktra algorithm to find the shortest path.
        auxi_links_capa: List[List[int]] [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        sd: Tuple[REPS_QNode, REPS_QNode] (2, 1)
        '''
        n = self.nodes_number
        path = []
        distance = [float('inf')] * n
        prev = [-1] * n
        visited = [False] * n
        min_heap = []

        distance[sd[0]] = 0
        heapq.heappush(min_heap, (distance[sd[0]], sd[0]))
        while min_heap:
            # print("min_heap:", min_heap)
            # exit(1)
            current_node = heapq.heappop(min_heap)[1]

            if visited[current_node]:
                continue
            visited[current_node] = True

            if current_node == sd[1]:
                path_to_dest = sd[1]
                while path_to_dest != -1:
                    path.append(path_to_dest)
                    path_to_dest = prev[path_to_dest]
                path.reverse()
                return path

            for v in range(n):
                if auxi_links_capa[current_node][v] <= 0 or visited[v]:
                    continue
                # TODO：最少跳数优先，如果想要最短路径，需要修改添加current_node和节点v之间的距离
                if distance[current_node] + 1 < distance[v]:
                    distance[v] = distance[current_node] + 1
                    prev[v] = current_node
                    heapq.heappush(min_heap, (distance[v], v))

        return path

    def DecomposePathsBasedonFlowMetric(self, sd, flow_metric):
        '''
        sd: Tuple[REPS_QNode, REPS_QNode] (2, 1)
        flow_metric: List[List[int]]
        '''
        # print("In DecomposePathsBasedonFlowMetric")
        full_paths = {}
        while True:
            new_path = self.Dijkstra(flow_metric, sd)
            # print("new_path:", new_path)
            # exit(1)
            if not new_path:
                break
            # new_path = [2, 1]
            new_flow = float('inf')
            for i in range(len(new_path) - 1):
                if flow_metric[new_path[i]][new_path[i+1]] < new_flow:
                    new_flow = flow_metric[new_path[i]][new_path[i+1]]
            # full_paths: {(0, 1): 3}
            full_paths[tuple(new_path)] = new_flow
            # print("full_paths:", full_paths)
            # exit(11)
            # 资源更新
            for i in range(len(new_path) - 1):
                flow_metric[new_path[i]][new_path[i+1]] -= new_flow

            # CHECK
            # print("new_path:", end=' ')
            # for node in new_path:
            #     print(node, end='-->')
            # print("end. flow =", new_flow)

        return full_paths

    def LP_EPS(self, t_hat):
        '''
        t_hat: Dict[int, int]
        '''
        # print("In LP_EPS")
        pos = 0
        pos2reqid = {}
        for id in self.session:
            pos2reqid[pos] = id
            pos += 1
        req_num = len(pos2reqid)

        prob = pulp.LpProblem("LP_EPS", pulp.LpMaximize)

        t = [[pulp.LpVariable(f't[{i}_{k}]', 0, 1, pulp.LpContinuous) for k in range(t_hat[pos2reqid[i]])]
             for i in range(req_num)]
        f = [[[[pulp.LpVariable(f'f[{i}_{k}_{u}_{v}]', 0, 1, pulp.LpContinuous) for v in range(self.nodes_number)]
               for u in range(self.nodes_number)] for k in range(t_hat[pos2reqid[i]])] for i in range(req_num)]

        prob += pulp.lpSum(t[i][k] for i in range(req_num) for k in range(t_hat[pos2reqid[i]]))

        for i in range(req_num):
            for k in range(t_hat[pos2reqid[i]]):
                for u in range(self.nodes_number):
                    flow_con_ki = pulp.lpSum(f[i][k][u][v] - f[i][k][v][u] for v in range(self.nodes_number))
                    if u == self.session_sd[pos2reqid[i]][0]:
                        prob += flow_con_ki == t[i][k]
                    elif u == self.session_sd[pos2reqid[i]][1]:
                        prob += flow_con_ki == -t[i][k]
                    else:
                        prob += flow_con_ki == 0

        for u in range(self.nodes_number):
            for v in range(u + 1):
                ELes_con_uv = pulp.lpSum(f[i][k][u][v] + f[i][k][v][u] for i in range(req_num) for k in
                                         range(t_hat[pos2reqid[i]]))
                prob += ELes_con_uv <= self.Eles[u][v]

        prob.solve()

        f_bar = {}
        for i in range(req_num):
            f_bar_i = []
            for k in range(t_hat[pos2reqid[i]]):
                f_bar_ik = []
                for u in range(self.nodes_number):
                    f_bar_iku = [f[i][k][u][v].varValue for v in range(self.nodes_number)]
                    f_bar_ik.append(f_bar_iku)
                f_bar_i.append(f_bar_ik)
            f_bar[pos2reqid[i]] = f_bar_i

        return f_bar

    def LP_PFS(self, remaining_node_capacity, remaining_link_capacity):
        '''
        remaining_node_capacity: List[int]
        remaining_link_capacity: List[List[int]]
        '''
        # print("In LP Model")
        req_num = len(self.session)
        n = self.nodes_number

        pos2reqid = {pos: id_session[0] for pos, id_session in enumerate(self.session.items())}

        prob = pulp.LpProblem("LP_PFS", pulp.LpMaximize)

        f = [[[pulp.LpVariable(f'f{i}_{u}_{v}', lowBound=0, upBound=None, cat='Continuous') for v in range(n)]
              for u in range(n)] for i in range(req_num)]
        t = [pulp.LpVariable(f't{i}', lowBound=0, upBound=None, cat='Continuous') for i in range(req_num)]
        x = [[pulp.LpVariable(f'x_{u}_{v}', lowBound=0, upBound=None, cat='Continuous') for v in range(u+1)] for u in range(n)]

        prob += pulp.lpSum(t)

        for i in range(req_num):
            for u in range(n):
                flow_expr = pulp.lpSum(f[i][u][v] - f[i][v][u] for v in range(n))
                if u == self.session_sd[pos2reqid[i]][0]:
                    prob += flow_expr == t[i]
                elif u == self.session_sd[pos2reqid[i]][1]:
                    prob += flow_expr == -t[i]
                else:
                    prob += flow_expr == 0

        for u in range(n):
            for v in range(u+1):
                link_res_cons = pulp.lpSum((f[i][u][v] + f[i][v][u]) for i in range(req_num))
                # print("link_res_cons:", link_res_cons)
                prob += link_res_cons <= self.links_prob[u][v] * x[u][v]
                prob += x[u][v] <= remaining_link_capacity[u][v]

        for u in range(n):
            node_res_expr = pulp.lpSum(x[max(u, v)][min(u, v)] for v in range(n))
            prob += node_res_expr <= remaining_node_capacity[u]

        prob.solve()

        if prob.status != pulp.LpStatusOptimal:
            # print("No Solution")
            return None
        # print("remianing_node_capacity:", remaining_node_capacity, "remaining_link_capacity:", remaining_link_capacity)
        # print("prob:", self.links_prob, self.nodes_prob)
        f_astro = {}
        for pos, id in pos2reqid.items():
            f_astro_id = [[0]*n for _ in range(n)]
            for u in range(n):
                for v in range(n):
                    f_astro_id[u][v] = f[pos][u][v].varValue
            f_astro[id] = f_astro_id
        return f_astro

    def ProvisioningforFailureTolerance(self):
        # 计算每条边应该创建的链路数量，为容错性提供服务
        # print("In ProvisioningforFailureTolerance")
        self.Init_Topology()
        self.create_session()
        # print("session:", self.session)
        # exit(1)

        remaining_node_capacity = self.nodes_capa
        remaining_link_capacity = self.links_capa
        # print("remaining_node_capacity:", remaining_node_capacity, "remaining_link_capacity:", remaining_link_capacity)
        # exit(1)
        # TODO: f_astro修改为LP_PFS的返回值，而不是固定值
        # f_astro = {0: [[1, 2], [3, 4]], 1: [[3, 3], [3, 3]]}
        f_astro = self.LP_PFS(remaining_node_capacity, remaining_link_capacity)
        # print("remianing_node_capacity:", remaining_node_capacity, "remaining_link_capacity:", remaining_link_capacity)
        # print("prob:", self.links_prob, self.nodes_prob)
        # print("f_astro:", f_astro)
        # exit(1)
        # print("f_astro", f_astro)

        estanblished_new_path = True
        while (estanblished_new_path):
            # print("In while loop")
            # print("f_astro:", f_astro)
            if not f_astro:
                break
            estanblished_new_path = False
            f_hat_curloop = {}  # {0: [[0, 0], [0, 0]], 1: [[0, 0], [0, 0]]}
            for id_sd in self.session_sd:
                id = id_sd
                value = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
                f_hat_curloop[id] = value
            # print("f_hat_curloop:", f_hat_curloop)
            # exit(1)

            # print("1.1 decompose paths according to f_astro")
            for id_of_session in f_astro:
                # print("id_of_session:", id_of_session)
                # exit(1)
                id = id_of_session
                flow_metric = f_astro[id_of_session]
                # print("flow_metric:", flow_metric)
                # print(self.session_sd[id])
                # exit(1)

                # self.session_sd = {0: (0, 1), 1: (1, 2)}
                # path: {(0, 1): 3}
                # flow_metric: [[3, 3], [3, 3]]
                path = self.DecomposePathsBasedonFlowMetric(self.session_sd[id], flow_metric)
                # print("path:", path)
                # exit(1)

                for path_flow in path:
                    # path_flow: (0, 1)
                    # print("path_flow:", path_flow)
                    # exit(1)
                    same_path = False
                    same_path_pos = 0

                    # 将路径 path 中的每条路径流量 path_flow 更新到 self.session_paths 和 self.session_flow 中
                    if id not in self.session_paths:
                        self.session_paths[id] = []
                        self.session_flow[id] = []
                        temp = list(path_flow)
                        self.session_paths[id].append(temp)
                        self.session_flow[id].append(path[path_flow])

                    else:
                        while (same_path_pos < len(self.session_paths[id])):
                            # print("same_path_pos:", same_path_pos)
                            same_path_pos += 1
                            # print(self.session_paths, id, same_path_pos - 1)
                            if len(self.session_paths[id][same_path_pos - 1]) != len(path_flow):
                                continue
                            same_path = True
                            for i in range(len(self.session_paths[id][same_path_pos - 1])):
                                if self.session_paths[id][same_path_pos - 1][i] != path_flow[i]:
                                    same_path = False
                                    break
                            if same_path:
                                break

                    if same_path:
                        self.session_flow[id][same_path_pos - 1] += path[path_flow]
                    else:
                        path_temp = list(self.session_paths[id])
                        path_temp.append(list(path_flow))
                        # print(self.session_paths, self.session_paths[id])
                        self.session_paths[id] = path_temp
                        self.session_flow[id].append(path[path_flow])
            # Check
            # print("++++++session path_flow++++++ OK")

            # TODO
            # print("1.2 floor paths_flow")
            # print("session_flow:", self.session_flow)
            # exit(1)
            # session_flow: {0: [2, 2], 1: [3, 3]}
            for id in self.session_flow:
                flows = self.session_flow[id]
                self.session_flow_int[id] = []
                for pos in range(len(flows)):
                    # temp = list(self.session_paths[id])
                    path = self.session_paths[id][pos]
                    # print(self.session_paths)
                    # print("path:", path, "temp:", temp)
                    # exit(1)
                    flow = int(flows[pos])
                    # print("flow:", flow)
                    # print("path:", path)
                    for i in range(len(path) - 1):
                        f_hat_curloop[id][path[i]][path[i+1]] += flow
                    self.session_flow[id][pos] -= flow
                    self.session_flow_int[id].append(flow)
            # Check
            # print("++++++session path_flow++++++ OK")

            # print("1.3 update resources node to use")
            link_expected_Eles = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
            for i in range(self.nodes_number):
                for j in range(self.nodes_number):
                    for id in self.session:
                        link_expected_Eles[i][j] += f_hat_curloop[id][i][j]
            # TODO 判断link_expected_Eles是不是对称矩阵？？？
            # print("link_expected_Eles:", link_expected_Eles)
            # exit(1)

            # print("++++++link_expected_Eles:++++++ OK")

            # print("additional step: check remaining resources over each node")
            for u in range(self.nodes_number):
                while True:
                    node_need_to_use = 0
                    for v in range(self.nodes_number):
                        if self.links_prob[u][v] == 0:
                            continue
                        node_need_to_use += math.ceil((link_expected_Eles[u][v] + link_expected_Eles[v][u]) / self.links_prob[u][v])
                    # Check
                    # print("++++++node_need_to_use++++++ OK")
                    if node_need_to_use <= remaining_node_capacity[u]:
                        # print("node_need_to_use succes")
                        break

                    # 超出资源数目，需要减少流量
                    max_flow_path_pos = None
                    max_flow = -1
                    for id_paths in self.session_paths.items():
                        id = id_paths[0]
                        for pos in range(len(id_paths[1])):
                            for i in range(len(self.session_paths[id][pos])):
                                if self.session_paths[id][pos][i] == u:
                                    if max_flow == -1:
                                        max_flow_path_pos = (id, pos)
                                        max_flow = self.session_flow_int[id][pos]
                                    elif self.session_flow_int[id][pos] >= max_flow:
                                        max_flow_path_pos = (id, pos)
                                        max_flow = self.session_flow_int[id][pos]
                    if max_flow == -1:
                        print("ERROR: max_flow == -1.")
                        exit(1)
                    id, pos = max_flow_path_pos

                    for i in range(len(self.session_paths[id][pos]) - 1):
                        left = self.session_paths[id][pos][i]
                        right = self.session_paths[id][pos][i + 1]
                        f_hat_curloop[id][left][right] -= 1
                        link_expected_Eles[left][right] -= 1
                    self.session_flow_int[id][pos] -= 1

            # print("1.4 sort as decreasingly reamining path_flow")

            decresing_path_info = []
            for id in self.session_paths:
                id = id
                for pos in range(len(self.session_paths[id])):
                    decresing_path_info.append((id, pos))
            length = len(decresing_path_info) - 1
            # 排序
            while length >= 0:
                for i in range(length):
                    left_flow = self.session_flow[decresing_path_info[i][0]][decresing_path_info[i][1]]
                    right_flow = self.session_flow[decresing_path_info[i+1][0]][decresing_path_info[i+1][1]]
                    if left_flow < right_flow:
                        temp_pos = decresing_path_info[i]
                        decresing_path_info[i] = decresing_path_info[i+1]
                        decresing_path_info[i+1] = temp_pos
                length -= 1

            # print("1.5 try to +1 OK")
            for path_info in decresing_path_info:
                id = path_info[0]
                pos = path_info[1]
                path = self.session_paths[id][pos]
                res_enought = True
                for i in range(len(path) - 1):
                    if self.links_prob[path[i]][path[i+1]] == 0:
                        continue
                    if math.ceil((link_expected_Eles[path[i]][path[i+1]] + link_expected_Eles[path[i + 1]][path[i]] + 1) /
                                 self.links_prob[path[i]][path[i+1]]) > remaining_link_capacity[path[i]][path[i+1]]:
                        res_enought = False
                        break
                if not res_enought:
                    continue
                for i in range(len(path) - 1):
                    f_hat_curloop[id][path[i]][path[i+1]] += 1
                    link_expected_Eles[path[i]][path[i+1]] += 1
                for i in range(len(path)):
                    if not res_enought:
                        break
                    node_need_to_use = 0
                    u = path[i]
                    for v in range(self.nodes_number):
                        if self.links_prob[u][v] == 0:
                            continue
                        node_need_to_use += math.ceil((link_expected_Eles[u][v] + link_expected_Eles[v][u]) /
                                                      self.links_prob[u][v])
                    if node_need_to_use > remaining_node_capacity[u]:
                        res_enought = False
                        break
                if res_enought:
                    for i in range(len(path) - 1):
                        f_hat_curloop[id][path[i]][path[i+1]] -= 1
                        link_expected_Eles[path[i]][path[i+1]] -= 1

            # print("1.6 update remaining resources OK")
            for u in range(self.nodes_number):
                node_need_to_use = 0
                for v in range(self.nodes_number):
                    if self.links_prob[u][v] == 0:
                        continue
                    link_need_to_use = math.ceil((link_expected_Eles[u][v] + link_expected_Eles[v][u]) /
                                                 self.links_prob[u][v])
                    node_need_to_use += link_need_to_use
                remaining_node_capacity[u] -= node_need_to_use

            # print("1.7 re-solve LP with current reamining resources")
            f_astro = self.LP_PFS(remaining_node_capacity, remaining_link_capacity)
            # print("f_astro:", f_astro)
            # print("remaining_node_capacity:", remaining_node_capacity, "remaining_link_capacity:", remaining_link_capacity)
            # f_astro = {0: [[3, 3], [3, 3]], 1: [[3, 3], [3, 3]]}

            for id in self.session_sd:
                for u in range(self.nodes_number):
                    for v in range(self.nodes_number):
                        self.f_hat[id][u][v] += f_hat_curloop[id][u][v]

            # print("session_flow_int:", self.session_flow_int)
            for id in self.session_flow_int:
                flows = self.session_flow_int[id]
                for flow in flows:
                    if flow > 0:
                        estanblished_new_path = True

    def CreateLinks(self):
        # print("In CreateLinks")
        link_expeted = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
        for id in self.session:
            for u in range(self.nodes_number):
                for v in range(self.nodes_number):
                    link_expeted[u][v] += self.f_hat[id][u][v]
        link_res = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
        for u in range(self.nodes_number):
            for v in range(self.nodes_number):
                if self.links_prob[u][v] == 0:
                    continue
                link_res[u][v] = math.ceil((link_expeted[u][v] + link_expeted[v][u]) / self.links_prob[u][v])
                link_res[v][u] = link_res[u][v]
        for u in range(self.nodes_number):
            for v in range(u):
                if self.links_prob[u][v] == 0:
                    continue
                for cnt in range(link_res[u][v]):
                    if get_rand() <= self.links_prob[u][v]:
                        # link = QuantumChannel(name=f"link_{u}_{v}")
                        # node = self.net.get_node(f"node{u}")
                        # next_hop = self.net.get_node(f"node{v}")
                        # node.add_qchannel(link)
                        # next_hop.add_qchannel(link)
                        # self.net.add_qchannel(link)
                        # print(self.Eles[u][v])
                        self.Eles[u][v] += 1
                self.Eles[v][u] = self.Eles[u][v]

    def EntanglementPathSelection(self):
        # print("In EntanglementPathSelection")
        t_hat = {}
        # print("f_hat:", self.f_hat)
        for id in self.f_hat:
            # print("id:", id)
            t_hat[id] = 0

            sd = self.session_sd[id]
            for v in range(self.nodes_number):
                t_hat[id] += self.f_hat[id][sd[0]][v] - self.f_hat[id][v][sd[0]]

        f_bar = self.LP_EPS(t_hat)
        t_bar = {}
        for id in f_bar:
            f_bar_i = f_bar[id]
            t_bar_i = []
            for k in range(t_hat[id]):
                f_bar_ik = f_bar_i[k]
                t_bar_ik = 0
                for u in range(self.nodes_number):
                    t_bar_ik += f_bar_ik[self.session_sd[id][0]][u] - f_bar_ik[u][self.session_sd[id][0]]
                t_bar_i.append(t_bar_ik)
            t_bar[id] = t_bar_i

        # random routing
        t = {}
        for id in t_bar:
            t_i = []
            for k in range(len(t_bar[id])):
                if get_rand() < t_bar[id][k]:
                    t_i.append(1)
                else:
                    t_i.append(0)
            t[id] = t_i

        for id in t:
            for k in range(len(t[id])):
                if t[id][k] == 0:
                    continue
                f_ik = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
                ik_paths = {}
                ik_flow = {}
                r = 0
                path = self.DecomposePathsBasedonFlowMetric(self.session_sd[id], f_bar[id][k])
                for path_flow in path:
                    ik_paths[r] = list(path_flow)
                    ik_flow[r] = path[path_flow]
                    r += 1
                rnd = get_rand()
                sum = 0.0
                for r in range(len(ik_flow)):
                    sum += ik_flow[r] / t_bar[id][k]
                    if sum >= rnd:
                        break
                for tmp in range(len(ik_paths[r]) - 1):
                    left = ik_paths[r][tmp]
                    right = ik_paths[r][tmp + 1]
                    f_ik[left][right] = 1

                self.f[(id, k)] = f_ik
                if id not in self.candidate_paths:
                    self.candidate_paths[id] = {}
                if k not in self.candidate_paths[id]:
                    self.candidate_paths[id][k] = []
                self.candidate_paths[id][k] = ik_paths[r]

    def EntanglementLinkSelection(self):
        # print("In EntanglementLinkSelection")
        auxi_links_capa = self.Eles.copy()
        y = [[0 for _ in range(self.nodes_number)] for _ in range(self.nodes_number)]
        T = set()

        for id_sd in self.session_sd.keys():
            T.add(id_sd)

        path_weights = {}

        for id_candidatepaths in self.candidate_paths.items():
            id = id_candidatepaths[0]
            for k_path in id_candidatepaths[1].items():
                k = k_path[0]
                weight = 1.0
                for i in range(1, len(k_path[1]) - 1):
                    weight *= self.nodes_prob[k_path[1][i]]
                if id not in path_weights:
                    path_weights[id] = {}
                if k not in path_weights[id]:
                    path_weights[id][k] = 0
                path_weights[id][k] = weight

        while T:
            for u in range(self.nodes_number):
                for v in range(u):
                    if y[u][v] < self.Eles[u][v]:
                        continue
                    auxi_links_capa[u][v] = 0
                    auxi_links_capa[v][u] = 0
                    for id_candidatepaths in self.candidate_paths.items():
                        id = id_candidatepaths[0]
                        to_delete_ks = set()
                        for k_path in id_candidatepaths[1].items():
                            k = k_path[0]
                            for i in range(len(k_path[1]) - 1):
                                if k_path[1][i] == u and k_path[1][i+1] == v:
                                    to_delete_ks.add(k)
                                    break
                                elif k_path[1][i] == v and k_path[1][i+1] == u:
                                    to_delete_ks.add(k)
                                    break
                        for k in to_delete_ks:
                            del self.candidate_paths[id][k]

            min_id = next(iter(T))
            for id in T:
                if id not in self.selected_path:
                    min_id = id
                    break
                elif len(self.selected_path[id]) < len(self.selected_path[min_id]):
                    min_id = id
            # print(min_id, self.candidate_paths)
            if min_id not in self.candidate_paths.keys():
                T.remove(min_id)
                continue
            if self.candidate_paths and self.candidate_paths[min_id]:
                max_k = next(iter(self.candidate_paths[min_id]))
                for k_path in self.candidate_paths[min_id].items():
                    k = k_path[0]
                    if path_weights[min_id][k] > path_weights[min_id][max_k]:
                        max_k = k
                if min_id not in self.selected_path:
                    self.selected_path[min_id] = []
                #  可能有错，需要对selected_path[min_id]进行排序
                self.selected_path[min_id].append(self.candidate_paths[min_id][max_k])
                self.selected_path[min_id].sort()
                for i in range(len(self.candidate_paths[min_id][max_k]) - 1):
                    left = self.candidate_paths[min_id][max_k][i]
                    right = self.candidate_paths[min_id][max_k][i + 1]
                    y[left][right] += 1
                    y[right][left] += 1
                continue
            T.remove(min_id)
            continue

        for id_sd in self.session_sd.items():
            T.add(id_sd[0])

        while T:
            min_id = next(iter(T))
            for id in T:
                if id not in self.selected_path:
                    min_id = id
                    break
                elif len(self.selected_path[id]) < len(self.selected_path[min_id]):
                    min_id = id

            for u in range(self.nodes_number):
                for v in range(self.nodes_number):
                    if y[u][v] < self.Eles[u][v]:
                        continue
                    auxi_links_capa[u][v] = 0
                    auxi_links_capa[v][u] = 0

            path = self.Dijkstra(auxi_links_capa, self.session_sd[min_id])
            if not path:
                T.remove(min_id)
                continue
            if min_id not in self.selected_path:
                self.selected_path[min_id] = []
            self.selected_path[min_id].append(list(path))
            for i in range(len(path) - 1):
                left = path[i]
                right = path[i + 1]
                y[left][right] += 1
                y[right][left] += 1

    def Teleport(self):
        # print("In Teleport")
        # print("selected_path:", self.selected_path)
        for id in self.session:
            self.ECs[id] = 0
        for id in self.selected_path:
            # print("id:", id)
            self.ECs[id] = 0
            for path in self.selected_path[id]:
                success_EC = True
                for i in range(len(path) - 1):
                    if get_rand() > self.nodes_prob[path[i]]:
                        success_EC = False
                        break
                if success_EC:
                    self.ECs[id] += 1
            if self.ECs[id] > 0 and len(self.session_time[id]) == 1:
                self.session_time[id].append(self._simulator.current_time.sec)
        # print(self.ECs)
        for session in self.session:
            # print(f"session {session} ECs: {self.ECs[session]}")
            self.session[session].W -= self.ECs[session]

        return self.ECs

    def Data_Process(self):
        # print("++++++++ ELes ++++++++\n")
        # for u in range(self.nodes_number):
        #     for v in range(self.nodes_number):
        #         print(self.Eles[u][v], end="\t")
        #     print()

        # print("++++++++ selected paths ++++++++\n")
        # for id_paths in self.selected_path.items():
        #     print("--------")
        #     print("id =", id_paths[0])
        #     for path in id_paths[1]:
        #         for node in path:
        #             print(node, end="->")
        #         print("end.")

        # print("++++++++ ELes need to used ++++++++\n")
        eles_to_use = [[0 for _ in range(self.nodes_number)] for _ in range(self.nodes_number)]
        for id_paths in self.selected_path.items():
            for path in id_paths[1]:
                for i in range(len(path) - 1):
                    eles_to_use[path[i]][path[i + 1]] += 1

        # for u in range(self.nodes_number):
        #     for v in range(self.nodes_number):
        #         # self.Eles_num[self.nodes_number] += eles_to_use[u][v]
        #         print(eles_to_use[u][v] + eles_to_use[v][u], end="\t")
        #     print()

        print("++++++++ ECs ++++++++\n")
        for id_ecs in self.ECs.items():
            print(f"<{id_ecs[0]}, {id_ecs[1]}>, ", end="")
        print("end.\n")

        # Statistic
        need_reqs = len(self.session)
        assign_reqs = 0
        max_conns = 0 if not self.session else float('-inf')
        min_conns = 0 if not self.session else float('inf')

        for id_connum in self.ECs.items():
            if id_connum[1] > 0:
                assign_reqs += 1
                max_conns = max(max_conns, id_connum[1])
                min_conns = min(min_conns, id_connum[1])

        max_conns = max_conns if assign_reqs > 0 else 0
        min_conns = min_conns if assign_reqs > 0 else 0

        print(f"{need_reqs}\t{assign_reqs}\t{max_conns}\t{min_conns}\n")
        for id_connum in self.ECs.items():
            if self.ECs[id_connum[0]] == 0:
                continue
            req_id = id_connum[0]
            resources = len(self.selected_path[req_id])
            conns = id_connum[1]
            # self.Eles_num[self.nodes_number] += conns
            print(f"{req_id}\t{resources}\t{conns}\n")

    def REPS(self):
        # set_seed(int(time.time()))
        self.ProvisioningforFailureTolerance()
        self.CreateLinks()
        self.EntanglementPathSelection()
        self.EntanglementLinkSelection()
        self.Teleport()
        self.Data_Process()

        t = self._simulator.current_time + self._simulator.time(sec=100)
        event = func_to_event(t, self.REPS, by=self)
        self._simulator.add_event(event)
