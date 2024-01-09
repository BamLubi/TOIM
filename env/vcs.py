from __future__ import annotations

import math
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from env.location import Location
from env.task import Task
from env.vehicle import Vehicle

CITY_M = 51
CITY_N = 71
MAX_TIME_SLICE = 48
MIN_TIME_SLICE = 0
HIS_NOV_MIN = 0 # history min number of nov
HIS_NOV_MAX = 365 # history max number of nov
SPEED_LEVEL = 79
COPY_VEHICLE = False

N_VEHICLE = 10
N_TASK = 300

BASE_PATH = "TOIM_EXP"
EXP_PATH = BASE_PATH + "\\Experiment - 1"

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

def copy_vehicle() -> None:
    path = ""
    new_path = ""
    cnt = 0
    for file_name in os.listdir(path):
        if cnt >= N_VEHICLE:
            break
        if np.random.uniform() > 0.1:
            continue
        fi = open(path + file_name, "r")
        t = 0
        is_wrong = False
        while True:
            line = fi.readline()
            if not line:
                break
            else:
                line = line.strip("\n").split(",")
                if int(line[0]) != t:
                    is_wrong = True
                    break
                t += 1
        fi.close()
        if is_wrong == False:
            cnt += 1
            shutil.copyfile(path + file_name, new_path + file_name)

class VCS(object):
    def __init__(self, n_vehicle=10, n_task=300, alg="") -> None:
        np.random.seed(2024)
        
        global N_VEHICLE, N_TASK
        N_VEHICLE = n_vehicle
        N_TASK = n_task
        
        self.alg_flg = alg # "AAA", "BBB", "CCC", "TOIM"
        self.vehicles = self.__gen_vehicle()
        self.tasks = self.__gen_task()
        
        self.cur_nov_map = np.load(EXP_PATH + "\\npy\\20150419_nov.npy")
        self.cur_asv_map = np.load(EXP_PATH + "\\npy\\20150419_asv.npy")
        self.all_rci = self.cal_all_rci()

    def __gen_vehicle(self) -> list[Vehicle]:
        global SPEED_LEVEL
        if COPY_VEHICLE == True:
            copy_vehicle()
        vehicles = []
        path = EXP_PATH + "\\data\\vechiles\\" + str(N_VEHICLE) + "\\"
        idx = 1
        for file_name in os.listdir(path):
            fi = open(path + file_name, "r")
            traj = []
            while True:
                line = fi.readline()
                if not line:
                    break
                else:
                    line = line.strip("\n").split(",")
                    traj.append([int(line[0]), int(line[1]), int(
                        line[2]), float(line[3]), int(line[4])])
                    SPEED_LEVEL = max(SPEED_LEVEL, float(line[3]))
            fi.close()
            vehicles.append(Vehicle(idx, traj=traj))
            idx += 1
        return vehicles

    def __gen_task(self) -> list[Task]:
        tasks = []
        
        # 位置
        x_list = np.random.randint(0, CITY_M, N_TASK)
        y_list = np.random.randint(0, CITY_N, N_TASK)
        
        # 生成符合泊松分布的到达时间
        at_list = np.random.poisson(lam=21, size=N_TASK)
        at_list = np.floor(at_list) % MAX_TIME_SLICE
        
        # 创建任务
        for idx in range(0, N_TASK):
            tasks.append(Task(idx + 1, Location(x_list[idx], y_list[idx]), at_list[idx]))

        return tasks

    @staticmethod
    def __gen_map() -> list[list[int]]:
        return [[0 for j in range(CITY_N)] for i in range(CITY_M)]
    
    def run(self, allocation_alg: function) -> None:
        for time_slice in tqdm(range(MIN_TIME_SLICE, MAX_TIME_SLICE)):
            tasks = self.get_avl_tasks(time_slice)
            vehicles = self.get_avl_vehicles(time_slice)
            random.shuffle(tasks)
            random.shuffle(vehicles)
            decision = allocation_alg(vehicles, tasks, time_slice)
            for i in range(0, len(decision)):
                # 计算意愿，并以意愿的概率接受
                wn = self.get_WN(vehicles[i], decision[i], time_slice)
                
                # 无效分配
                if decision[i] == 0:
                    continue
                # 意愿
                if wn < 1 and np.random.uniform() < 1-wn:
                    continue
                
                # 确定奖励
                r = self.get_reward(vehicles[i], decision[i], time_slice, wn)
                
                # 确认分配
                self.vehicles[vehicles[i]-1].sense(decision[i], time_slice, r, self.get_DC(vehicles[i], decision[i]))
                self.tasks[decision[i]-1].sense(vehicles[i], time_slice)
    
    def pre_sense(self, vehicle_id, task_id, sensed_time) -> float:
        self.tasks[task_id-1].sense(vehicle_id, sensed_time)
    
    def pre_desense(self, vehicle_id, task_id, sensed_time) -> float:
        self.tasks[task_id-1].de_sense(vehicle_id, sensed_time)
    
    def can_sense(self, vehicle_id, task_id, sensed_time) -> bool:
        return self.vehicles[vehicle_id - 1].is_free(sensed_time) and self.tasks[task_id - 1].is_free(sensed_time)

    def get_avl_tasks(self, time_slice: int) -> list[int]:
        tasks_id = []
        for task in self.tasks:
            if task.is_free(time_slice) == True:
                tasks_id.append(task.id)
        return tasks_id

    def get_avl_vehicles(self, time_slice: int) -> list[int]:
        vehicles_id = []
        for vehicle in self.vehicles:
            if vehicle.is_free(time_slice) == True:
                vehicles_id.append(vehicle.id)
        return vehicles_id

    def get_WN(self, vehicle_id, task_id, time_slice) -> float:
        st_loc: Location = self.vehicles[vehicle_id-1].loc
        ed_loc: Location = self.tasks[task_id-1].loc
        rci_list = []
        
        # 遍历区间所有RCI
        if st_loc.x >= ed_loc.x and st_loc.y <= ed_loc. y:
            x = st_loc.x - ed_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            for i in range(x - 1, -1, -1):
                for j in range(0, y):
                    rci_list.append(self.get_RCI(ed_loc.x + i , st_loc.y + j, time_slice))
        elif st_loc.x <= ed_loc.x and st_loc.y <= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            for i in range(0, x):
                for j in range(0, y):
                    rci_list.append(self.get_RCI(ed_loc.x + i , st_loc.y + j, time_slice))
        elif st_loc.x <= ed_loc.x and st_loc.y >= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            for i in range(0,  x):
                for j in range(y-1, -1, -1):
                    rci_list.append(self.get_RCI(ed_loc.x + i , st_loc.y + j, time_slice))
        elif st_loc.x >= ed_loc.x and st_loc.y >= ed_loc.y:
            x = st_loc.x - ed_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            for i in range(x - 1, -1, -1):
                for j in range(y-1, -1, -1):
                    rci_list.append(self.get_RCI(ed_loc.x + i , st_loc.y + j, time_slice))
        # 路径的平均RCI
        avg_rci = sum(rci_list) / len(rci_list)
        
        DC = self.get_DC(vehicle_id, task_id)
        wn = 0.4 * (1-DC/120) + 0.6 * (0.5 * np.cos(avg_rci * np.pi) + 0.5)
        
        return wn
    
    def get_DC(self, vehicle_id, task_id) -> int:
        return self.vehicles[vehicle_id-1].loc - self.tasks[task_id-1].loc
    
    def get_EC(self, vehicle_id, task_id, sensed_time) -> float:
        DC = self.get_DC(vehicle_id, task_id)
        DC = 1 if DC == 0 else DC
        
        # 理想时间
        vehicle_loc = self.vehicles[vehicle_id-1].loc
        v = self.get_ASV(vehicle_loc.x, vehicle_loc.y, sensed_time)
        v = SPEED_LEVEL / 4 if v == 0 else v
        TC = 2 * DC / v
        TC = 1 if TC <= 1 else TC
        
        return TC + DC
    
    def get_reward(self, vehicle_id, task_id, sensed_time, wn) -> tuple:
        # flg 代表着是真实分配0，还是预分配1
        # 理想距离
        DC = self.get_DC(vehicle_id, task_id)
        DC = 1 if DC == 0 else DC
        
        # 理想时间
        vehicle_loc = self.vehicles[vehicle_id-1].loc
        v = self.get_ASV(vehicle_loc.x, vehicle_loc.y, sensed_time)
        v = SPEED_LEVEL / 4 if v == 0 else v
        TC = 2 * DC / v
        TC = 1 if TC <= 1 else TC
 
        # 普通算法
        if self.alg_flg != "TOIM":
            R = (TC + DC) * np.random.poisson(lam=1, size=1)[0]
            if R == 0:
                R = TC + DC
            if self.alg_flg == "CCC":
                R += 5
            return round(R, 4)
        
        # 我们的算法
        if self.alg_flg == "TOIM":
            ## RCI
            min_rci = self.get_min_RCI(vehicle_id, task_id, sensed_time) / DC
            max_rci = self.get_max_RCI(vehicle_id, task_id, sensed_time) / DC
            ## EC
            EC = math.exp(min_rci - 0.6) * (TC + DC)
            R = EC
            if wn > 1:
                return EC
            else:
                return EC * (2 - wn)

        return 0
    
    def get_min_RCI(self, vehicle_id: int, task_id: int, sensed_time: int) -> float:
        st_loc: Location = self.vehicles[vehicle_id-1].loc
        ed_loc: Location = self.tasks[task_id-1].loc
        
        if st_loc == ed_loc:
            return self.get_RCI(st_loc.x, st_loc.y, sensed_time)
        
        ## dp 求解最小值
        if st_loc.x >= ed_loc.x and st_loc.y <= ed_loc. y:
            x = st_loc.x - ed_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(0, y):
                    rci = self.get_RCI(ed_loc.x + i , st_loc.y + j, sensed_time)
                    if j == 0 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif i == x-1 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i][j-1], dp[i+1][j])
            return dp[0][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y <= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0, x):
                for j in range(0, y):
                    rci = self.get_RCI(st_loc.x + i, st_loc.y + j, sensed_time)
                    if i == 0 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif j == 0 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i][j-1], dp[i-1][j])
            return dp[x-1][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y >= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0,  x):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(st_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == 0 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i-1][j], dp[i][j+1])
            return dp[x-1][0]
        elif st_loc.x >= ed_loc.x and st_loc.y >= ed_loc.y:
            x = st_loc.x - ed_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(ed_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == x-1 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i+1][j], dp[i][j+1])
            return dp[0][0]

    def get_max_RCI(self, vehicle_id: int, task_id: int, sensed_time: int) -> float:
        st_loc: Location = self.vehicles[vehicle_id-1].loc
        ed_loc: Location = self.tasks[task_id-1].loc
        
        if st_loc == ed_loc:
            return self.get_RCI(st_loc.x, st_loc.y, sensed_time)
        
        ## dp 求解最小值
        if st_loc.x >= ed_loc.x and st_loc.y <= ed_loc. y:
            x = st_loc.x - ed_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(0, y):
                    rci = self.get_RCI(ed_loc.x + i , st_loc.y + j, sensed_time)
                    if j == 0 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif i == x-1 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i][j-1], dp[i+1][j])
            return dp[0][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y <= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0, x):
                for j in range(0, y):
                    rci = self.get_RCI(st_loc.x + i, st_loc.y + j, sensed_time)
                    if i == 0 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif j == 0 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i][j-1], dp[i-1][j])
            return dp[x-1][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y >= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0,  x):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(st_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == 0 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i-1][j], dp[i][j+1])
            return dp[x-1][0]
        elif st_loc.x >= ed_loc.x and st_loc.y >= ed_loc.y:
            x = st_loc.x - ed_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(ed_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == x-1 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i+1][j], dp[i][j+1])
            return dp[0][0]
        
    def get_ASV(self, x, y, t) -> float:
        asv = self.cur_asv_map[x][y][t]
        asv = asv if asv >= 0 else 0
        asv = asv if asv <= SPEED_LEVEL else SPEED_LEVEL
        return asv
    
    def get_NOV(self, x, y, t) -> int:
        return self.cur_nov_map[x][y][t]
    
    def get_RCI(self, x, y, t) -> float:
        if x >= 0 and x <= 50 and y >= 0 and y <= 70:
            return self.all_rci[t][x][y]                                                                        
        else:
            return 0.5
    
    def cal_all_rci(self):
        rci = [[[0 for j in range(CITY_N)] for i in range(CITY_M)] for k in range(MAX_TIME_SLICE)]
        for t in range(MAX_TIME_SLICE):
            for x in range(CITY_M):
                for y in range(CITY_N):
                    mi = np.min(np.array(self.cur_nov_map[x][y]))
                    ma = np.max(np.array(self.cur_nov_map[x][y]))
                    ma = 1 if ma == 0 else ma
                    rci[t][x][y] = 0.5 * (self.cur_nov_map[x][y][t] - mi) / (ma - mi) + 0.5 * (1 - self.get_ASV(x, y, t) / SPEED_LEVEL)
        return rci
    
    def get_EP(self) -> float:
        EP = 0
        for vehicle in self.vehicles:
            EP += vehicle.reward
        return round(EP / N_VEHICLE, 4)
    
    def get_FI(self) -> tuple:
        # 2. 所有任务的数据量 -- SF
        a = 0
        b = 0
        for task in self.tasks:
            wt = task.data
            a += wt
            b += wt ** 2
        SF = (a ** 2) / (len(self.tasks) * b) if b != 0 else 0
        
        return SF
    
    def get_coverage(self) -> float:
        TC = 0
        for task in self.tasks:
            if task.data > 0:
                TC += 1
        TC /= N_TASK
        return round(TC, 4)
    
    def get_stats(self):
        # 4. 平均单个车辆成本
        EP = self.get_EP()

        # 6. 任务覆盖率
        TC = self.get_coverage()
        
        # 7. 单位覆盖率单个车辆成本
        CPC = EP / (TC * 100) if TC != 0 else 0
        
        print("Vehicles:{}, Tasks:{}, City:({},{}), EP:{}, TC:{}, CPC:{}".format(N_VEHICLE, N_TASK, CITY_M, CITY_N, round(EP, 4), round(TC, 4), round(CPC, 4)))
        
        return [EP, TC, CPC]
