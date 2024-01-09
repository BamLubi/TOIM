import numpy as np

from env.vcs import VCS, sigmoid


def RUN(n_vehicle, n_task):

    vcs = VCS(n_vehicle, n_task, alg="BBB")

    def get_BBB_target(vehicle_id, task_id, sensed_time) -> float:
        return vcs.get_coverage()

    def ALG_BBB(vehicles, tasks, time_slice) -> list[int]:
        if len(tasks) == 0 or len(vehicles) == 0:
            return []
        
        visit = [0] * len(tasks)
        decision = [0] * len(vehicles)
        allocated_tasks_cnt = 0
        for i in range(0, len(vehicles)):
            max_reward = -1000
            sel_task = -1
            for j in range(0, len(tasks)):
                if visit[j] == 0 and vcs.can_sense(vehicles[i], tasks[j], time_slice) == True:
                    r = get_BBB_target(vehicles[i], tasks[j], time_slice)
                    if r >= max_reward:
                        max_reward = r
                        sel_task = j
            
            visit[sel_task] = 1
            decision[i] = tasks[sel_task]
            
            allocated_tasks_cnt += 1
            if allocated_tasks_cnt >= len(tasks):
                break
        return decision

    vcs.run(ALG_BBB)

    return vcs.get_stats()


if __name__ == '__main__':
    # for n_vehicle in [10]:
    # for n_vehicle in [10, 20, 30, 40, 50]:
    #     a = RUN(n_vehicle, 300)
    for n_task in [100, 200, 300, 400, 500]:
        a = RUN(40, n_task)


# Vehicles:10, Tasks:300, City:(51,71), EP:302.9294, TC:0.2033, CPC:14.9006
# Vehicles:20, Tasks:300, City:(51,71), EP:279.4237, TC:0.3567, CPC:7.8336
# Vehicles:30, Tasks:300, City:(51,71), EP:245.7193, TC:0.43, CPC:5.7144
# Vehicles:40, Tasks:300, City:(51,71), EP:183.4861, TC:0.4833, CPC:3.7965
# Vehicles:50, Tasks:300, City:(51,71), EP:199.8697, TC:0.6267, CPC:3.1892

# Vehicles:40, Tasks:100, City:(51,71), EP:118.7275, TC:1.0, CPC:1.1873
# Vehicles:40, Tasks:200, City:(51,71), EP:198.1323, TC:0.805, CPC:2.4613
# Vehicles:40, Tasks:300, City:(51,71), EP:202.151, TC:0.5, CPC:4.043
# Vehicles:40, Tasks:400, City:(51,71), EP:199.1911, TC:0.405, CPC:4.9183
# Vehicles:40, Tasks:500, City:(51,71), EP:186.6425, TC:0.328, CPC:5.6903