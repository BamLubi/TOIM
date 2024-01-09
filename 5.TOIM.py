import numpy as np

from env.vcs import VCS


def RUN(n_vehicle, n_task):
    vcs = VCS(n_vehicle, n_task, alg="TOIM")

    def get_TOIM_target(vehicle_id, task_id, sensed_time) -> float:
        wn = vcs.get_WN(vehicle_id, task_id, sensed_time)
        r = vcs.get_reward(vehicle_id, task_id, sensed_time, wn)
        
        if wn > 1:
            return -r
        else:
            return -(1 - wn) * r

    def ALG_TOIM(vehicles, tasks, time_slice) -> list[int]:
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
                    r = get_TOIM_target(vehicles[i], tasks[j], time_slice)
                    if r > max_reward:
                        max_reward = r
                        sel_task = j
            
            visit[sel_task] = 1
            decision[i] = tasks[sel_task]
            
            allocated_tasks_cnt += 1
            if allocated_tasks_cnt >= len(tasks):
                break
        return decision

    vcs.run(ALG_TOIM)

    return vcs.get_stats()


if __name__ == '__main__':
    # for n_vehicle in [10, 20, 30, 40, 50]:
    #     a = RUN(n_vehicle, 300)
    for n_task in [100, 200, 300, 400, 500]:
        a = RUN(40, n_task)


# Vehicles:10, Tasks:300, City:(51,71), EP:105.9188, TC:0.2467, CPC:4.2934
# Vehicles:20, Tasks:300, City:(51,71), EP:133.0939, TC:0.4033, CPC:3.3001
# Vehicles:30, Tasks:300, City:(51,71), EP:84.1564, TC:0.4833, CPC:1.7413
# Vehicles:40, Tasks:300, City:(51,71), EP:84.1543, TC:0.5, CPC:1.6831
# Vehicles:50, Tasks:300, City:(51,71), EP:106.2766, TC:0.6533, CPC:1.6268

# Vehicles:40, Tasks:100, City:(51,71), EP:91.1767, TC:1.0, CPC:0.9118
# Vehicles:40, Tasks:200, City:(51,71), EP:123.2206, TC:0.835, CPC:1.4757
# Vehicles:40, Tasks:300, City:(51,71), EP:91.7864, TC:0.57, CPC:1.6103
# Vehicles:40, Tasks:400, City:(51,71), EP:96.3306, TC:0.4725, CPC:2.0387
# Vehicles:40, Tasks:500, City:(51,71), EP:77.8425, TC:0.396, CPC:1.9657