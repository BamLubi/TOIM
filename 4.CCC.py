import numpy as np

from env.vcs import VCS


def RUN(n_vehicle, n_task):
    
    vcs = VCS(n_vehicle, n_task, alg="CCC")

    def get_CCC_target(vehicle_id, task_id, sensed_time) -> float:
        return -vcs.get_EC(vehicle_id, task_id, sensed_time)

    def ALG_CCC(vehicles, tasks, time_slice) -> list[int]:
        if len(tasks) == 0 or len(vehicles) == 0:
            return []
        
        visit = [0] * len(tasks)
        decision = [0] * len(vehicles)
        allocated_tasks_cnt = 0
        for i in range(0, len(vehicles)):
            max_reward = -10000
            sel_task = -1
            for j in range(0, len(tasks)):
                if visit[j] == 0 and vcs.can_sense(vehicles[i], tasks[j], time_slice) == True:
                    r = get_CCC_target(vehicles[i], tasks[j], time_slice)
                    if r > max_reward:
                        max_reward = r
                        sel_task = j
            
            visit[sel_task] = 1
            decision[i] = tasks[sel_task]
            
            allocated_tasks_cnt += 1
            if allocated_tasks_cnt >= len(tasks):
                break
        return decision

    vcs.run(ALG_CCC)

    return vcs.get_stats()


if __name__ == '__main__':
    # for n_vehicle in [10, 20, 30, 40, 50]:
    #     a = RUN(n_vehicle, 300)
    for n_task in [100, 200, 300, 400, 500]:
        a = RUN(40, n_task)


# Vehicles:10, Tasks:300, City:(51,71), EP:214.555, TC:0.23, CPC:9.3285
# Vehicles:20, Tasks:300, City:(51,71), EP:133.9784, TC:0.3733, CPC:3.589
# Vehicles:30, Tasks:300, City:(51,71), EP:110.6022, TC:0.47, CPC:2.3532
# Vehicles:40, Tasks:300, City:(51,71), EP:99.1161, TC:0.4967, CPC:1.9955
# Vehicles:50, Tasks:300, City:(51,71), EP:114.5332, TC:0.6333, CPC:1.8085

# Vehicles:40, Tasks:100, City:(51,71), EP:106.3216, TC:1.0, CPC:1.0632
# Vehicles:40, Tasks:200, City:(51,71), EP:147.6674, TC:0.835, CPC:1.7685
# Vehicles:40, Tasks:300, City:(51,71), EP:108.6267, TC:0.54, CPC:2.0116
# Vehicles:40, Tasks:400, City:(51,71), EP:104.8095, TC:0.4375, CPC:2.3956
# Vehicles:40, Tasks:500, City:(51,71), EP:90.8859, TC:0.358, CPC:2.5387