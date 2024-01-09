import numpy as np

from env.vcs import VCS


def RUN(n_vehicle, n_task):

    vcs = VCS(n_vehicle, n_task, alg="AAA")

    def get_AAA_target(vehicle_id, task_id, sensed_time) -> float:
        return vcs.get_DC(vehicle_id, task_id)

    def ALG_AAA(vehicles, tasks, time_slice) -> list[int]:
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
                    r = get_AAA_target(vehicles[i], tasks[j], time_slice)
                    if r > max_reward:
                        max_reward = r
                        sel_task = j
            
            visit[sel_task] = 1
            decision[i] = tasks[sel_task]
            
            allocated_tasks_cnt += 1
            if allocated_tasks_cnt >= len(tasks):
                break
        return decision

    vcs.run(ALG_AAA)

    return vcs.get_stats()


if __name__ == '__main__':
    # for n_vehicle in [10, 20, 30, 40, 50]:
    #     a = RUN(n_vehicle, 300)
    for n_task in [100, 200, 300, 400, 500]:
        a = RUN(40, n_task)


# Vehicles:10, Tasks:300, City:(51,71), EP:511.4602, TC:0.1933, CPC:26.4594
# Vehicles:20, Tasks:300, City:(51,71), EP:441.6481, TC:0.3367, CPC:13.117
# Vehicles:30, Tasks:300, City:(51,71), EP:389.8406, TC:0.41, CPC:9.5083
# Vehicles:40, Tasks:300, City:(51,71), EP:272.8193, TC:0.45, CPC:6.0627
# Vehicles:50, Tasks:300, City:(51,71), EP:261.631, TC:0.6067, CPC:4.3124

# Vehicles:40, Tasks:100, City:(51,71), EP:143.0748, TC:1.0, CPC:1.4307
# Vehicles:40, Tasks:200, City:(51,71), EP:277.3276, TC:0.755, CPC:3.6732
# Vehicles:40, Tasks:300, City:(51,71), EP:304.692, TC:0.4867, CPC:6.2604
# Vehicles:40, Tasks:400, City:(51,71), EP:289.0197, TC:0.355, CPC:8.1414
# Vehicles:40, Tasks:500, City:(51,71), EP:291.5898, TC:0.276, CPC:10.5648