import torch
import os

dir = os.path.dirname(os.path.realpath(__file__))

traj = torch.load(os.path.join(dir, "..", "expert_datasets/ppo_walker_25.pt"))
print(traj["obs"].shape)
print(traj["actions"].shape)
print(traj["done"].sum())

target_list = [1, 2, 3, 5]
res_trajs = {}
for target in target_list:
    res_trajs[target] = {}

for (k, v) in traj.items():
    trajs = v.split(1000, dim=0)
    for target in target_list:
        res_trajs[target][k] = torch.cat(trajs[:target], dim=0)
    
for target, trajs in res_trajs.items():
    print(target)
    torch.save(trajs, os.path.join(dir, "..", f'expert_datasets/ppo_walker_{target}.pt'))
