import torch
import matplotlib.pyplot as plt

data = torch.load('sine_2000.pt')
states = data['obs']
actions = data['actions']

plt.figure(figsize=(8, 6))
plt.scatter(states, actions, s=10)  # s controls marker size
plt.title('(State, Action) Pairs')
plt.xlabel('State')
plt.ylabel('Action')
plt.grid(True)
plt.axis('equal')
plt.savefig('sine_2000.png')

# import matplotlib.pyplot as plt
# import numpy as np

# # Create sample data and replace it with the real data when needed.
# data = np.random.rand(10, 10)  # 10x10 sample data
# min_x = 0
# max_x = 10
# min_y = 0
# max_y = 10

# plt.imshow(data, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='viridis', aspect='auto')
# plt.colorbar(label='Frequency')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Density of Division')

# # Save the figure.
# plt.savefig('division_plot.png')
# plt.show()
