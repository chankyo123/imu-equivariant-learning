import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_data.txt' with the actual path to your text data file
data = pd.read_csv('./batch_test_outputs/models-eq_4res_2.0/145820422949970/net_outputs.txt')

# Extract time and last three columns
time = data.iloc[:, 0]
last_columns_data = data.iloc[:, 1:4]

# Plot the data
fig, axs = plt.subplots(3, 3, figsize=(8, 8), sharex=True)

# Plot against the last three columns
for i in range(3):
    axs[i,0].plot(time, last_columns_data.iloc[:, i])
    axs[i,0].set_title(f'Column {data.shape[1] - 3 + i}')
    axs[i,0].set_ylabel(f'Data in Column {data.shape[1] - 3 + i}')
    axs[i,0].grid(True)

# Set common x-axis label
# axs[-1].set_xlabel('Time')

# Adjust layout for better readability
# plt.tight_layout()
# plt.show()

data2 = pd.read_csv('./batch_test_outputs/models-eq/145820422949970/net_outputs.txt')

# Extract time and last three columns
time2 = data2.iloc[:, 0]
last_columns_data2 = data2.iloc[:, 1:4]

# Plot the data
# fig2, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Plot against the last three columns
for i in range(3):
    axs[i,1].plot(time2, last_columns_data2.iloc[:, i])
    axs[i,1].set_title(f'Column {data.shape[1] - 3 + i}')
    axs[i,1].set_ylabel(f'Data in Column {data.shape[1] - 3 + i}')
    axs[i,1].grid(True)

data3 = pd.read_csv('./batch_test_outputs/models-resnet/145820422949970/net_outputs.txt')

# Extract time and last three columns
time3 = data3.iloc[:, 0]
last_columns_data3 = data3.iloc[:, 1:4]

# Plot the data
# fig2, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Plot against the last three columns
for i in range(3):
    axs[i,2].plot(time3, last_columns_data3.iloc[:, i])
    axs[i,2].set_title(f'Column {data.shape[1] - 3 + i}')
    axs[i,2].set_ylabel(f'Data in Column {data.shape[1] - 3 + i}')
    axs[i,2].grid(True)
    
    
# Set common x-axis label
# axs[-1].set_xlabel('Time')

# Adjust layout for better readability
plt.tight_layout()
plt.show()