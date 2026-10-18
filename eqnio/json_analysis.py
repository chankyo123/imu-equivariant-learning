"""Average the per-sequence NN metrics (ronin.ate / rpe / rmse_vel) of a main_net.py --mode test
metrics.json. Usage: python json_analysis.py <path/to/metrics.json>
Ported from Server B; the ledger of hard-coded paths it used to carry (e.g.
models/tlio_o2/test_again_TLIO_o2/metrics.json -> 5.511) is in results/ledger_comments of the handoff."""
import json
import sys

if len(sys.argv) < 2:
    sys.exit("usage: json_analysis.py <metrics.json>")
json_file = sys.argv[1]

print('json file : ', json_file)
with open(json_file, "r") as f:
    data = json.load(f)

# Initialize lists to store the ate and rpe_rmse_1000 values
ate_values = []
rmse_values = []
rmse_vel_values = []

for key, value in data.items():
    # if key in ("2100421890282669", "2142509999304237", "2142577795957193"):
    #     continue
    
    # filter_dict = value.get("filter")
    filter_dict = value.get("ronin")
    
    if filter_dict:
        ate_value = filter_dict.get("ate")
        # rmse_value = filter_dict.get("rpe_rmse_1000")
        # rmse_value = filter_dict.get("rmse")
        rmse_value = filter_dict.get("rpe")
        rmse_vel_value = filter_dict.get("rmse_vel")
        
        if ate_value is not None:
            ate_values.append(ate_value)
        
        if rmse_value is not None:
            rmse_values.append(rmse_value)
        
        if rmse_vel_value is not None:
            rmse_vel_values.append(rmse_vel_value)
        
        # Print the current ate, rmse, and rmse_vel values
        print(f"Key: {key}, ate value: {ate_value}, rmse value: {rmse_value}, rmse_vel value: {rmse_vel_value}")

# Calculate the average ate and rpe_rmse_1000 (if there are any values)
if ate_values:
    average_ate = sum(ate_values) / len(ate_values)
    print(f"Average ate value in 'filter': {average_ate}")
    print(f"Number of 'ate' values: {len(ate_values)}")
else:
    print("No 'ate' values found in 'filter' key")

if rmse_values:
    average_rmse = sum(rmse_values) / len(rmse_values)
    print(f"Average rmse value in 'filter': {average_rmse}")
    print(f"Number of 'rmse' values: {len(rmse_values)}")
else:
    print("No 'rmse' values found in 'filter' key")
    
if rmse_vel_values:
    average_rmse_vel = sum(rmse_vel_values) / len(rmse_vel_values)
    print(f"Average rmse_vel value in 'filter': {average_rmse_vel}")
    print(f"Number of 'rmse_vel' values: {len(rmse_vel_values)}")
else:
    print("No 'rmse_vel' values found in 'filter' key")
