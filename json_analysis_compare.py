import json

# Define the paths to the JSON files
json_file1 = "batch_filter_outputs_uf10/eq_2res_epmore2_N_1_uf10/eq_2res_200hz_3input_ep_more2/metrics.json"
json_file2 = "../TLIO/batch_filter_outputs_uf20/models-resnet/metrics.json" 

# Read and parse the JSON data from both files
with open(json_file1, "r") as f1, open(json_file2, "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Function to extract ate and rpe_rmse_1000 values
def extract_metrics(data):
    results = {}
    for key, value in data.items():
        sequence_name = key
        filter_dict = value.get("filter")
        if filter_dict:
            ate_value = filter_dict.get("ate")
            rpe_rmse_1000_value = filter_dict.get("rpe_rmse_100000")
            if ate_value is not None and rpe_rmse_1000_value is not None:
                results[sequence_name] = (ate_value, rpe_rmse_1000_value)
    
    return results

# Extract metrics from both JSON files
results1 = extract_metrics(data1)
results2 = extract_metrics(data2)

# Get sorted sequence names
sequence_names = sorted(set(results1.keys()).intersection(set(results2.keys())))

# Print values in one row
print(f"JSON File 1: {json_file1}")
for sequence_name in sequence_names:
    if sequence_name not in ("263070267597787", "286510035288905", "1014753008676428", "1718648064889757"):
        ate1, rpe1 = results1[sequence_name]
        ate2, rpe2 = results2[sequence_name]
        ate_diff = ate1 - ate2
        rpe_diff = rpe1 - rpe2
        print(f"Sequence Name: {sequence_name}")
        print(f"Ate1: {ate1}, Ate2: {ate2}, Difference in Ate: {ate_diff}")
        print(f"RPE_RMSE_1000_1: {rpe1}, RPE_RMSE_1000_2: {rpe2}, Difference in RPE_RMSE_1000: {rpe_diff}")
        print()

# Calculate and print differences
ate_differences = [results1[name][0] - results2[name][0] for name in sequence_names]
rpe_rmse_1000_differences = [results1[name][1] - results2[name][1] for name in sequence_names]
print(f"Differences in ate values: {ate_differences}")
print(f"Differences in rpe_rmse_1000 values: {rpe_rmse_1000_differences}")