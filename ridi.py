import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/ridi-robust-imu-double-integration")

print("Path to dataset files:", path)