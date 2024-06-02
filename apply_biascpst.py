import numpy as np
import json
import os
import os.path as osp
import csv

class SensorCalibrator:
    @classmethod
    def from_offline_calib(cls, dataset, args):
        ret = cls()
        print(f"loading offline calib from {osp.join(args.root_dir, dataset, 'calibration.json')}")
        with open(osp.join(args.root_dir, dataset, 'calibration.json'), 'r') as f:
            calib_json = json.load(f)
            
        ret.accelBias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"])[:, None]
        ret.gyroBias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"])[:, None]
        ret.accelScaleInv = np.linalg.inv(np.array(
            calib_json["Accelerometer"]["Model"]["RectificationMatrix"]
        ))
        ret.gyroScaleInv = np.linalg.inv(np.array(
            calib_json["Gyroscope"]["Model"]["RectificationMatrix"]
        ))
        return ret

    def calibrate_raw(self, acc, gyr):
        assert len(acc.shape) == 2
        N = acc.shape[1]
        assert acc.shape == (3, N)
        assert gyr.shape == (3, N)
        acc_cal = self.accelScaleInv @ acc - self.accelBias
        gyr_cal = self.gyroScaleInv @ gyr - self.gyroBias
        assert acc_cal.shape == (3, N)
        assert gyr_cal.shape == (3, N)

        return acc_cal, gyr_cal

def apply_calibration_to_files(root_dir, args):
    # Recursively find all subdirectories under root_dir
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for dataset in subdirs:
        dataset_dir = osp.join(root_dir, dataset)
        csv_file = osp.join(dataset_dir, 'imu_samples_0.csv')
        
        if not osp.exists(csv_file):
            print(f"imu_samples_0.csv not found in {dataset_dir}")
            continue
        
        # Load the calibrator
        calibrator = SensorCalibrator.from_offline_calib(dataset, args)
        
        # Read the raw data from the CSV file
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                raw_data = []
                for row in reader:
                    try:
                        raw_data.append([float(item) for item in row])
                    except ValueError:
                        pass  # Skip rows with non-numeric values

            raw_data = np.array(raw_data)
            
            # Extract time, temperature, gyro and acc data from raw_data
            time = raw_data[:, 0]
            temperature = raw_data[:, 1]
            gyro = raw_data[:, 2:5].T  # Shape (3, N)
            acc = raw_data[:, 5:8].T  # Shape (3, N)
            
            # Apply calibration
            acc_cal, gyro_cal = calibrator.calibrate_raw(acc, gyro)
            
            # Prepare the calibrated data for saving
            calibrated_data = np.hstack((time[:, None], temperature[:, None], gyro_cal.T, acc_cal.T))
            
            # Save the calibrated data to a new CSV file
            output_csv = osp.join(dataset_dir, 'imu_samples_calibrated.csv')
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(calibrated_data)
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            
        print(output_csv)

class Args:
    def __init__(self, root_dir):
        self.root_dir = root_dir

# Example usage
args = Args(root_dir="./local_data/tlio_golden")
root_dir = args.root_dir

apply_calibration_to_files(root_dir, args)
