import os
import matplotlib.pyplot as plt
import json
import sys
from typing import List, Dict, Any
from pykalman import KalmanFilter
import numpy as np

# Kalman Filter 알고리즘
def smooth_gps_with_kalman(records: list) -> tuple:
    measurements = np.array([[r['lat'], r['lng']] for r in records])
    transition_matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    observation_matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]
    initial_state_mean = [measurements[0, 0], measurements[0, 1], 0, 0]

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        observation_covariance=np.eye(2) * 1e-4,          # GPS 측정 오차
        transition_covariance=np.eye(4) * 1e-8            # 상태 전이(예측) 오차
    )
    
    smoothed_states_means, _ = kf.smooth(measurements)
    smoothed_records = []
    for i, r in enumerate(records):
        new_record = r.copy()
        new_record['lat'] = smoothed_states_means[i, 0]
        new_record['lng'] = smoothed_states_means[i, 1]
        smoothed_records.append(new_record)
    
    return smoothed_records, measurements, smoothed_states_means

# 원본 러닝 데이터 업로드
def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"[경고] {line_num}번째 줄에서 JSON 파싱 오류: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"[오류] 파일 처리 중 예기치 않은 오류가 발생했습니다: {e}", file=sys.stderr)
    return records

# 보간한 데이터 저장
def save_data_to_jsonl(data: List[Dict[str, Any]], file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in data:
                json_string = json.dumps(record, ensure_ascii=False)
                f.write(json_string + '\n')
        print(f"\n[성공] 데이터가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"\n[오류] 파일 저장 중 오류가 발생했습니다: {e}", file=sys.stderr)

# Main
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "dummy/data2.jsonl")
    output_file = os.path.join(script_dir, "result/smoothed_data.jsonl")

    # 러닝 데이터 업로드
    running_data = load_jsonl_data(input_file)
    
    # Kalman Filter를 활용하여 보간
    smoothed_data, original_measurements, smoothed_measurements = smooth_gps_with_kalman(running_data)
    
    # 보간한 데이터 저장
    save_data_to_jsonl(smoothed_data, output_file)