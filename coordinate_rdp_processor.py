import numpy as np
import math
import sys
import os
import json
import time

from pykalman import KalmanFilter
from typing import List, Dict, Any
from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5179')

class Point:
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = idx

    def __repr__(self):
        return f"Point(x={self.x:.6f}, y={self.y:.6f}, idx={self.idx})"


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
        observation_covariance=np.eye(2) * 1e-5,          # GPS 측정 오차
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


# UTM 좌표계로 변환
def to_utm(points):
    utm_points = []
    for i, p in enumerate(points):
        y, x = transformer.transform(p['lat'], p['lng'])
        utm_points.append({'x': x, 'y': y, 'original_index': i})
    return utm_points


# RDP 알고리즘
def rdp(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) < 3:
        return points

    start_point, end_point = points[0], points[-1]     # 시작 / 끝점
    dmax, index = 0.0, 0
    
    for i in range(1, len(points) - 1):       # 점 - 선분 수직 거리 최대값 탐색
        d = perpendicular_distance(points[i], start_point, end_point)
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:      # 허용 오차보다 크면 재귀
        rec_results1 = rdp(points[0:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)
        simplified_points = rec_results1[:-1] + rec_results2
        return simplified_points
    else:
        return [start_point, end_point]
    
def perpendicular_distance(p: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    
    if dx == 0 and dy == 0:
        return ((p.x - start.x)**2 + (p.y - start.y)**2)**0.5

    numerator = abs(dy * p.x - dx * p.y + end.x * start.y - end.y * start.x)
    denominator = (dx**2 + dy**2)**0.5
    
    return numerator / denominator    
      
      
# 데이터 로드/저장
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
if __name__ == '__main__':
    
    # 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "dummy/data1.jsonl")
    output_file = os.path.join(script_dir, "result/smoothed_data.jsonl")

    # 러닝 데이터 로드
    running_data = load_jsonl_data(input_file)
    
    # Kalman Filter를 활용하여 보간 및 저장
    kalman_data, original_measurements, smoothed_measurements = smooth_gps_with_kalman(running_data)
    save_data_to_jsonl(kalman_data, os.path.join(script_dir, "result/kalman_filtered_data.jsonl"))
    
    # UTM 변환
    utm_kalman_data = to_utm(kalman_data)
    
    # Point 변환
    points = []
    for i, record in enumerate(utm_kalman_data):
        points.append(Point(record['x'], record['y'], i))
    
    # 원본
    print("\n========== 원본 ==================\n")
    print("알고리즘 적용 전 : " + str(len(utm_kalman_data)))
    
    # RPD 알고리즘 적용
    print("\n========== RDP 알고리즘 ==================\n")
    start_time = time.time()
    simplified_track = rdp(points, 8.0)
    end_time = time.time()
    print(f"단순화된 데이터 점 개수): {len(simplified_track)}")
    print(f"처리 시간: {end_time - start_time:.4f}초")
    
    # 보간한 데이터 저장
    result = []
    for point in simplified_track:
        idx = point.idx
        result.append(kalman_data[idx])
    save_data_to_jsonl(result, output_file)