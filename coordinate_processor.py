import time
import os, json, sys
import numpy as np
import heapq

from typing import List, Dict, Any
from pykalman import KalmanFilter
from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5179')

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


# 평균 보간법
def smooth_gps_with_average_time(records: list) -> List[Dict[str, Any]]:
    
    smoothed_records = []
    for i in range(2, len(records), 3):
        avg_lat = (records[i - 2]['lat'] + records[i - 1]['lat'] + records[i]['lat']) / 3
        avg_lng = (records[i - 2]['lng'] + records[i - 1]['lng'] + records[i]['lng']) / 3
        smoothed_records.append({
            'lat': avg_lat,
            'lng': avg_lng
        })
    return smoothed_records

def smooth_gps_with_average_distance(points, target_distance=3):
    
    utm_points = to_utm(points)
    
    filtered = [utm_points[0]]
    last_point = utm_points[0]
    for point in utm_points[1:]:
        distance = pythagoras_distance(
            last_point['y'], last_point['x'],
            point['y'], point['x']
        )
        if distance >= target_distance:
            filtered.append(point)
            last_point = point
    
    return filtered

def pythagoras_distance(lat1, lon1, lat2, lon2):
    return ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5


# UTM 좌표계로 변환
def to_utm(points):
    utm_points = []
    for i, p in enumerate(points):
        y, x = transformer.transform(p['lat'], p['lng'])
        utm_points.append({'x': x, 'y': y, 'original_index': i})
    return utm_points


# 비스발링감-와이엇 알고리즘
def simplify_with_visvalingam(points: List[Dict[str, Any]], threshold_area: int) -> List[Dict[str, Any]]:
    if len(points) < 3 or not transformer:
        return points

    utm_points = to_utm(points)

    while len(utm_points) > 2:
        min_area = float('inf')
        min_index = -1

        for i in range(1, len(utm_points) - 1):
            area = calculate_utm_area(
                utm_points[i - 1],
                utm_points[i],
                utm_points[i + 1]
            )
            if area < min_area:
                min_area = area
                min_index = i

        if min_area >= threshold_area:
            break
        
        if min_index != -1:
            utm_points.pop(min_index)
        else:
            break
    
    final_points = [points[p['original_index']] for p in utm_points]
    return final_points

def calculate_utm_area(p1: Dict, p2: Dict, p3: Dict) -> float:
    return 0.5 * abs(
        p1['y'] * (p2['x'] - p3['x']) +
        p2['y'] * (p3['x'] - p1['x']) +
        p3['y'] * (p1['x'] - p2['x'])
    )


# 비스발링감-와이엇 알고리즘, 최소힙 사용
def simplify_with_visvalingam_min_heap(points: List[Dict[str, Any]], threshold_area: int) -> List[Dict[str, Any]]:
    if len(points) < 3 or not transformer:
        return points

    utm_points = to_utm(points)

    heap = []           # 힙 구성
    for i in range(1, len(utm_points)-1):
        area = calculate_utm_area(
                utm_points[i - 1],
                utm_points[i],
                utm_points[i + 1]
        )
        heapq.heappush(heap, (area, i-1, i, i+1))
    
    is_active = [True] * len(utm_points)            # 각 점의 활성화 상태
    prev_nodes = [None] + [i-1 for i in range(1, len(utm_points))]          # 이전/다음 노드의 정보
    next_nodes = [i+1 for i in range(len(utm_points)-1)] + [None]
    
    while heap:
        area, start, mid, end = heapq.heappop(heap)
        if not is_active[mid] or prev_nodes[mid] != start or next_nodes[mid] != end:        # 비활성화 점, 변경된 점 스킵
            continue
        if area < threshold_area:
            is_active[mid] = False                  # 가운데 인덱스 비활성화, 주변 노드 정보 갱신
            next_nodes[start] = next_nodes[mid]
            prev_nodes[end] = prev_nodes[mid]
            if prev_nodes[start] is not None:       # 왼쪽 삼각형
                area1 = calculate_utm_area(
                    utm_points[prev_nodes[start]],
                    utm_points[start],
                    utm_points[end]
                )
                heapq.heappush(heap, (area1, prev_nodes[start], start, end))
            if next_nodes[end] is not None:         # 오른쪽 삼각형
                area2 = calculate_utm_area(
                    utm_points[start],
                    utm_points[end],
                    utm_points[next_nodes[end]]
                )
                heapq.heappush(heap, (area2, start, end, next_nodes[end]))
        else:
            break
    
    result = []         # 활성화되어 있는 점만 뽑아내기
    for idx in range(len(is_active)):
        if is_active[idx]:
            result.append(points[idx])
    return result

def is_all_active(is_active, start, mid, end):
    return is_active[start] and is_active[mid] and is_active[end]


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
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "dummy/data1.jsonl")
    output_file = os.path.join(script_dir, "result/smoothed_data.jsonl")

    # 러닝 데이터 업로드
    running_data = load_jsonl_data(input_file)
    
    # Kalman Filter를 활용하여 보간 및 저장
    kalman_data, original_measurements, smoothed_measurements = smooth_gps_with_kalman(running_data)
    save_data_to_jsonl(kalman_data, os.path.join(script_dir, "result/kalman_filtered_data.jsonl"))
    
    # 원본
    print("\n========== 원본 ==================\n")
    print("알고리즘 적용 전 : " + str(len(kalman_data)))
    
    # 평균 보간법
    print("\n========== 평균보간법 알고리즘 ==================\n")
    start_time = time.time()
    average_data = smooth_gps_with_average_time(kalman_data)
    average_data = smooth_gps_with_average_distance(average_data)
    print("평균보간법 적용 후 : " + str(len(average_data)))
    print(f"평균보간법 처리 시간: {time.time() - start_time:.4f}초")
    
    # 비스발링감-와이엇 알고리즘
    threshold_area = 2.0
    
    print("\n========== 기본 비스발링감-와이엇 알고리즘 ==================\n")
    start_time = time.time()
    vis_data = simplify_with_visvalingam(kalman_data, threshold_area)
    print("비스발링감-와이엇 알고리즘 적용 후 : " + str(len(vis_data)))
    print(f"비스발링감-와이엇 알고리즘 처리 시간: {time.time() - start_time:.4f}초")
    
    print("\n========== 최소힙 비스발링감-와이엇 알고리즘 ==================\n")
    start_time = time.time()
    vis_data_heap = simplify_with_visvalingam_min_heap(kalman_data, threshold_area)
    print("비스발링감-와이엇 알고리즘 적용 후 : " + str(len(vis_data_heap)))
    print(f"비스발링감-와이엇 알고리즘 처리 시간: {time.time() - start_time:.4f}초")
    
    
    # 보간한 데이터 저장
    save_data_to_jsonl(vis_data_heap, output_file)