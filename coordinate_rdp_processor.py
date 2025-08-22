from typing import List, Dict, Any
import math
import sys
import os
import pyproj
import json
import time


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x:.6f}, y={self.y:.6f})"


class RDPPointProcessor:
    # WGS84 좌표계 정의
    def __init__(self):
        self.crs_wgs84 = pyproj.CRS("EPSG:4326")


    # GPS 단순화
    # lat_lon_points : 프론트엔드에서 받은 GPS 좌표 리스트, epsilon : 허용 오차 (미터 단위)
    def process_gps_track(self, lat_lon_points: list[Point], epsilon: float) -> list[Point]:
        if not lat_lon_points or len(lat_lon_points) < 2:
            return lat_lon_points

        utm_points, utm_crs = self._convert_to_utm(lat_lon_points)
        simplified_utm_points = self._rdp_simplify(utm_points, epsilon)
        simplified_lat_lon_points = self._convert_to_wgs84(simplified_utm_points, utm_crs)

        return simplified_lat_lon_points
      
    # UTM 좌표계로 변환
    def _get_utm_crs(self, lat: float, lon: float) -> pyproj.CRS:
        # 경도를 이용해 UTM Zone 번호를 계산합니다. (1~60)
        zone_number = math.floor((lon + 180) / 6) + 1
        
        # 위도를 이용해 남반구인지 북반구인지 결정
        # 북반구는 326xx, 남반구는 327xx EPSG 코드를 사용합니다.
        epsg_base = 32600 if lat >= 0 else 32700
        
        # 최종 EPSG 코드를 조합하여 CRS 객체를 생성합니다.
        epsg_code = f"EPSG:{epsg_base + zone_number}"
        return pyproj.CRS(epsg_code)

    # 위경도 -> UTM
    def _convert_to_utm(self, lat_lon_points: list[Point]) -> tuple[list[Point], pyproj.CRS]:
        # UTM Zone 설정
        first_point = lat_lon_points[0]
        utm_crs = self._get_utm_crs(first_point.x, first_point.y)

        # WGS84 -> UTM 변환을 위한 Transformer 생성
        transformer = pyproj.Transformer.from_crs(self.crs_wgs84, utm_crs, always_xy=True)

        # 변환된 UTM 좌표 리스트 생성
        utm_points = []
        for point in lat_lon_points:
            utm_x, utm_y = transformer.transform(point.y, point.x)
            utm_points.append(Point(utm_x, utm_y))
        
        return utm_points, utm_crs


    # RDP 알고리즘을 사용하여 GPS 좌표 단순화
    def _rdp_simplify(self, points: list[Point], epsilon: float) -> list[Point]:
        if len(points) < 3:
            return points

        start_point = points[0]     # 시작 / 끝점
        end_point = points[-1]
        
        dmax = 0.0
        index = 0
        
        # 매 점을 순회하며 시작점과 끝점 사이의 최대 수직 거리 계산
        for i in range(1, len(points) - 1):
            d = self._perpendicular_distance(points[i], start_point, end_point)
            if d > dmax:
                dmax = d
                index = i

        # 최대 수직 거리가 허용 오차보다 크면 재귀
        if dmax > epsilon:
            rec_results1 = self._rdp_simplify(points[0:index+1], epsilon)
            rec_results2 = self._rdp_simplify(points[index:], epsilon)
            
            simplified_points = rec_results1[:-1] + rec_results2
            return simplified_points
        # 허용 오차 이내면 시작점과 끝점만 반환
        else:
            return [start_point, end_point]
            
            
    # 위경도 -> UTM -> WGS84
    def _convert_to_wgs84(self, utm_points: list[Point], utm_crs: pyproj.CRS) -> list[Point]:
        # UTM -> WGS84 역변환을 위한 Transformer 생성
        transformer = pyproj.Transformer.from_crs(utm_crs, self.crs_wgs84, always_xy=True)

        lat_lon_points = []
        for point in utm_points:
            lon, lat = transformer.transform(point.x, point.y)
            lat_lon_points.append(Point(lat, lon))
            
        return lat_lon_points

    
    # 수직 거리 계산
    def _perpendicular_distance(self, p: Point, start: Point, end: Point) -> float:
        dx = end.x - start.x
        dy = end.y - start.y
        
        if dx == 0 and dy == 0:
            return ((p.x - start.x)**2 + (p.y - start.y)**2)**0.5

        numerator = abs(dy * p.x - dx * p.y + end.x * start.y - end.y * start.x)
        denominator = (dx**2 + dy**2)**0.5
        
        return numerator / denominator


class JsonlIoProcessor:
        
    # 원본 러닝 데이터 업로드
    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
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

    # 보간 데이터 저장
    def save_data_to_jsonl(self, data: List[Any], file_path: str):
      try:
        with open(file_path, 'w', encoding='utf-8') as f:
          for record in data:
            # Point 객체일 경우 dict로 변환
            if isinstance(record, Point):
              json_string = json.dumps({'lat': record.x, 'lng': record.y}, ensure_ascii=False)
            else:
              json_string = json.dumps(record, ensure_ascii=False)
            f.write(json_string + '\n')
        print(f"\n[성공] 데이터가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
      except Exception as e:
        print(f"\n[오류] 파일 저장 중 오류가 발생했습니다: {e}", file=sys.stderr)
      

# Main
if __name__ == '__main__':
    
    # 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "result/smoothed_data.jsonl")
    output_file = os.path.join(script_dir, "result/smoothed_data.jsonl")

    # 러닝 데이터 로드
    jsonl_processor = JsonlIoProcessor()
    running_data = jsonl_processor.load_jsonl_data(input_file)

    # RDP 적용
    point_processor = RDPPointProcessor()
    
    # 코스 렌더링용 해상도 축소
    epsilon_in_meters = 1.0
    points = [Point(record['lat'], record['lng']) for record in running_data]
    print(f"처리할 데이터 점 개수: {len(running_data)}")
    start_time = time.time()
    simplified_track = point_processor.process_gps_track(points, epsilon_in_meters)
    end_time = time.time()
    print(f"단순화된 데이터 점 개수 (epsilon={epsilon_in_meters}m): {len(simplified_track)}")
    print(f"처리 시간: {end_time - start_time:.4f}초")
    
    # 보간한 데이터 저장
    jsonl_processor.save_data_to_jsonl(simplified_track, output_file)