import argparse
import json
import xml.etree.ElementTree as ET
import sys
import time
from datetime import datetime


def parse_gpx(input_file: str) -> list[dict]:
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        raise Exception(f"GPX 파일 파싱 오류: {e}")
    except FileNotFoundError:
        raise Exception(f"파일을 찾을 수 없습니다: {input_file}")
    
    # GPX 네임스페이스 처리
    namespaces = {}
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0] + '}'
        namespaces['gpx'] = namespace[1:-1]  # { } 제거
        trkpt_tag = f"{namespace}trk/{namespace}trkseg/{namespace}trkpt"
    else:
        trkpt_tag = "trk/trkseg/trkpt"
    
    points = []
    
    # 모든 trkpt 요소 찾기
    if namespaces:
        trkpts = root.findall('.//gpx:trkpt', namespaces)
    else:
        trkpts = root.findall('.//trkpt')
    
    if not trkpts:
        raise Exception("GPX 파일에서 트랙포인트를 찾을 수 없습니다")
    
    for trkpt in trkpts:
        try:
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            
            # timestamp 추출
            timestamp = None
            if namespaces:
                time_elem = trkpt.find('gpx:time', namespaces)
            else:
                time_elem = trkpt.find('time')
            
            if time_elem is not None and time_elem.text:
                try:
                    # ISO 8601 형식 -> Unix timestamp (milliseconds)로 변환
                    dt = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                    timestamp = int(dt.timestamp() * 1000)
                except ValueError:
                    # 시간 파싱 실패시 현재 시간 사용
                    timestamp = int(time.time() * 1000)
            else:
                # 시간 정보가 없으면 현재 시간 사용
                timestamp = int(time.time() * 1000)
            
            points.append({"lat": lat, "lng": lon, "ts": timestamp})
        except (ValueError, TypeError) as e:
            raise Exception(f"좌표 변환 오류: lat={trkpt.get('lat')}, lon={trkpt.get('lon')}")
    
    return points


def write_jsonl(points: list[dict], output_file: str):
    """JSONL 형식으로 출력"""
    if output_file:
        # 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for pt in points:
                f.write(json.dumps(pt, ensure_ascii=False, separators=(',', ':')) + '\n')
    else:
        # 표준출력
        for pt in points:
            print(json.dumps(pt, ensure_ascii=False, separators=(',', ':')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPX 파일을 JSONL 형식으로 변환")
    parser.add_argument("input", help="입력 GPX 파일 경로")
    parser.add_argument("-o", "--output", help="출력 JSONL 파일 경로 (미지정시 표준출력)")
    
    args = parser.parse_args()
    
    try:
        points = parse_gpx(args.input)
        write_jsonl(points, args.output)
        
        if args.output:
            print(f"변환 완료: {len(points)}개 점을 {args.output}에 저장했습니다.", file=sys.stderr)
        else:
            print(f"변환 완료: {len(points)}개 점을 출력했습니다.", file=sys.stderr)
            
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)