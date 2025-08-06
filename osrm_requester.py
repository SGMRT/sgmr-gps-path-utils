import sys
import json
import argparse
import requests

# ORSM 요청
def request_osrm_match(input_path: str,
                       host: str = "localhost:5050",
                       profile: str = "foot"):
                       
    coords = []
    timestamps = []

    # 파일 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 파라미터 검증 -> 위경도 및 타임스탬프 추가
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] line {line_num} JSON error: {e}", file=sys.stderr)
                continue

            if 'lng' not in obj or 'lat' not in obj or 'timeStamp' not in obj:
                print(f"[Warning] line {line_num} missing required fields.", file=sys.stderr)
                continue

            coords.append(f"{obj['lng']},{obj['lat']}")
            timestamps.append(str(obj['timeStamp']))

    # URL 구성
    coord_str = ";".join(coords)
    ts_str = ";".join(timestamps)
    radis_list = [20] * len(coords)
    radis_str = ";".join(map(str, radis_list))
    url = f"http://{host}/match/v1/{profile}/{coord_str}?steps=true&radiuses={radis_str}"

    # OSRM 요청
    try:
        # 응답
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        result_json = response.json()

        # 저장
        output_path = "result/osrm_result.json"
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(result_json, out_f, ensure_ascii=False, indent=2)
    
    # 예외 처리
    except requests.exceptions.HTTPError as e:
        print(f"\n[Error] OSRM API Error: {e}", file=sys.stderr)
        print(f"Response from server: {response.text}", file=sys.stderr)
    except requests.exceptions.RequestException as e:
        print(f"\n[Error] HTTP Request Failed: {e}", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"\n[Error] Failed to decode JSON response from server.", file=sys.stderr)
        print(f"Raw response: {response.text}", file=sys.stderr)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts lng, lat, timeStamp from a .jsonl file and calls the OSRM match API."
    )
    parser.add_argument(
        'input_file',
        help="Input file with one JSON object per line (e.g., data.jsonl)"
    )
    parser.add_argument(
        '--host',
        default='localhost:5050',
        help="OSRM server address and port (default: localhost:5050)"
    )
    parser.add_argument(
        '--profile',
        default='foot',
        help="Profile for the match API (default: foot)"
    )
    args = parser.parse_args()
    request_osrm_match(args.input_file, args.host, args.profile)
