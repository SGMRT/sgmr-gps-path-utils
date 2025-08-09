import json
import boto3
import requests
from rdp_module import RDPPointProcessor
from rdp_module import Point

course_directory_name = "course-telemetry"

s3_client = boto3.client('s3')
rdp_processor = RDPPointProcessor()

def lambda_handler(event, context):
  
    payload = event
    
    run_id = payload['runId']
    running_directory_name = payload['directoryName']
    member_id = payload['memberId']
    file_name = payload['fileName']
    
    S3_BUCKET = payload['bucketName']
    RUNNING_TELEMETRY_OBJECT_KEY = running_directory_name + "/" + member_id + "/" + file_name
    COURSE_TELEMETRY_OBJECT_KEY = course_directory_name + "/" + member_id + "/" + file_name
    
    callback_url = payload['callbackUrl']

    print(f"Received payload: {payload}")
    print(f"S3 버킷명: {S3_BUCKET}")
    print(f"러닝 시계열 Object 키: {RUNNING_TELEMETRY_OBJECT_KEY}")
    print(f"코스 시계열 Object 키: {COURSE_TELEMETRY_OBJECT_KEY}")
    print(f"콜백 URL: {callback_url}")
    
    try:
        # S3에서 다운로드
        print(f"Processing Run ID: {run_id}")
        print(f"Downloading raw data from s3://{S3_BUCKET}/{RUNNING_TELEMETRY_OBJECT_KEY}")
        telemetry_data_content = s3_client.get_object(Bucket=S3_BUCKET, Key=RUNNING_TELEMETRY_OBJECT_KEY)['Body'].read().decode('utf-8-sig')
        
        # 파싱
        lines = telemetry_data_content.strip().splitlines()
        records = [json.loads(line) for line in lines if line]
              
        # RDP 오차 및 Points 설정
        epsilon_meters = 8.0
        points = [Point(record['lat'], record['lng']) for record in records]

        # RDP 적용
        print(f"Generating course data with RDP (epsilon={epsilon_meters}m)")
        course_points_list = rdp_processor.process_gps_track(points, epsilon_meters)

        # 업로드
        final_jsonl_body = '\n'.join(json.dumps(p) for p in course_points_list)
        s3_client.put_object(Bucket=S3_BUCKET, Key=COURSE_TELEMETRY_OBJECT_KEY, Body=final_jsonl_body)
        print(f"Successfully uploaded course data to s3://{S3_BUCKET}/{COURSE_TELEMETRY_OBJECT_KEY}")

        # 성공 콜백
        callback_payload = {
            "runId": run_id,
            "courseDataUrl": f"s3://{S3_BUCKET}/{COURSE_TELEMETRY_OBJECT_KEY}"
        }
        print(f"Calling success callback to Spring: {callback_url}")
        requests.post(callback_url, json=callback_payload)
        return {'statusCode': 200, 'body': json.dumps('Processing completed successfully!')}

    except Exception as e:
        print(f"Error processing runId {run_id}: {e}")
        
        # 실패 콜백
        callback_payload = {"runId": run_id, "status": "FAILED"}
        print(f"Calling failure callback to Spring: {callback_url}")
        requests.post(callback_url, json=callback_payload)
        
        # Lambda 함수 자체도 실패했음을 AWS에 알림 (CloudWatch 및 DLQ 연동을 위함)
        raise e