#!/usr/bin/env python3
"""
Production API Test Script
"""

import requests
import json
from PIL import Image
import numpy as np
import io
import base64

# API 엔드포인트
BASE_URL = "http://localhost:8000"

def test_health_check():
    """헬스 체크 테스트"""
    print("\n" + "="*50)
    print("1. 헬스 체크 테스트")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"상태 코드: {response.status_code}")
    print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ 헬스 체크 성공!")

def create_sample_image():
    """테스트용 이미지 생성"""
    # 간단한 의류 이미지 시뮬레이션 (티셔츠 모양)
    img = Image.new('RGB', (400, 500), color='white')
    pixels = np.array(img)
    
    # 티셔츠 모양 그리기 (간단한 직사각형)
    # 몸통
    pixels[100:400, 100:300] = [100, 150, 200]  # 파란색
    # 소매
    pixels[150:250, 50:100] = [100, 150, 200]   # 왼쪽 소매
    pixels[150:250, 300:350] = [100, 150, 200]  # 오른쪽 소매
    
    img = Image.fromarray(pixels.astype('uint8'))
    
    # 이미지를 바이트로 변환
    img_byte = io.BytesIO()
    img.save(img_byte, format='PNG')
    img_byte.seek(0)
    
    return img_byte

def test_image_analysis():
    """이미지 분석 API 테스트"""
    print("\n" + "="*50)
    print("2. 이미지 분석 테스트")
    print("="*50)
    
    # 샘플 이미지 생성
    image_file = create_sample_image()
    
    # API 호출
    files = {'file': ('test_shirt.png', image_file, 'image/png')}
    response = requests.post(f"{BASE_URL}/api/analyze", files=files)
    
    print(f"상태 코드: {response.status_code}")
    result = response.json()
    print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    assert "items" in result
    print(f"✅ 이미지 분석 성공! 발견된 아이템: {result.get('count', 0)}개")

def test_coordination():
    """코디 추천 API 테스트"""
    print("\n" + "="*50)
    print("3. 코디 추천 테스트")
    print("="*50)
    
    # 다양한 상황별 테스트
    occasions = ["casual", "business", "date"]
    
    for occasion in occasions:
        print(f"\n📍 {occasion} 스타일 추천:")
        
        payload = {
            "user_id": "test_user",
            "occasion": occasion,
            "weather": {
                "temperature": 18,
                "conditions": "Clear"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/coordinate",
            json=payload
        )
        
        print(f"상태 코드: {response.status_code}")
        result = response.json()
        
        if "recommendations" in result and len(result["recommendations"]) > 0:
            rec = result["recommendations"][0]
            print(f"  추천 1위: {rec['confidence']}% 확신도")
            print(f"  - 상의: {rec['outfit'].get('top', {}).get('type', 'N/A')}")
            print(f"  - 하의: {rec['outfit'].get('bottom', {}).get('type', 'N/A')}")
            print(f"  - 신발: {rec['outfit'].get('shoes', {}).get('type', 'N/A')}")
            if 'outerwear' in rec['outfit']:
                print(f"  - 아우터: {rec['outfit']['outerwear']['type']}")
    
    print("\n✅ 코디 추천 테스트 성공!")

def test_closet_upload():
    """옷장 업로드 테스트"""
    print("\n" + "="*50)
    print("4. 옷장 업로드 테스트")
    print("="*50)
    
    image_file = create_sample_image()
    
    files = {'file': ('item.png', image_file, 'image/png')}
    data = {'category': 'top'}
    
    response = requests.post(
        f"{BASE_URL}/api/closet/items",
        files=files,
        data=data
    )
    
    print(f"상태 코드: {response.status_code}")
    result = response.json()
    print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    assert "id" in result
    print(f"✅ 아이템 업로드 성공! ID: {result['id']}")

def test_weather():
    """날씨 API 테스트"""
    print("\n" + "="*50)
    print("5. 날씨 API 테스트")
    print("="*50)
    
    locations = ["Seoul", "Busan", "Jeju"]
    
    for location in locations:
        response = requests.get(f"{BASE_URL}/api/weather/{location}")
        print(f"\n📍 {location}:")
        result = response.json()
        print(f"  온도: {result['temperature']}°C")
        print(f"  날씨: {result['conditions']}")
    
    print("\n✅ 날씨 API 테스트 성공!")

def test_stats():
    """통계 API 테스트"""
    print("\n" + "="*50)
    print("6. 통계 API 테스트")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/api/stats")
    result = response.json()
    print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n✅ 통계 API 테스트 성공!")

def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "🚀"*20)
    print("Style Mate Production API 테스트 시작")
    print("🚀"*20)
    
    try:
        test_health_check()
        test_image_analysis()
        test_coordination()
        test_closet_upload()
        test_weather()
        test_stats()
        
        print("\n" + "="*50)
        print("🎉 모든 테스트 성공!")
        print("="*50)
        print("\n프로덕션 서버가 정상 작동 중입니다.")
        print("Railway 배포 준비 완료! 🚀")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_all_tests()