#!/usr/bin/env python3
"""StyleMate API 테스트 스크립트"""

import requests
import json
import os
from typing import Dict, Any

class StyleMateAPITester:
    """StyleMate API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.auth_token = None
    
    def test_health_check(self) -> bool:
        """헬스 체크 테스트"""
        try:
            response = self.session.get(f"{self.base_url}/api/health/")
            print(f"🔍 Health Check: {response.status_code}")
            if response.status_code == 200:
                print(f"✅ Response: {response.json()}")
                return True
            else:
                print(f"❌ Failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_product_search(self) -> bool:
        """상품 검색 API 테스트"""
        try:
            # 기본 검색
            params = {
                'q': '티셔츠',
                'limit': 5
            }
            response = self.session.get(f"{self.base_url}/api/search/", params=params)
            print(f"🔍 Product Search: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {data.get('pagination', {}).get('total', 0)} products")
                print(f"   Engine: {data.get('search_info', {}).get('engine', 'unknown')}")
                return True
            else:
                print(f"❌ Search failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Search test failed: {e}")
            return False
    
    def test_trending_products(self) -> bool:
        """인기 상품 API 테스트"""
        try:
            response = self.session.get(f"{self.base_url}/api/trending/")
            print(f"🔍 Trending Products: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                trending_count = len(data.get('trending_products', []))
                print(f"✅ Found {trending_count} trending products")
                return True
            else:
                print(f"❌ Trending failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Trending test failed: {e}")
            return False
    
    def test_search_suggestions(self) -> bool:
        """검색 자동완성 API 테스트"""
        try:
            params = {'q': '나이키'}
            response = self.session.get(f"{self.base_url}/api/search/suggestions/", params=params)
            print(f"🔍 Search Suggestions: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                suggestions_count = len(data.get('suggestions', []))
                print(f"✅ Found {suggestions_count} suggestions")
                return True
            else:
                print(f"❌ Suggestions failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Suggestions test failed: {e}")
            return False
    
    def test_ai_trend_analysis(self) -> bool:
        """AI 트렌드 분석 API 테스트"""
        try:
            response = self.session.get(f"{self.base_url}/api/ai/trend-analysis/")
            print(f"🔍 AI Trend Analysis: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Season: {data.get('season', 'unknown')}")
                print(f"   Trending colors: {len(data.get('trending_colors', []))}")
                print(f"   Trending styles: {len(data.get('trending_styles', []))}")
                return True
            else:
                print(f"❌ AI trend failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ AI trend test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        print("🚀 StyleMate API 테스트 시작!")
        print("=" * 50)
        
        tests = {
            'health_check': self.test_health_check,
            'product_search': self.test_product_search,
            'trending_products': self.test_trending_products,
            'search_suggestions': self.test_search_suggestions,
            'ai_trend_analysis': self.test_ai_trend_analysis,
        }
        
        results = {}
        for test_name, test_func in tests.items():
            print(f"\n📋 {test_name.replace('_', ' ').title()} 테스트...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
            
            if results[test_name]:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("📊 테스트 결과 요약:")
        passed = sum(results.values())
        total = len(results)
        print(f"   통과: {passed}/{total}")
        
        if passed == total:
            print("🎉 모든 API 테스트 통과! 프론트엔드 연동 준비 완료!")
        else:
            print("⚠️ 일부 테스트 실패. API 확인 필요.")
        
        return results

def main():
    """메인 실행 함수"""
    tester = StyleMateAPITester()
    results = tester.run_all_tests()
    
    # 실패한 테스트가 있으면 종료 코드 1 반환
    if not all(results.values()):
        exit(1)

if __name__ == "__main__":
    main()