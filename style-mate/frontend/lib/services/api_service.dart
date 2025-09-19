import 'dart:io';
import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

class ApiService {
  late Dio _dio;
  static const String baseUrl = 'https://money-production-452c.up.railway.app/api';
  // For local development: 'http://localhost:8000/api'
  
  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 30),
      headers: {
        'Content-Type': 'application/json',
      },
    ));
    
    // Add logger in debug mode
    if (const bool.fromEnvironment('dart.vm.product') == false) {
      _dio.interceptors.add(PrettyDioLogger(
        requestHeader: true,
        requestBody: true,
        responseBody: true,
        responseHeader: false,
        error: true,
        compact: true,
        maxWidth: 90,
      ));
    }
  }

  /// Analyze clothing image
  Future<Map<String, dynamic>> analyzeImage(File imageFile) async {
    try {
      FormData formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          imageFile.path,
          filename: 'clothing.jpg',
        ),
      });
      
      final response = await _dio.post('/analyze', data: formData);
      return response.data;
    } catch (e) {
      throw _handleError(e);
    }
  }

  /// Get outfit recommendations
  Future<List<Map<String, dynamic>>> getRecommendations({
    required String occasion,
    Map<String, dynamic>? weather,
    String? userId,
  }) async {
    try {
      final response = await _dio.post('/coordinate', data: {
        'user_id': userId ?? 'default_user',
        'occasion': occasion,
        'weather_override': weather,
      });
      
      if (response.data['recommendations'] != null) {
        return List<Map<String, dynamic>>.from(response.data['recommendations']);
      }
      return [];
    } catch (e) {
      // Return mock data if API fails
      return _getMockRecommendations();
    }
  }

  /// Get weather data
  Future<Map<String, dynamic>> getWeather(String location) async {
    try {
      final response = await _dio.get('/weather/$location');
      return response.data;
    } catch (e) {
      // Return mock weather if fails
      return {
        'temperature': 22,
        'conditions': 'Clear',
        'humidity': 60,
        'location': location,
      };
    }
  }

  /// Upload clothing item
  Future<Map<String, dynamic>> uploadClothingItem({
    required File imageFile,
    required String category,
    String? color,
    String? brand,
    String? notes,
  }) async {
    try {
      // First analyze the image
      final analysisResult = await analyzeImage(imageFile);
      
      // Then save to wardrobe
      FormData formData = FormData.fromMap({
        'image': await MultipartFile.fromFile(
          imageFile.path,
          filename: 'item_${DateTime.now().millisecondsSinceEpoch}.jpg',
        ),
        'category': category,
        'color': color ?? analysisResult['attributes']?[0]?['color'] ?? 'unknown',
        'brand': brand ?? '',
        'notes': notes ?? '',
        'ai_analysis': analysisResult,
      });
      
      final response = await _dio.post('/closet/items', data: formData);
      return response.data;
    } catch (e) {
      throw _handleError(e);
    }
  }

  /// Get user's wardrobe items
  Future<List<Map<String, dynamic>>> getWardrobeItems({String? userId}) async {
    try {
      final response = await _dio.get('/closet/items', queryParameters: {
        'user_id': userId ?? 'default_user',
      });
      
      if (response.data['items'] != null) {
        return List<Map<String, dynamic>>.from(response.data['items']);
      }
      return [];
    } catch (e) {
      return _getMockWardrobeItems();
    }
  }

  /// Delete wardrobe item
  Future<bool> deleteWardrobeItem(String itemId) async {
    try {
      await _dio.delete('/closet/items/$itemId');
      return true;
    } catch (e) {
      return false;
    }
  }

  /// Get outfit history
  Future<List<Map<String, dynamic>>> getOutfitHistory({String? userId}) async {
    try {
      final response = await _dio.get('/outfits/history', queryParameters: {
        'user_id': userId ?? 'default_user',
      });
      
      if (response.data['outfits'] != null) {
        return List<Map<String, dynamic>>.from(response.data['outfits']);
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  /// Save outfit
  Future<bool> saveOutfit(Map<String, dynamic> outfit) async {
    try {
      await _dio.post('/outfits/save', data: outfit);
      return true;
    } catch (e) {
      return false;
    }
  }

  /// Error handler
  String _handleError(dynamic error) {
    if (error is DioException) {
      switch (error.type) {
        case DioExceptionType.connectionTimeout:
        case DioExceptionType.sendTimeout:
        case DioExceptionType.receiveTimeout:
          return '연결 시간이 초과되었습니다';
        case DioExceptionType.connectionError:
          return '인터넷 연결을 확인해주세요';
        case DioExceptionType.badResponse:
          return error.response?.data['message'] ?? '서버 오류가 발생했습니다';
        default:
          return '알 수 없는 오류가 발생했습니다';
      }
    }
    return error.toString();
  }

  // Mock data for development
  List<Map<String, dynamic>> _getMockRecommendations() {
    return [
      {
        'ranking': 1,
        'outfit': {
          'top': {'type': 'shirt', 'color': 'white', 'description': '화이트 셔츠'},
          'bottom': {'type': 'jeans', 'color': 'blue', 'description': '청바지'},
          'shoes': {'type': 'sneakers', 'color': 'white', 'description': '흰색 스니커즈'},
        },
        'styling_tips': ['클래식한 캐주얼 룩', '어디에나 어울리는 조합'],
        'confidence': 92,
      },
      {
        'ranking': 2,
        'outfit': {
          'top': {'type': 'tshirt', 'color': 'black', 'description': '블랙 티셔츠'},
          'bottom': {'type': 'chinos', 'color': 'beige', 'description': '베이지 치노팬츠'},
          'shoes': {'type': 'loafers', 'color': 'brown', 'description': '브라운 로퍼'},
        },
        'styling_tips': ['세미 캐주얼 스타일', '데이트룩으로 추천'],
        'confidence': 88,
      },
    ];
  }

  List<Map<String, dynamic>> _getMockWardrobeItems() {
    return [
      {
        'id': '1',
        'category': 'top',
        'type': 'shirt',
        'color': 'white',
        'imageUrl': 'https://via.placeholder.com/150',
        'wearCount': 5,
        'lastWorn': DateTime.now().subtract(const Duration(days: 3)).toIso8601String(),
      },
      {
        'id': '2',
        'category': 'bottom',
        'type': 'jeans',
        'color': 'blue',
        'imageUrl': 'https://via.placeholder.com/150',
        'wearCount': 8,
        'lastWorn': DateTime.now().subtract(const Duration(days: 1)).toIso8601String(),
      },
    ];
  }
}