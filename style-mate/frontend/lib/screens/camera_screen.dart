import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../services/api_service.dart';
import '../models/app_state.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  bool _isProcessing = false;
  Map<String, dynamic>? _analysisResult;
  
  // Form fields
  String _selectedCategory = 'top';
  String? _selectedColor;
  String? _brand;
  String? _notes;

  final List<String> _categories = ['top', 'bottom', 'dress', 'outerwear', 'shoes', 'bag', 'accessory'];
  final List<String> _colors = [
    'black', 'white', 'gray', 'navy', 'blue', 'red', 
    'green', 'yellow', 'pink', 'beige', 'brown', 'other'
  ];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        _cameraController = CameraController(
          cameras.first,
          ResolutionPreset.high,
        );
        await _cameraController!.initialize();
        if (mounted) setState(() {});
      }
    } catch (e) {
      debugPrint('Camera initialization error: $e');
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final XFile photo = await _cameraController!.takePicture();
      setState(() {
        _selectedImage = File(photo.path);
      });
      _analyzeImage();
    } catch (e) {
      debugPrint('Error taking picture: $e');
    }
  }

  Future<void> _pickFromGallery() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.gallery);
    if (photo != null) {
      setState(() {
        _selectedImage = File(photo.path);
      });
      _analyzeImage();
    }
  }

  Future<void> _analyzeImage() async {
    if (_selectedImage == null) return;

    setState(() => _isProcessing = true);

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final result = await apiService.analyzeImage(_selectedImage!);
      
      setState(() {
        _analysisResult = result;
        // Auto-fill detected attributes
        if (result['items'] != null && result['items'].isNotEmpty) {
          final item = result['items'][0];
          _selectedCategory = item['category'] ?? 'top';
        }
        if (result['attributes'] != null && result['attributes'].isNotEmpty) {
          final attr = result['attributes'][0];
          _selectedColor = attr['color'];
        }
      });
    } catch (e) {
      _showErrorDialog('분석 실패', '이미지 분석에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _saveToWardrobe() async {
    if (_selectedImage == null) return;

    setState(() => _isProcessing = true);

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final result = await apiService.uploadClothingItem(
        imageFile: _selectedImage!,
        category: _selectedCategory,
        color: _selectedColor,
        brand: _brand,
        notes: _notes,
      );

      if (mounted) {
        // Update app state
        final appState = Provider.of<AppState>(context, listen: false);
        appState.addClothingItem(result);

        // Show success and navigate back
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('옷장에 추가되었습니다!'),
            backgroundColor: Colors.green,
          ),
        );
        Navigator.pop(context);
      }
    } catch (e) {
      _showErrorDialog('저장 실패', '옷장 저장에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  void _showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('옷 추가하기'),
        elevation: 0,
      ),
      body: _selectedImage == null
          ? _buildCameraView(theme)
          : _buildImagePreview(theme),
    );
  }

  Widget _buildCameraView(ThemeData theme) {
    return Column(
      children: [
        Expanded(
          child: _cameraController != null &&
                  _cameraController!.value.isInitialized
              ? ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: CameraPreview(_cameraController!),
                )
              : Container(
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surface,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: const Center(
                    child: CircularProgressIndicator(),
                  ),
                ),
        ),
        Container(
          padding: const EdgeInsets.all(24),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // Gallery button
              IconButton(
                onPressed: _pickFromGallery,
                icon: const Icon(Icons.photo_library),
                iconSize: 32,
                style: IconButton.styleFrom(
                  backgroundColor: theme.colorScheme.secondary,
                  foregroundColor: theme.colorScheme.onSecondary,
                  padding: const EdgeInsets.all(16),
                ),
              ).animate().scale(delay: 100.ms),
              
              // Capture button
              IconButton(
                onPressed: _takePicture,
                icon: const Icon(Icons.camera),
                iconSize: 48,
                style: IconButton.styleFrom(
                  backgroundColor: theme.colorScheme.primary,
                  foregroundColor: theme.colorScheme.onPrimary,
                  padding: const EdgeInsets.all(20),
                ),
              ).animate().scale(delay: 200.ms),
              
              // Tips button
              IconButton(
                onPressed: () {
                  showModalBottomSheet(
                    context: context,
                    builder: (context) => _buildTipsSheet(theme),
                  );
                },
                icon: const Icon(Icons.help_outline),
                iconSize: 32,
                style: IconButton.styleFrom(
                  backgroundColor: theme.colorScheme.tertiary,
                  foregroundColor: theme.colorScheme.onTertiary,
                  padding: const EdgeInsets.all(16),
                ),
              ).animate().scale(delay: 300.ms),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildImagePreview(ThemeData theme) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Image preview
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.file(
              _selectedImage!,
              height: 300,
              width: double.infinity,
              fit: BoxFit.cover,
            ),
          ).animate().fadeIn(),
          
          const SizedBox(height: 16),
          
          // AI Analysis Result
          if (_isProcessing)
            const Center(child: CircularProgressIndicator())
          else if (_analysisResult != null)
            _buildAnalysisResult(theme),
          
          const SizedBox(height: 24),
          
          // Category selection
          Text(
            '카테고리',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            children: _categories.map((category) {
              return ChoiceChip(
                label: Text(_getCategoryLabel(category)),
                selected: _selectedCategory == category,
                onSelected: (selected) {
                  setState(() => _selectedCategory = category);
                },
              );
            }).toList(),
          ),
          
          const SizedBox(height: 16),
          
          // Color selection
          Text(
            '색상',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            children: _colors.map((color) {
              return ChoiceChip(
                label: Text(_getColorLabel(color)),
                selected: _selectedColor == color,
                onSelected: (selected) {
                  setState(() => _selectedColor = selected ? color : null);
                },
              );
            }).toList(),
          ),
          
          const SizedBox(height: 16),
          
          // Brand input
          TextField(
            onChanged: (value) => _brand = value,
            decoration: InputDecoration(
              labelText: '브랜드 (선택)',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
          
          const SizedBox(height: 16),
          
          // Notes input
          TextField(
            onChanged: (value) => _notes = value,
            maxLines: 3,
            decoration: InputDecoration(
              labelText: '메모 (선택)',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
          
          const SizedBox(height: 24),
          
          // Action buttons
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: () {
                    setState(() {
                      _selectedImage = null;
                      _analysisResult = null;
                    });
                  },
                  child: const Text('다시 찍기'),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: FilledButton(
                  onPressed: _isProcessing ? null : _saveToWardrobe,
                  child: _isProcessing
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                          ),
                        )
                      : const Text('옷장에 추가'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAnalysisResult(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.primaryContainer.withOpacity(0.3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: theme.colorScheme.primary.withOpacity(0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.auto_awesome,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(width: 8),
              Text(
                'AI 분석 결과',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: theme.colorScheme.primary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (_analysisResult!['items'] != null)
            ...(_analysisResult!['items'] as List).map((item) {
              return Text(
                '• ${item['category'] ?? 'Unknown'} (${(item['confidence'] * 100).toStringAsFixed(0)}% 확신도)',
                style: theme.textTheme.bodyMedium,
              );
            }).toList(),
        ],
      ),
    ).animate().slideY(begin: 0.1).fadeIn();
  }

  Widget _buildTipsSheet(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '촬영 팁',
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          _buildTipItem('🔆', '밝은 곳에서 촬영하세요'),
          _buildTipItem('📐', '옷을 평평하게 놓고 찍으세요'),
          _buildTipItem('🎯', '옷이 화면 중앙에 오도록 하세요'),
          _buildTipItem('🖼️', '배경은 단색이 좋아요'),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('확인'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTipItem(String emoji, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Text(emoji, style: const TextStyle(fontSize: 24)),
          const SizedBox(width: 12),
          Text(text),
        ],
      ),
    );
  }

  String _getCategoryLabel(String category) {
    final labels = {
      'top': '상의',
      'bottom': '하의',
      'dress': '원피스',
      'outerwear': '아우터',
      'shoes': '신발',
      'bag': '가방',
      'accessory': '액세서리',
    };
    return labels[category] ?? category;
  }

  String _getColorLabel(String color) {
    final labels = {
      'black': '블랙',
      'white': '화이트',
      'gray': '그레이',
      'navy': '네이비',
      'blue': '블루',
      'red': '레드',
      'green': '그린',
      'yellow': '옐로우',
      'pink': '핑크',
      'beige': '베이지',
      'brown': '브라운',
      'other': '기타',
    };
    return labels[color] ?? color;
  }
}