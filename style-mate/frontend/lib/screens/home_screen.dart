import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../services/api_service.dart';
import '../services/weather_service.dart';
import '../widgets/outfit_card.dart';
import '../widgets/weather_widget.dart';
import '../models/app_state.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isLoading = false;
  Map<String, dynamic>? _weatherData;
  List<Map<String, dynamic>> _recommendations = [];

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    
    try {
      // Get weather data
      final weatherService = Provider.of<WeatherService>(context, listen: false);
      final weather = await weatherService.getCurrentWeather();
      
      // Get outfit recommendations
      final apiService = Provider.of<ApiService>(context, listen: false);
      final recommendations = await apiService.getRecommendations(
        occasion: 'casual',
        weather: weather,
      );
      
      setState(() {
        _weatherData = weather;
        _recommendations = recommendations;
      });
    } catch (e) {
      // Handle error
      debugPrint('Error loading data: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 120,
            floating: true,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              title: Text(
                'Style Mate',
                style: theme.textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              background: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      theme.colorScheme.primary,
                      theme.colorScheme.secondary,
                    ],
                  ),
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Weather Section
                  Text(
                    '오늘의 날씨',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ).animate().fadeIn(duration: 500.ms),
                  const SizedBox(height: 12),
                  
                  if (_weatherData != null)
                    WeatherWidget(weatherData: _weatherData!)
                        .animate()
                        .slideX(begin: -0.2, duration: 600.ms)
                        .fadeIn(),
                  
                  const SizedBox(height: 24),
                  
                  // Today's Recommendations
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '오늘의 추천 코디',
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          // Navigate to outfit screen
                          Navigator.pushNamed(context, '/outfit');
                        },
                        child: const Text('더보기'),
                      ),
                    ],
                  ).animate().fadeIn(delay: 200.ms),
                  
                  const SizedBox(height: 12),
                  
                  if (_isLoading)
                    const Center(
                      child: CircularProgressIndicator(),
                    )
                  else if (_recommendations.isEmpty)
                    _buildEmptyState(theme)
                  else
                    _buildRecommendations(),
                  
                  const SizedBox(height: 24),
                  
                  // Quick Actions
                  Text(
                    '빠른 메뉴',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ).animate().fadeIn(delay: 400.ms),
                  
                  const SizedBox(height: 12),
                  
                  _buildQuickActions(theme),
                ],
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          Navigator.pushNamed(context, '/camera');
        },
        label: const Text('옷 추가'),
        icon: const Icon(Icons.add_a_photo),
      ).animate().scale(delay: 800.ms),
    );
  }

  Widget _buildEmptyState(ThemeData theme) {
    return Container(
      height: 200,
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: theme.colorScheme.outline.withOpacity(0.2),
        ),
      ),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.checkroom,
              size: 64,
              color: theme.colorScheme.primary.withOpacity(0.5),
            ),
            const SizedBox(height: 16),
            Text(
              '옷장에 옷을 추가해주세요',
              style: theme.textTheme.titleMedium?.copyWith(
                color: theme.colorScheme.onSurface.withOpacity(0.7),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '사진 10장으로 시작할 수 있어요',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurface.withOpacity(0.5),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 600.ms);
  }

  Widget _buildRecommendations() {
    return SizedBox(
      height: 280,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _recommendations.length,
        itemBuilder: (context, index) {
          return Padding(
            padding: EdgeInsets.only(
              right: index < _recommendations.length - 1 ? 16 : 0,
            ),
            child: OutfitCard(
              outfit: _recommendations[index],
              ranking: index + 1,
            ).animate().slideX(
              begin: 0.2,
              delay: Duration(milliseconds: 100 * index),
            ).fadeIn(),
          );
        },
      ),
    );
  }

  Widget _buildQuickActions(ThemeData theme) {
    final actions = [
      {'icon': Icons.wb_sunny, 'label': '날씨별 추천', 'color': Colors.orange},
      {'icon': Icons.event, 'label': 'TPO별 추천', 'color': Colors.blue},
      {'icon': Icons.favorite, 'label': '즐겨찾기', 'color': Colors.red},
      {'icon': Icons.history, 'label': '코디 기록', 'color': Colors.green},
    ];

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        childAspectRatio: 3,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
      ),
      itemCount: actions.length,
      itemBuilder: (context, index) {
        final action = actions[index];
        return Material(
          borderRadius: BorderRadius.circular(12),
          color: (action['color'] as Color).withOpacity(0.1),
          child: InkWell(
            onTap: () {
              // Handle quick action
            },
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Icon(
                    action['icon'] as IconData,
                    color: action['color'] as Color,
                  ),
                  const SizedBox(width: 12),
                  Text(
                    action['label'] as String,
                    style: theme.textTheme.bodyLarge?.copyWith(
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ).animate().scale(
          delay: Duration(milliseconds: 500 + (100 * index)),
        );
      },
    );
  }
}