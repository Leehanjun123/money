# 📱 Flutter 앱 빌드 및 배포 가이드

## 🎯 사전 준비

### 필수 설치 프로그램
- Flutter SDK 3.0 이상
- Android Studio (Android 빌드용)
- Xcode 14+ (iOS 빌드용, Mac 필요)
- VS Code 또는 IntelliJ IDEA

## 🔧 Railway API 연동

### 1. API URL 업데이트

`style-mate/frontend/lib/services/api_service.dart` 파일 수정:

```dart
// 개발 환경
// static const String baseUrl = 'http://localhost:8000/api';

// 프로덕션 환경 (Railway URL로 변경)
static const String baseUrl = 'https://YOUR-APP.up.railway.app/api';
```

Railway 대시보드에서 실제 URL 확인 후 변경하세요.

## 📦 Android APK 빌드

### 1. 프로젝트 설정

```bash
cd style-mate/frontend
flutter clean
flutter pub get
```

### 2. 앱 아이콘 설정

`android/app/src/main/res/` 폴더에 아이콘 추가:
- mipmap-hdpi (72x72)
- mipmap-mdpi (48x48)
- mipmap-xhdpi (96x96)
- mipmap-xxhdpi (144x144)
- mipmap-xxxhdpi (192x192)

### 3. 앱 정보 수정

`android/app/src/main/AndroidManifest.xml`:

```xml
<application
    android:label="Style Mate"
    android:icon="@mipmap/ic_launcher">
```

### 4. 권한 설정

AndroidManifest.xml에 필요한 권한 추가:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
```

### 5. Release APK 빌드

```bash
# 릴리즈 APK 빌드
flutter build apk --release

# 빌드된 APK 위치
# build/app/outputs/flutter-apk/app-release.apk
```

### 6. App Bundle 빌드 (Play Store용)

```bash
flutter build appbundle --release

# 빌드된 AAB 위치
# build/app/outputs/bundle/release/app-release.aab
```

## 🍎 iOS 빌드 (TestFlight)

### 1. Xcode 프로젝트 설정

```bash
cd ios
pod install
open Runner.xcworkspace
```

### 2. Bundle Identifier 설정

Xcode에서:
1. Runner 프로젝트 선택
2. General 탭
3. Bundle Identifier: `com.yourcompany.stylemate`

### 3. 앱 아이콘 설정

`ios/Runner/Assets.xcassets/AppIcon.appiconset/`에 아이콘 추가

### 4. Info.plist 권한 설정

```xml
<key>NSCameraUsageDescription</key>
<string>옷 사진을 촬영하기 위해 카메라 접근이 필요합니다</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>갤러리에서 옷 사진을 선택하기 위해 접근이 필요합니다</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>날씨 정보를 제공하기 위해 위치 정보가 필요합니다</string>
```

### 5. 빌드 및 아카이브

```bash
flutter build ios --release
```

Xcode에서:
1. Product → Archive
2. Distribute App → App Store Connect
3. Upload

## 🧪 베타 테스트 배포

### Android - APK 직접 배포

1. APK 파일을 Google Drive 또는 Dropbox에 업로드
2. 공유 링크 생성
3. 베타 테스터에게 링크 전송
4. 설치 가이드:
   - 설정 → 보안 → 출처를 알 수 없는 앱 설치 허용
   - APK 다운로드 및 설치

### iOS - TestFlight

1. App Store Connect 접속
2. 앱 생성
3. TestFlight 탭
4. 빌드 선택
5. 외부 테스터 그룹 생성
6. 이메일로 초대

## 📝 베타 테스터를 위한 설치 가이드

### Android 사용자

```markdown
1. 이메일로 받은 APK 다운로드 링크 클릭
2. "출처를 알 수 없는 앱" 설치 허용
   - 설정 > 보안 > 출처를 알 수 없는 앱
3. 다운로드된 APK 파일 실행
4. "설치" 버튼 클릭
5. 앱 실행
```

### iOS 사용자

```markdown
1. TestFlight 앱 다운로드 (App Store)
2. 이메일로 받은 초대 링크 클릭
3. "View in TestFlight" 버튼 클릭
4. "Accept" 후 "Install" 클릭
5. 앱 실행
```

## 🐛 디버깅 팁

### API 연결 문제

```dart
// lib/services/api_service.dart에 로그 추가
print('API URL: $baseUrl');
print('Response: ${response.statusCode}');
print('Body: ${response.body}');
```

### 빌드 에러 해결

```bash
# 캐시 클리어
flutter clean
flutter pub cache clean
flutter pub get

# iOS 관련 문제
cd ios
pod deintegrate
pod install

# Android 관련 문제
cd android
./gradlew clean
./gradlew build
```

## 📱 앱 버전 관리

### pubspec.yaml

```yaml
version: 1.0.0+1
# 1.0.0 = 버전명 (사용자에게 표시)
# +1 = 빌드 번호 (내부 관리용)
```

### 버전 업데이트 시

```yaml
# Minor update
version: 1.0.1+2

# Major update  
version: 1.1.0+10
```

## 🚀 배포 체크리스트

### 빌드 전
- [ ] Railway API URL 업데이트
- [ ] 앱 아이콘 설정
- [ ] 스플래시 화면 설정
- [ ] 권한 설정 확인
- [ ] 버전 번호 업데이트

### 테스트
- [ ] API 연결 테스트
- [ ] 카메라 기능 테스트
- [ ] 이미지 업로드 테스트
- [ ] 코디 추천 테스트
- [ ] 오프라인 모드 테스트

### 배포
- [ ] Release 빌드 생성
- [ ] APK/IPA 파일 백업
- [ ] 베타 테스터 그룹 생성
- [ ] 설치 가이드 전송
- [ ] 피드백 채널 안내

## 📞 지원

문제 발생 시:
- GitHub Issues: https://github.com/yourusername/style-mate
- Email: stylemate.support@gmail.com
- Discord: [초대 링크]

---

**💡 Tip**: 첫 배포는 소규모 그룹(10-20명)으로 시작하여 피드백을 받고 개선한 후 확대하세요!