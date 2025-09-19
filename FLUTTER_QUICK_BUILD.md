# 🚀 Flutter APK 빠른 빌드 가이드

## 1️⃣ Flutter 설치 (Mac)

```bash
# Homebrew로 Flutter 설치
brew install --cask flutter

# 또는 직접 다운로드
# https://docs.flutter.dev/get-started/install/macos
```

## 2️⃣ 프로젝트 준비

```bash
cd /Users/leehanjun/Desktop/money/style-mate/frontend

# Flutter 의존성 설치
flutter pub get

# Flutter doctor 실행 (설치 확인)
flutter doctor
```

## 3️⃣ APK 빌드

```bash
# Release APK 빌드 (약 5-10분 소요)
flutter build apk --release

# 빌드 완료 후 APK 위치:
# build/app/outputs/flutter-apk/app-release.apk
```

## 4️⃣ APK 파일 찾기

```bash
# APK 파일 위치 확인
ls -la build/app/outputs/flutter-apk/

# 파일 크기 확인 (보통 15-30MB)
du -h build/app/outputs/flutter-apk/app-release.apk
```

## 5️⃣ Google Drive 업로드

1. Google Drive 접속: https://drive.google.com
2. 새 폴더 생성: "Style Mate Beta APK"
3. APK 파일 업로드
4. 공유 설정: "링크가 있는 모든 사용자"
5. 링크 복사

## 6️⃣ 베타 테스터에게 전달

### 설치 가이드 메시지:
```
🎉 Style Mate 베타 테스터 선정 축하드립니다!

📱 Android APK 다운로드:
[Google Drive 링크]

설치 방법:
1. 위 링크로 APK 다운로드
2. 다운로드 폴더에서 APK 파일 실행
3. "출처를 알 수 없는 앱" 허용 (설정 > 보안)
4. 설치 완료!

문제 발생시 답장 주세요 😊
```

## ⚡ 빠른 명령어 (복사용)

```bash
# 한 번에 실행
cd /Users/leehanjun/Desktop/money/style-mate/frontend && \
flutter clean && \
flutter pub get && \
flutter build apk --release && \
open build/app/outputs/flutter-apk/
```

## 🔧 문제 해결

### Flutter 설치 안 됨
```bash
# M1/M2 Mac
softwareupdate --install-rosetta --agree-to-license
brew install --cask flutter
```

### Android SDK 없음
```bash
# Android Studio 설치 필요
brew install --cask android-studio
```

### 빌드 실패
```bash
flutter clean
flutter pub cache clean
flutter pub get
flutter build apk --release --verbose
```

---

**참고**: Flutter 설치 후 첫 빌드는 시간이 좀 걸립니다 (10-15분)