# SciG Mode MCP Mobile App

Aplikasi mobile React Native untuk mengontrol AI Xiaozhi dari smartphone.

## Fitur

- 🔐 Login aman dengan akun SciG Mode
- 🤖 Kontrol mode & persona AI
- 💬 Monitor chat Xiaozhi realtime
- ⚙️ Atur bahasa terjemahan
- 🔒 Komunikasi HTTPS terenkripsi
- 🔑 Token disimpan terenkripsi di device

## Struktur Project

```
mobile/
├── App.tsx                      # Entry point
├── index.js                     # RN registration
├── app.json                     # App name config
├── package.json                 # Dependencies
├── tsconfig.json                # TypeScript config
├── babel.config.js              # Babel config
├── metro.config.js              # Metro bundler config
├── src/
│   ├── api/
│   │   └── client.ts            # Axios API client + auth interceptor
│   ├── screens/
│   │   ├── LoginScreen.tsx      # Login (backend bawaan + credentials)
│   │   ├── DashboardScreen.tsx  # Mode control, chat viewer, MCP status
│   │   └── SettingsScreen.tsx   # Info backend, tema, bahasa
│   ├── navigation/
│   │   └── AppNavigator.tsx     # Screen routing with auth check
│   ├── stores/
│   │   └── authStore.ts         # Encrypted token storage
│   ├── config/
│   │   └── appConfig.ts         # Nama app + default backend URL
│   └── theme/
│       └── colors.ts            # Design tokens (matches web theme)
```

## Prerequisites

1. **Node.js ≥ 18**: https://nodejs.org/
2. **JDK 17**: `sudo apt install openjdk-17-jdk`
3. **Android Studio** dengan SDK dan emulator
4. **React Native CLI**: `npm install -g @react-native-community/cli`

## Setup

```bash
# 1. Masuk ke folder mobile
cd mobile

# 2. Install dependencies
npm install

# 3. Generate native project (Android)
npx react-native init ScigModeMcp --template react-native-template-typescript --directory .

# Atau jika sudah punya android/ folder:
npx react-native run-android
```

## Komunikasi Backend

Mobile app berkomunikasi dengan backend FastAPI yang sama:
- Login via `POST /login` (form data, cookie-based auth)
- Semua API call menggunakan cookie `access_token`
- Token disimpan terenkripsi di device menggunakan `react-native-encrypted-storage`
- Axios interceptor otomatis menambahkan cookie di setiap request

## Build APK

```bash
cd android
./gradlew assembleRelease
```

APK output di: `android/app/build/outputs/apk/release/app-release.apk`

## Release Signing Production

Release build production sebaiknya memakai keystore sendiri, bukan debug keystore.

Gradle akan otomatis memakai env berikut bila tersedia:

```bash
export SCIG_UPLOAD_STORE_FILE=/absolute/path/to/your-release.keystore
export SCIG_UPLOAD_STORE_PASSWORD=your_store_password
export SCIG_UPLOAD_KEY_ALIAS=your_key_alias
export SCIG_UPLOAD_KEY_PASSWORD=your_key_password
```

Workflow GitHub Actions juga mendukung secret berikut:

- `ANDROID_KEYSTORE_BASE64`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

Upload ke GitHub releases: https://github.com/Franklnir/xiaozhi-ai-mobile-mcp/releases

## Catatan

- **Backend URL bawaan app**: Atur sekali di `src/config/appConfig.ts`, lalu user APK tidak perlu mengetik URL server lagi.
- **CORS**: Backend sudah dikonfigurasi dengan CORS middleware yang mendukung cross-origin requests.
- **HTTPS**: Untuk produksi, gunakan HTTPS. Self-signed certificate tidak akan bekerja di Android tanpa konfigurasi tambahan.
