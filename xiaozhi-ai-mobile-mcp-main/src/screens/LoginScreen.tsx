import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Animated,
  Easing,
} from 'react-native';
import { APP_PUBLIC_NAME } from '../config/appConfig';
import { Theme, useTheme } from '../theme/theme';
import { apiGetPublicSettings, apiLogin, apiRegister } from '../api/client';
import { authStore } from '../stores/authStore';

interface LoginScreenProps {
  onLoginSuccess?: () => void;
}

type LangKey = 'id' | 'en' | 'ar' | 'zh' | 'ru' | 'vi';

const LANG_OPTIONS: { key: LangKey; label: string }[] = [
  { key: 'id', label: 'ID' },
  { key: 'en', label: 'EN' },
  { key: 'ar', label: 'AR' },
  { key: 'zh', label: 'ZH' },
  { key: 'ru', label: 'RU' },
  { key: 'vi', label: 'VI' },
];

const TEXT: Record<LangKey, Record<string, string>> = {
  id: {
    login: 'Login',
    register: 'Register',
    loginSubtitle: `Masuk dengan akun ${APP_PUBLIC_NAME} Anda`,
    registerSubtitle: `Buat akun baru ${APP_PUBLIC_NAME}`,
    backend: 'Backend',
    backendHint: 'Alamat server sudah ditanam di APK ini, jadi user tidak perlu isi URL manual.',
    username: 'Username',
    password: 'Password',
    confirmPassword: 'Konfirmasi Password',
    registerCode: 'Kode Register',
    loginBtn: 'Masuk',
    registerBtn: 'Daftar',
    haveAccount: 'Sudah punya akun?',
    noAccount: 'Belum punya akun?',
    switchToLogin: 'Masuk di sini',
    switchToRegister: 'Daftar di sini',
    required: 'Semua field wajib diisi',
    codeRequired: 'Kode register wajib diisi',
    registerNoCode: 'Registrasi tanpa code (diset oleh admin).',
  },
  en: {
    login: 'Login',
    register: 'Register',
    loginSubtitle: `Sign in with your ${APP_PUBLIC_NAME} account`,
    registerSubtitle: `Create a new ${APP_PUBLIC_NAME} account`,
    backend: 'Backend',
    backendHint: 'The backend address is bundled into this APK, so end users do not need to type a server URL.',
    username: 'Username',
    password: 'Password',
    confirmPassword: 'Confirm Password',
    registerCode: 'Register Code',
    loginBtn: 'Sign In',
    registerBtn: 'Sign Up',
    haveAccount: 'Already have an account?',
    noAccount: "Don't have an account?",
    switchToLogin: 'Sign in here',
    switchToRegister: 'Register here',
    required: 'All fields are required',
    codeRequired: 'Register code is required',
    registerNoCode: 'Registration without code (set by admin).',
  },
  ar: {
    login: 'تسجيل الدخول',
    register: 'تسجيل',
    loginSubtitle: `سجّل الدخول بحساب ${APP_PUBLIC_NAME}`,
    registerSubtitle: `أنشئ حساب ${APP_PUBLIC_NAME} جديد`,
    backend: 'الخادم',
    backendHint: 'عنوان الخادم مدمج داخل ملف APK، لذلك لا يحتاج المستخدم لإدخال الرابط يدويًا.',
    username: 'اسم المستخدم',
    password: 'كلمة المرور',
    confirmPassword: 'تأكيد كلمة المرور',
    registerCode: 'رمز التسجيل',
    loginBtn: 'دخول',
    registerBtn: 'تسجيل',
    haveAccount: 'لديك حساب؟',
    noAccount: 'ليس لديك حساب؟',
    switchToLogin: 'سجّل هنا',
    switchToRegister: 'سجّل هنا',
    required: 'جميع الحقول مطلوبة',
    codeRequired: 'رمز التسجيل مطلوب',
    registerNoCode: 'التسجيل بدون رمز (من الإعدادات).',
  },
  zh: {
    login: '登录',
    register: '注册',
    loginSubtitle: `使用您的 ${APP_PUBLIC_NAME} 账号登录`,
    registerSubtitle: `创建新的 ${APP_PUBLIC_NAME} 账号`,
    backend: '后端',
    backendHint: '后端地址已经内置在此 APK 中，普通用户无需手动填写服务器 URL。',
    username: '用户名',
    password: '密码',
    confirmPassword: '确认密码',
    registerCode: '注册码',
    loginBtn: '登录',
    registerBtn: '注册',
    haveAccount: '已经有账号？',
    noAccount: '还没有账号？',
    switchToLogin: '在这里登录',
    switchToRegister: '在这里注册',
    required: '所有字段都必须填写',
    codeRequired: '必须填写注册码',
    registerNoCode: '无需注册码注册（由管理员设置）。',
  },
  ru: {
    login: 'Вход',
    register: 'Регистрация',
    loginSubtitle: `Войдите с аккаунтом ${APP_PUBLIC_NAME}`,
    registerSubtitle: `Создайте новый аккаунт ${APP_PUBLIC_NAME}`,
    backend: 'Бэкенд',
    backendHint: 'Адрес сервера уже встроен в этот APK, поэтому обычному пользователю не нужно вводить URL вручную.',
    username: 'Имя пользователя',
    password: 'Пароль',
    confirmPassword: 'Подтвердите пароль',
    registerCode: 'Код регистрации',
    loginBtn: 'Войти',
    registerBtn: 'Создать аккаунт',
    haveAccount: 'Уже есть аккаунт?',
    noAccount: 'Нет аккаунта?',
    switchToLogin: 'Войти здесь',
    switchToRegister: 'Зарегистрироваться здесь',
    required: 'Все поля обязательны',
    codeRequired: 'Нужен код регистрации',
    registerNoCode: 'Регистрация без кода (настраивается админом).',
  },
  vi: {
    login: 'Đăng nhập',
    register: 'Đăng ký',
    loginSubtitle: `Đăng nhập bằng tài khoản ${APP_PUBLIC_NAME}`,
    registerSubtitle: `Tạo tài khoản ${APP_PUBLIC_NAME} mới`,
    backend: 'Backend',
    backendHint: 'Địa chỉ backend đã được nhúng sẵn trong APK này nên người dùng không cần nhập URL máy chủ thủ công.',
    username: 'Tên đăng nhập',
    password: 'Mật khẩu',
    confirmPassword: 'Xác nhận mật khẩu',
    registerCode: 'Mã đăng ký',
    loginBtn: 'Đăng nhập',
    registerBtn: 'Đăng ký',
    haveAccount: 'Đã có tài khoản?',
    noAccount: 'Chưa có tài khoản?',
    switchToLogin: 'Đăng nhập tại đây',
    switchToRegister: 'Đăng ký tại đây',
    required: 'Vui lòng điền đầy đủ tất cả trường',
    codeRequired: 'Bắt buộc nhập mã đăng ký',
    registerNoCode: 'Đăng ký không cần mã (do quản trị viên thiết lập).',
  },
};

export default function LoginScreen({ onLoginSuccess }: LoginScreenProps) {
  const { theme } = useTheme();
  const [backendUrl, setBackendUrl] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [language, setLanguage] = useState<LangKey>('id');
  const [registerRequiresCode, setRegisterRequiresCode] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [registerCode, setRegisterCode] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      const saved = await authStore.getServerUrl();
      setBackendUrl(saved);
      const savedLang = await authStore.getLanguage();
      if (
        savedLang === 'id' ||
        savedLang === 'en' ||
        savedLang === 'ar' ||
        savedLang === 'zh' ||
        savedLang === 'ru' ||
        savedLang === 'vi'
      ) {
        setLanguage(savedLang);
      }
      try {
        const settings = await apiGetPublicSettings(saved);
        setRegisterRequiresCode(settings.register_requires_code !== false);
      } catch {
        // ignore
      }
    })();
  }, []);

  const enterAnim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(enterAnim, {
      toValue: 1,
      duration: 520,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [enterAnim]);

  const styles = useMemo(() => createStyles(theme), [theme]);
  const t = (key: string) => TEXT[language]?.[key] || TEXT.id[key] || key;

  const logoAnimStyle = {
    opacity: enterAnim,
    transform: [
      {
        translateY: enterAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [-12, 0],
        }),
      },
    ],
  };

  const cardAnimStyle = {
    opacity: enterAnim,
    transform: [
      {
        translateY: enterAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [18, 0],
        }),
      },
    ],
  };

  function switchMode(nextIsRegister: boolean) {
    setIsRegister(nextIsRegister);
    setError('');
  }

  async function handleLogin() {
    if (!username.trim() || !password.trim()) {
      setError(t('required'));
      return;
    }

    setLoading(true);
    setError('');

    try {
      const result = await apiLogin(undefined, username.trim(), password);
      if (result.success) {
        onLoginSuccess?.();
      } else {
        setError(result.error || 'Login gagal');
      }
    } catch (e: any) {
      setError(e.message || 'Terjadi kesalahan');
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister() {
    if (!username.trim() || !password.trim() || !confirmPassword.trim()) {
      setError(t('required'));
      return;
    }
    if (registerRequiresCode && !registerCode.trim()) {
      setError(t('codeRequired'));
      return;
    }

    setLoading(true);
    setError('');

    try {
      const result = await apiRegister(
        undefined,
        username.trim(),
        password,
        confirmPassword,
        registerRequiresCode ? registerCode.trim() : undefined,
      );
      if (result.success) {
        onLoginSuccess?.();
      } else {
        setError(result.error || 'Register gagal');
      }
    } catch (e: any) {
      setError(e.message || 'Terjadi kesalahan');
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.langSelector}>
        {LANG_OPTIONS.map((opt) => {
          const active = language === opt.key;
          return (
            <TouchableOpacity
              key={opt.key}
              style={[styles.langChip, active && styles.langChipActive]}
              onPress={() => {
                setLanguage(opt.key);
                authStore.setLanguage(opt.key).catch(() => {});
              }}
            >
              <Text style={[styles.langChipText, active && styles.langChipTextActive]}>{opt.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        {/* Logo area */}
        <Animated.View style={[styles.logoArea, logoAnimStyle]}>
          <Text style={styles.logoIcon}>🤖</Text>
          <Text style={styles.logoTitle}>{APP_PUBLIC_NAME}</Text>
          <Text style={styles.logoSubtitle}>Xiaozhi AI Controller</Text>
        </Animated.View>

        {/* Login card */}
        <Animated.View style={[styles.card, cardAnimStyle]}>
          <Text style={styles.cardTitle}>{isRegister ? t('register') : t('login')}</Text>
          <Text style={styles.cardSubtitle}>
            {isRegister ? t('registerSubtitle') : t('loginSubtitle')}
          </Text>

          {error ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>⚠️ {error}</Text>
            </View>
          ) : null}

          <View style={styles.backendBox}>
            <Text style={styles.backendLabel}>{t('backend')}</Text>
            <Text style={styles.backendValue}>{backendUrl}</Text>
            <Text style={styles.backendHint}>{t('backendHint')}</Text>
          </View>

          <Text style={styles.label}>{t('username')}</Text>
          <TextInput
            style={styles.input}
            placeholder="username"
            placeholderTextColor={theme.colors.textMuted}
            value={username}
            onChangeText={setUsername}
            autoCapitalize="none"
            autoCorrect={false}
          />

          <Text style={styles.label}>{t('password')}</Text>
          <View style={styles.passwordRow}>
            <TextInput
              style={styles.passwordInput}
              placeholder="••••••"
              placeholderTextColor={theme.colors.textMuted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPassword}
            />
            <TouchableOpacity
              style={styles.eyeButton}
              onPress={() => setShowPassword((prev) => !prev)}
              activeOpacity={0.7}
            >
              <Text style={styles.eyeText}>{showPassword ? '🙈' : '👁'}</Text>
            </TouchableOpacity>
          </View>

          {isRegister ? (
            <>
              <Text style={styles.label}>{t('confirmPassword')}</Text>
              <View style={styles.passwordRow}>
                <TextInput
                  style={styles.passwordInput}
                  placeholder="••••••"
                  placeholderTextColor={theme.colors.textMuted}
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  secureTextEntry={!showConfirmPassword}
                />
                <TouchableOpacity
                  style={styles.eyeButton}
                  onPress={() => setShowConfirmPassword((prev) => !prev)}
                  activeOpacity={0.7}
                >
                  <Text style={styles.eyeText}>{showConfirmPassword ? '🙈' : '👁'}</Text>
                </TouchableOpacity>
              </View>

              {registerRequiresCode ? (
                <>
                  <Text style={styles.label}>{t('registerCode')}</Text>
                  <TextInput
                    style={styles.input}
                    placeholder="10 karakter"
                    placeholderTextColor={theme.colors.textMuted}
                    value={registerCode}
                    onChangeText={setRegisterCode}
                    autoCapitalize="characters"
                    autoCorrect={false}
                  />
                </>
              ) : (
                <Text style={styles.infoText}>{t('registerNoCode')}</Text>
              )}
            </>
          ) : null}

          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={isRegister ? handleRegister : handleLogin}
            disabled={loading}
            activeOpacity={0.8}
          >
            {loading ? (
              <ActivityIndicator color={theme.colors.white} />
            ) : (
              <Text style={styles.buttonText}>{isRegister ? t('registerBtn') : t('loginBtn')}</Text>
            )}
          </TouchableOpacity>

          <Text style={styles.hint}>
            {isRegister ? t('haveAccount') : t('noAccount')}{' '}
            <Text
              style={styles.hintLink}
              onPress={() => switchMode(!isRegister)}
            >
              {isRegister ? t('switchToLogin') : t('switchToRegister')}
            </Text>
          </Text>
        </Animated.View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const createStyles = (theme: Theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.bg,
    },
    langSelector: {
      position: 'absolute',
      top: theme.spacing.lg,
      left: theme.spacing.lg,
      flexDirection: 'row',
      gap: theme.spacing.xs,
      zIndex: 10,
    },
    langChip: {
      backgroundColor: theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.full,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.xs,
    },
    langChipActive: {
      backgroundColor: theme.colors.accentLight,
      borderColor: theme.isNeo ? theme.colors.black : theme.colors.accent,
    },
    langChipText: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },
    langChipTextActive: {
      color: theme.colors.black,
      fontWeight: '700',
      fontFamily: theme.fonts.heading,
    },
    scrollContent: {
      flexGrow: 1,
      justifyContent: 'center',
      padding: theme.spacing.xl,
    },
    logoArea: {
      alignItems: 'center',
      marginBottom: theme.spacing.xxl,
    },
    logoIcon: {
      fontSize: 56,
      marginBottom: theme.spacing.sm,
    },
    logoTitle: {
      fontSize: theme.fontSize.xxl,
      fontWeight: '800',
      color: theme.colors.accentLight,
      fontFamily: theme.fonts.heading,
    },
    logoSubtitle: {
      fontSize: theme.fontSize.sm,
      color: theme.colors.textSecondary,
      marginTop: theme.spacing.xs,
      fontFamily: theme.fonts.body,
    },
    card: {
      backgroundColor: theme.colors.panel,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.xl,
      padding: theme.spacing.xl,
      ...theme.effects.cardShadow,
    },
    cardTitle: {
      fontSize: theme.fontSize.xl,
      fontWeight: '700',
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
      fontFamily: theme.fonts.heading,
    },
    cardSubtitle: {
      fontSize: theme.fontSize.sm,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.lg,
      fontFamily: theme.fonts.body,
    },
    errorBox: {
      backgroundColor: theme.isNeo ? '#fee2e2' : 'rgba(239,68,68,0.1)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.redDark : 'rgba(239,68,68,0.3)',
      borderRadius: theme.radius.md,
      padding: theme.spacing.md,
      marginBottom: theme.spacing.lg,
    },
    errorText: {
      color: theme.isNeo ? theme.colors.redDark : '#fca5a5',
      fontSize: theme.fontSize.sm,
      fontFamily: theme.fonts.body,
    },
    backendBox: {
      marginTop: theme.spacing.md,
      backgroundColor: theme.isNeo ? '#fff7ed' : theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.md,
    },
    backendLabel: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },
    backendValue: {
      marginTop: theme.spacing.xs,
      fontSize: theme.fontSize.sm,
      color: theme.colors.text,
      fontFamily: theme.fonts.mono,
    },
    backendHint: {
      marginTop: theme.spacing.sm,
      fontSize: theme.fontSize.xs,
      lineHeight: 18,
      color: theme.colors.textMuted,
      fontFamily: theme.fonts.body,
    },
    label: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.xs,
      marginTop: theme.spacing.md,
      fontFamily: theme.fonts.body,
    },
    infoText: {
      marginTop: theme.spacing.sm,
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },
    input: {
      backgroundColor: theme.isNeo ? '#fff7ed' : theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.md,
      color: theme.colors.text,
      fontSize: theme.fontSize.md,
      fontFamily: theme.fonts.body,
    },
    passwordRow: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.isNeo ? '#fff7ed' : theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
    },
    passwordInput: {
      flex: 1,
      paddingVertical: theme.spacing.md,
      color: theme.colors.text,
      fontSize: theme.fontSize.md,
      fontFamily: theme.fonts.body,
    },
    eyeButton: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: theme.spacing.sm,
    },
    eyeText: {
      color: theme.colors.textSecondary,
      fontSize: theme.fontSize.md,
    },
    button: {
      backgroundColor: theme.colors.accent,
      borderRadius: theme.radius.md,
      paddingVertical: theme.spacing.lg,
      alignItems: 'center',
      justifyContent: 'center',
      marginTop: theme.spacing.xl,
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.black : 'transparent',
      ...theme.effects.buttonShadow,
    },
    buttonDisabled: {
      opacity: 0.6,
    },
    buttonText: {
      color: theme.colors.white,
      fontSize: theme.fontSize.lg,
      fontWeight: '700',
      fontFamily: theme.fonts.heading,
    },
    hint: {
      textAlign: 'center',
      fontSize: theme.fontSize.xs,
      color: theme.colors.textMuted,
      marginTop: theme.spacing.lg,
      fontFamily: theme.fonts.body,
    },
    hintLink: {
      color: theme.colors.accentLight,
      textDecorationLine: 'underline',
    },
  });
