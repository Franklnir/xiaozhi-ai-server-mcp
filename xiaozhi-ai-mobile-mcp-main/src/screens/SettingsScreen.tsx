import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  Animated,
  Easing,
} from 'react-native';
import { APP_PUBLIC_NAME, APP_PUBLIC_VERSION } from '../config/appConfig';
import { Theme, ThemeName, useTheme } from '../theme/theme';
import { authStore } from '../stores/authStore';
import { apiGetDeviceSettings, apiSetDeviceSettings, apiGetConfig } from '../api/client';

interface SettingsScreenProps {
  onBack: () => void;
}

export default function SettingsScreen({ onBack }: SettingsScreenProps) {
  const { theme, themeName, setThemeName } = useTheme();
  const [backendUrl, setBackendUrl] = useState('');
  const [source, setSource] = useState('Indonesia');
  const [target, setTarget] = useState('Arab');
  const [languages, setLanguages] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

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

  const contentAnimStyle = {
    opacity: enterAnim,
    transform: [
      {
        translateY: enterAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [10, 0],
        }),
      },
    ],
  };

  const themeOptions: { key: ThemeName; label: string }[] = [
    { key: 'default', label: 'Default (Sistem)' },
    { key: 'dark', label: 'Gelap' },
    { key: 'light', label: 'Terang' },
    { key: 'neo', label: 'Neo Brutalism' },
  ];

  useEffect(() => {
    (async () => {
      const url = await authStore.getServerUrl();
      setBackendUrl(url);
      try {
        const config = await apiGetConfig();
        setLanguages(config.languages || []);
        const settings = await apiGetDeviceSettings('default');
        setSource(settings.source || 'Indonesia');
        setTarget(settings.target || 'Arab');
      } catch (e) {
        console.warn('Settings load error:', e);
      }
    })();
  }, []);

  async function saveLanguage() {
    setSaving(true);
    try {
      await apiSetDeviceSettings('default', source, target);
      Alert.alert('Tersimpan', 'Bahasa berhasil disimpan.');
    } catch (e: any) {
      Alert.alert('Error', e.message);
    }
    setSaving(false);
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backBtnText}>← Kembali</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Pengaturan</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        <Animated.View style={contentAnimStyle}>
          {/* Server URL */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Backend Aplikasi</Text>
            <Text style={styles.cardSubtitle}>Alamat backend default untuk {APP_PUBLIC_NAME} sudah tertanam di APK</Text>
            <View style={styles.backendBox}>
              <Text style={styles.backendValue}>{backendUrl}</Text>
              <Text style={styles.backendHint}>Kalau mau ganti server, cukup ubah konfigurasi build. User biasa tidak perlu mengisi URL lagi.</Text>
            </View>
          </View>

          {/* Theme */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Tema Tampilan</Text>
            <Text style={styles.cardSubtitle}>Pilih mode gelap, terang, default, atau neo brutalism</Text>

            <View style={styles.themeRow}>
              {themeOptions.map((opt) => {
                const active = themeName === opt.key;
                return (
                  <TouchableOpacity
                    key={opt.key}
                    style={[styles.themeChip, active && styles.themeChipActive]}
                    onPress={() => setThemeName(opt.key)}
                  >
                    <Text style={[styles.themeChipText, active && styles.themeChipTextActive]}>
                      {opt.label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>

          {/* Language */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Bahasa Terjemahan</Text>
            <Text style={styles.cardSubtitle}>
              Atur bahasa sumber dan target untuk mode terjemahan
            </Text>

            <Text style={styles.label}>Bahasa Sumber</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.chipScroll}>
              {languages.map((lang) => (
                <TouchableOpacity
                  key={lang}
                  style={[styles.langChip, source === lang && styles.langChipActive]}
                  onPress={() => setSource(lang)}
                >
                  <Text style={[styles.langChipText, source === lang && styles.langChipTextActive]}>
                    {lang}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            <Text style={styles.label}>Bahasa Target</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.chipScroll}>
              {languages.map((lang) => (
                <TouchableOpacity
                  key={lang}
                  style={[styles.langChip, target === lang && styles.langChipActive]}
                  onPress={() => setTarget(lang)}
                >
                  <Text style={[styles.langChipText, target === lang && styles.langChipTextActive]}>
                    {lang}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            <TouchableOpacity
              style={[styles.saveBtn, styles.saveBtnGreen]}
              onPress={saveLanguage}
              disabled={saving}
            >
              <Text style={styles.saveBtnText}>{saving ? 'Menyimpan...' : 'Simpan Bahasa'}</Text>
            </TouchableOpacity>
          </View>

          {/* Info */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Informasi</Text>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>App</Text>
              <Text style={styles.infoValue}>{APP_PUBLIC_NAME} v{APP_PUBLIC_VERSION}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Backend</Text>
              <Text style={styles.infoValue}>{backendUrl}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Platform</Text>
              <Text style={styles.infoValue}>React Native</Text>
            </View>
          </View>

          <View style={{ height: 40 }} />
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const createStyles = (theme: Theme) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: theme.colors.bg },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingHorizontal: theme.spacing.lg,
      paddingVertical: theme.spacing.md,
      paddingTop: theme.spacing.xxl + 16,
      borderBottomWidth: theme.isNeo ? 2 : 1,
      borderBottomColor: theme.colors.panelBorder,
      backgroundColor: theme.isNeo ? theme.colors.surface : theme.colors.surfaceLight,
      ...theme.effects.cardShadow,
    },
    backBtn: { padding: theme.spacing.sm },
    backBtnText: {
      color: theme.colors.accentLight,
      fontSize: theme.fontSize.sm,
      fontWeight: '600',
      fontFamily: theme.fonts.body,
    },
    headerTitle: {
      fontSize: theme.fontSize.lg,
      fontWeight: '700',
      color: theme.colors.text,
      fontFamily: theme.fonts.heading,
    },
    scroll: { flex: 1 },
    scrollContent: { padding: theme.spacing.lg },
    card: {
      backgroundColor: theme.colors.panel,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.lg,
      padding: theme.spacing.lg,
      marginBottom: theme.spacing.lg,
      ...theme.effects.cardShadow,
    },
    cardTitle: { fontSize: theme.fontSize.md, fontWeight: '700', color: theme.colors.text, fontFamily: theme.fonts.heading },
    cardSubtitle: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: 2,
      marginBottom: theme.spacing.md,
      fontFamily: theme.fonts.body,
    },
    label: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: theme.spacing.md,
      marginBottom: theme.spacing.sm,
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
    backendBox: {
      backgroundColor: theme.isNeo ? '#fff7ed' : theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.md,
    },
    backendValue: {
      color: theme.colors.text,
      fontSize: theme.fontSize.sm,
      fontFamily: theme.fonts.mono,
    },
    backendHint: {
      marginTop: theme.spacing.sm,
      color: theme.colors.textMuted,
      fontSize: theme.fontSize.xs,
      lineHeight: 18,
      fontFamily: theme.fonts.body,
    },
    saveBtn: {
      backgroundColor: theme.colors.accent,
      borderRadius: theme.radius.md,
      paddingVertical: theme.spacing.md,
      alignItems: 'center',
      marginTop: theme.spacing.lg,
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.black : 'transparent',
      ...theme.effects.buttonShadow,
    },
    saveBtnGreen: { backgroundColor: theme.colors.emeraldDark },
    saveBtnText: { color: theme.colors.white, fontWeight: '700', fontSize: theme.fontSize.md, fontFamily: theme.fonts.heading },
    chipScroll: { flexDirection: 'row', marginBottom: theme.spacing.sm },
    langChip: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.25)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.full,
      paddingHorizontal: theme.spacing.lg,
      paddingVertical: theme.spacing.sm,
      marginRight: theme.spacing.sm,
    },
    langChipActive: {
      backgroundColor: theme.isNeo ? theme.colors.accentLight : 'rgba(16,185,129,0.2)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.5)',
    },
    langChipText: { fontSize: theme.fontSize.sm, color: theme.colors.textSecondary, fontFamily: theme.fonts.body },
    langChipTextActive: { color: theme.isNeo ? theme.colors.black : theme.colors.emerald, fontWeight: '600', fontFamily: theme.fonts.heading },
    themeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing.sm },
    themeChip: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.25)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
    },
    themeChipActive: {
      backgroundColor: theme.isNeo ? theme.colors.accentLight : 'rgba(59,130,246,0.2)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(59,130,246,0.5)',
    },
    themeChipText: { fontSize: theme.fontSize.sm, color: theme.colors.textSecondary, fontFamily: theme.fonts.body },
    themeChipTextActive: { color: theme.isNeo ? theme.colors.black : theme.colors.accentLight, fontWeight: '700', fontFamily: theme.fonts.heading },
    infoRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      paddingVertical: theme.spacing.sm,
      borderBottomWidth: theme.isNeo ? 2 : 1,
      borderBottomColor: theme.colors.panelBorder,
    },
    infoLabel: { fontSize: theme.fontSize.sm, color: theme.colors.textSecondary, fontFamily: theme.fonts.body },
    infoValue: { fontSize: theme.fontSize.sm, color: theme.colors.text, fontFamily: theme.fonts.body },
  });
