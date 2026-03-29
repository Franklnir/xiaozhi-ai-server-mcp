import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AppState,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  Animated,
  Easing,
  Linking,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { APP_PUBLIC_NAME } from '../config/appConfig';
import { Theme, ThemeName, useTheme } from '../theme/theme';
import { authStore } from '../stores/authStore';
import { deviceStore } from '../stores/deviceStore';
import {
  activateBackgroundMonitoring,
  getTrackingPermissionSummary,
  isTrackingRunning,
  openAppSettings,
  stopTracking,
  TRACKING_INTERVAL_LABEL,
  TrackingPermissionSummary,
} from '../services/deviceService';
import { apiGetDeviceSettings, apiSetDeviceSettings, apiGetConfig, apiGetPublicSettings, SocialLink } from '../api/client';

export default function AccountScreen() {
  const { theme, themeName, setThemeName } = useTheme();
  const navigation = useNavigation<any>();
  const [backendUrl, setBackendUrl] = useState('');
  const [source, setSource] = useState('Indonesia');
  const [target, setTarget] = useState('Arab');
  const [languages, setLanguages] = useState<string[]>([]);
  const [socialLinks, setSocialLinks] = useState<SocialLink[]>([]);
  const [saving, setSaving] = useState(false);
  const [trackingSummary, setTrackingSummary] = useState<TrackingPermissionSummary | null>(null);
  const [trackingRunning, setTrackingRunning] = useState(false);
  const [trackingBusy, setTrackingBusy] = useState(false);
  const [pendingBackgroundActivation, setPendingBackgroundActivation] = useState(false);

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

  const refreshBackgroundState = useCallback(async () => {
    try {
      const [summary, running] = await Promise.all([
        getTrackingPermissionSummary(),
        isTrackingRunning(),
      ]);
      setTrackingSummary(summary);
      setTrackingRunning(running);
      return { summary, running };
    } catch {
      return null;
    }
  }, []);

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
        if (url) {
          const pub = await apiGetPublicSettings(url);
          setSocialLinks(pub.social_links || []);
        }
      } catch (e) {
        // ignore
      }
      await refreshBackgroundState();
    })();
  }, [refreshBackgroundState]);

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

  useEffect(() => {
    const sub = AppState.addEventListener('change', async (state) => {
      if (state !== 'active') {
        return;
      }

      const info = await refreshBackgroundState();
      if (!pendingBackgroundActivation || !info?.summary.ready || info.running) {
        if (pendingBackgroundActivation && info?.running) {
          setPendingBackgroundActivation(false);
        }
        return;
      }

      setTrackingBusy(true);
      try {
        const activated = await activateBackgroundMonitoring();
        setTrackingSummary(activated.summary);
        setTrackingRunning(activated.trackingEnabled);
        if (activated.trackingEnabled) {
          setPendingBackgroundActivation(false);
          Alert.alert('Aktif', 'Monitor latar belakang sudah aktif dan notif status akan tampil di atas layar.');
        }
      } catch {
        // ignore refresh-time activation errors
      } finally {
        setTrackingBusy(false);
      }
    });

    return () => sub.remove();
  }, [pendingBackgroundActivation, refreshBackgroundState]);

  async function handleBackgroundMonitor() {
    setTrackingBusy(true);
    try {
      if (trackingRunning) {
        await stopTracking();
        setPendingBackgroundActivation(false);
        await refreshBackgroundState();
        return;
      }

      const result = await activateBackgroundMonitoring();
      setTrackingSummary(result.summary);
      setTrackingRunning(result.trackingEnabled);

      if (result.trackingEnabled) {
        setPendingBackgroundActivation(false);
        Alert.alert('Aktif', `Monitor latar belakang aktif. Data HP akan dikirim tiap ${TRACKING_INTERVAL_LABEL}.`);
        return;
      }

      if (result.openedSettings) {
        setPendingBackgroundActivation(true);
        Alert.alert(
          'Lanjutkan di Pengaturan',
          'Aktifkan izin lokasi di latar belakang dari halaman Pengaturan Aplikasi, lalu kembali ke aplikasi. Setelah izin aktif, monitor akan dinyalakan otomatis.',
        );
        return;
      }

      Alert.alert(
        'Izin belum lengkap',
        `Yang masih kurang: ${result.summary.missing.join(', ') || 'cek lagi izin perangkat Anda.'}`,
      );
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Gagal mengaktifkan monitor latar belakang.');
    } finally {
      setTrackingBusy(false);
    }
  }

  async function handleLogout() {
    await stopTracking().catch(() => {});
    await authStore.clear();
    await deviceStore.clear();
  }

  const backgroundStatusText = trackingRunning
    ? `Monitor aktif. Data HP berjalan di background dan sinkron tiap ${TRACKING_INTERVAL_LABEL}.`
    : trackingSummary?.ready
    ? 'Semua izin sudah siap. Tekan tombol aktifkan untuk menyalakan monitor background.'
    : 'Izin background belum lengkap. Aktifkan dari sini agar tidak perlu memicu flow izin saat login.';

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        <Animated.View style={contentAnimStyle}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Akun & Pengaturan</Text>
            <Text style={styles.headerSubtitle}>Sinkron dengan backend & web</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Backend Aplikasi</Text>
            <Text style={styles.cardSubtitle}>Alamat backend untuk {APP_PUBLIC_NAME} sudah dibundel otomatis</Text>
            <View style={styles.backendBox}>
              <Text style={styles.backendValue}>{backendUrl}</Text>
              <Text style={styles.backendHint}>User akhir tidak perlu input URL server manual dari layar akun.</Text>
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>🔗 Xiaozhi MCP Token</Text>
            <Text style={styles.cardSubtitle}>Kelola koneksi MCP endpoint ke Xiaozhi AI</Text>
            <TouchableOpacity
              style={[styles.saveBtn, { backgroundColor: theme.isNeo ? '#a5f3fc' : '#0891b2' }]}
              onPress={() => navigation.navigate('McpToken')}
            >
              <Text style={[styles.saveBtnText, theme.isNeo && { color: theme.colors.black }]}>Buka MCP Token Manager</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.card}>
            <View style={styles.monitorHeaderRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.cardTitle}>Monitor Latar Belakang</Text>
                <Text style={styles.cardSubtitle}>
                  Aktifkan izin background dari sini. Setelah aktif, notif Android akan menampilkan status online atau offline.
                </Text>
              </View>
              <View style={[styles.monitorPill, trackingRunning ? styles.monitorPillOn : styles.monitorPillOff]}>
                <Text style={styles.monitorPillText}>{trackingRunning ? 'ON' : 'OFF'}</Text>
              </View>
            </View>

            <View style={styles.monitorInfoBox}>
              <Text style={styles.monitorInfoText}>{backgroundStatusText}</Text>
            </View>

            <View style={styles.permissionGrid}>
              <View style={[styles.permissionChip, trackingSummary?.locationGranted ? styles.permissionChipOn : styles.permissionChipOff]}>
                <Text style={styles.permissionChipText}>Lokasi</Text>
              </View>
              <View style={[styles.permissionChip, trackingSummary?.backgroundGranted ? styles.permissionChipOn : styles.permissionChipOff]}>
                <Text style={styles.permissionChipText}>Latar Belakang</Text>
              </View>
              <View style={[styles.permissionChip, trackingSummary?.notificationsGranted ? styles.permissionChipOn : styles.permissionChipOff]}>
                <Text style={styles.permissionChipText}>Notifikasi</Text>
              </View>
            </View>

            <View style={styles.actionRow}>
              <TouchableOpacity
                style={[styles.saveBtn, trackingRunning ? styles.monitorStopBtn : styles.saveBtnGreen]}
                onPress={handleBackgroundMonitor}
                disabled={trackingBusy}
              >
                <Text style={styles.saveBtnText}>
                  {trackingBusy
                    ? 'Memproses...'
                    : trackingRunning
                    ? 'Matikan Monitor'
                    : 'Aktifkan Izin di Latar Belakang'}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.secondaryActionBtn} onPress={openAppSettings}>
                <Text style={styles.secondaryActionText}>Buka Pengaturan App</Text>
              </TouchableOpacity>
            </View>
          </View>

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

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Bahasa Terjemahan</Text>
            <Text style={styles.cardSubtitle}>Atur bahasa sumber dan target</Text>

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

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Media Sosial</Text>
            <Text style={styles.cardSubtitle}>Link resmi dari admin</Text>
            {socialLinks.length === 0 ? (
              <Text style={styles.emptyText}>Belum ada link.</Text>
            ) : (
              socialLinks.map((item, idx) => (
                <TouchableOpacity
                  key={`${item.label}-${idx}`}
                  style={styles.socialBtn}
                  onPress={() => Linking.openURL(item.url)}
                >
                  <Text style={styles.socialLabel}>{item.label}</Text>
                  <Text style={styles.socialUrl} numberOfLines={1}>
                    {item.url}
                  </Text>
                </TouchableOpacity>
              ))
            )}
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Akun</Text>
            <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
              <Text style={styles.logoutText}>Logout</Text>
            </TouchableOpacity>
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
    scroll: { flex: 1 },
    scrollContent: { paddingBottom: 24 },
    header: {
      paddingHorizontal: theme.spacing.lg,
      paddingTop: theme.spacing.xxl + 16,
      paddingBottom: theme.spacing.lg,
    },
    headerTitle: {
      fontSize: theme.fontSize.lg,
      fontWeight: '800',
      color: theme.colors.accentLight,
      fontFamily: theme.fonts.heading,
    },
    headerSubtitle: {
      fontSize: theme.fontSize.sm,
      color: theme.colors.textMuted,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },
    card: {
      marginHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.md,
      padding: theme.spacing.lg,
      borderRadius: theme.radius.lg,
      backgroundColor: theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      ...theme.effects.cardShadow,
    },
    cardTitle: {
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      color: theme.colors.text,
      fontFamily: theme.fonts.heading,
    },
    cardSubtitle: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textMuted,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },
    backendBox: {
      marginTop: theme.spacing.md,
      backgroundColor: theme.colors.surfaceLight,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    backendValue: {
      color: theme.colors.text,
      fontFamily: theme.fonts.mono,
      fontSize: theme.fontSize.sm,
    },
    backendHint: {
      marginTop: theme.spacing.sm,
      color: theme.colors.textMuted,
      fontSize: theme.fontSize.xs,
      lineHeight: 18,
      fontFamily: theme.fonts.body,
    },
    monitorHeaderRow: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      justifyContent: 'space-between',
      gap: theme.spacing.sm,
    },
    monitorPill: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: 6,
      borderRadius: theme.radius.full,
      borderWidth: theme.isNeo ? 2 : 1,
    },
    monitorPillOn: {
      backgroundColor: theme.isNeo ? '#dcfce7' : 'rgba(16,185,129,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.25)',
    },
    monitorPillOff: {
      backgroundColor: theme.isNeo ? '#fef3c7' : 'rgba(245,158,11,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(245,158,11,0.25)',
    },
    monitorPillText: {
      color: theme.colors.text,
      fontWeight: '700',
      fontSize: 11,
      fontFamily: theme.fonts.body,
    },
    monitorInfoBox: {
      marginTop: theme.spacing.md,
      padding: theme.spacing.sm,
      borderRadius: theme.radius.md,
      backgroundColor: theme.isNeo ? '#fff7ed' : 'rgba(59,130,246,0.08)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(59,130,246,0.16)',
    },
    monitorInfoText: {
      color: theme.colors.textSecondary,
      fontSize: theme.fontSize.xs,
      lineHeight: 18,
      fontFamily: theme.fonts.body,
    },
    permissionGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: theme.spacing.sm,
      marginTop: theme.spacing.md,
    },
    permissionChip: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: 8,
      borderRadius: theme.radius.full,
      borderWidth: theme.isNeo ? 2 : 1,
    },
    permissionChipOn: {
      backgroundColor: theme.isNeo ? '#dcfce7' : 'rgba(16,185,129,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.25)',
    },
    permissionChipOff: {
      backgroundColor: theme.isNeo ? '#fee2e2' : 'rgba(239,68,68,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(239,68,68,0.25)',
    },
    permissionChipText: {
      color: theme.colors.text,
      fontSize: 11,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    input: {
      marginTop: theme.spacing.md,
      backgroundColor: theme.colors.surfaceLight,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      color: theme.colors.text,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      fontFamily: theme.fonts.body,
    },
    saveBtn: {
      marginTop: theme.spacing.md,
      backgroundColor: theme.colors.accent,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
    },
    saveBtnGreen: { backgroundColor: theme.colors.emerald },
    monitorStopBtn: { backgroundColor: theme.colors.red },
    saveBtnText: {
      color: theme.colors.white,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    actionRow: {
      marginTop: theme.spacing.md,
      gap: theme.spacing.sm,
    },
    secondaryActionBtn: {
      backgroundColor: theme.colors.surfaceLight,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    secondaryActionText: {
      color: theme.colors.text,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    themeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing.sm, marginTop: theme.spacing.md },
    themeChip: {
      backgroundColor: theme.colors.surfaceLight,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.radius.full,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    themeChipActive: { backgroundColor: theme.colors.accentLight, borderColor: theme.colors.black },
    themeChipText: { fontSize: theme.fontSize.xs, color: theme.colors.textSecondary, fontFamily: theme.fonts.body },
    themeChipTextActive: { color: theme.colors.black, fontWeight: '700', fontFamily: theme.fonts.heading },
    label: { marginTop: theme.spacing.md, fontSize: theme.fontSize.xs, color: theme.colors.textMuted },
    chipScroll: { marginTop: theme.spacing.sm },
    langChip: {
      backgroundColor: theme.colors.surfaceLight,
      borderRadius: theme.radius.full,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      marginRight: theme.spacing.sm,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    langChipActive: { backgroundColor: theme.colors.accentLight, borderColor: theme.colors.black },
    langChipText: { fontSize: theme.fontSize.xs, color: theme.colors.textSecondary, fontFamily: theme.fonts.body },
    langChipTextActive: { color: theme.colors.black, fontWeight: '700', fontFamily: theme.fonts.heading },
    logoutBtn: {
      marginTop: theme.spacing.md,
      backgroundColor: theme.colors.red,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
    },
    logoutText: { color: theme.colors.white, fontWeight: '700' },
    emptyText: {
      marginTop: theme.spacing.sm,
      fontSize: theme.fontSize.xs,
      color: theme.colors.textMuted,
      fontFamily: theme.fonts.body,
    },
    socialBtn: {
      marginTop: theme.spacing.sm,
      backgroundColor: theme.colors.surfaceLight,
      borderRadius: theme.radius.md,
      paddingVertical: theme.spacing.sm,
      paddingHorizontal: theme.spacing.md,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    socialLabel: {
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      color: theme.colors.text,
      fontFamily: theme.fonts.heading,
    },
    socialUrl: {
      marginTop: 4,
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },
  });
