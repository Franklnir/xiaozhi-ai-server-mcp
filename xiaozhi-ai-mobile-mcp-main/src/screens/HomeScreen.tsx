import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  RefreshControl,
  StyleSheet,
  Animated,
  Easing,
} from 'react-native';
import { Theme, useTheme } from '../theme/theme';
import { apiGetActiveMode, apiGetLastDevice, apiGetMyCodes, McpCodeInfo, ModeInfo } from '../api/client';

export default function HomeScreen() {
  const { theme } = useTheme();
  const [refreshing, setRefreshing] = useState(false);
  const [activeMode, setActiveMode] = useState<ModeInfo | null>(null);
  const [codes, setCodes] = useState<McpCodeInfo[]>([]);
  const [lastDevice, setLastDevice] = useState<string>('default');

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

  const load = useCallback(async () => {
    try {
      const last = await apiGetLastDevice();
      const deviceId = last.active_psid || 'default';
      setLastDevice(deviceId);
      const [mode, codesData] = await Promise.all([
        apiGetActiveMode(deviceId),
        apiGetMyCodes(),
      ]);
      setActiveMode(mode);
      setCodes(codesData);
    } catch (e) {
      // ignore for now
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function onRefresh() {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }

  const animStyle = {
    opacity: enterAnim,
    transform: [
      {
        translateY: enterAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [12, 0],
        }),
      },
    ],
  };

  return (
    <View style={styles.container}>
      <ScrollView
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={theme.colors.accent} />}
      >
        <Animated.View style={animStyle}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>SciG Mode</Text>
            <Text style={styles.headerSubtitle}>Home Dashboard</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Mode Aktif</Text>
            <Text style={styles.cardValue}>{activeMode?.title || '-'}</Text>
            <Text style={styles.cardMeta}>{activeMode?.name || '-'}</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>PSID Aktif</Text>
            <Text style={styles.cardValue}>{lastDevice}</Text>
            <Text style={styles.cardMeta}>Terakhir dipakai untuk routing</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Koneksi MCP</Text>
            {codes.length === 0 ? (
              <Text style={styles.muted}>Belum ada code.</Text>
            ) : (
              codes.map((c, i) => (
                <View key={i} style={styles.codeRow}>
                  <Text style={styles.codeText}>{c.code}</Text>
                  <Text style={[styles.codeStatus, c.is_connected ? styles.online : styles.offline]}>
                    {c.is_connected ? 'Online' : 'Offline'}
                  </Text>
                </View>
              ))
            )}
          </View>

          <View style={{ height: 32 }} />
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const createStyles = (theme: Theme) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: theme.colors.bg },
    header: {
      paddingHorizontal: theme.spacing.lg,
      paddingTop: theme.spacing.xxl + 16,
      paddingBottom: theme.spacing.lg,
    },
    headerTitle: {
      fontSize: theme.fontSize.xxl,
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
      color: theme.colors.textMuted,
      fontFamily: theme.fonts.body,
    },
    cardValue: {
      fontSize: theme.fontSize.lg,
      color: theme.colors.text,
      marginTop: 6,
      fontWeight: '700',
      fontFamily: theme.fonts.heading,
    },
    cardMeta: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },
    muted: {
      fontSize: theme.fontSize.sm,
      color: theme.colors.textMuted,
      marginTop: 8,
      fontFamily: theme.fonts.body,
    },
    codeRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginTop: 10,
    },
    codeText: {
      fontSize: theme.fontSize.sm,
      fontFamily: theme.fonts.mono,
      color: theme.colors.text,
    },
    codeStatus: {
      fontSize: theme.fontSize.xs,
      fontWeight: '600',
      fontFamily: theme.fonts.body,
    },
    online: { color: theme.colors.emerald },
    offline: { color: theme.colors.red },
  });
