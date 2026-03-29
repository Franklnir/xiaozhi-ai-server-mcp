import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  RefreshControl,
  FlatList,
  Alert,
  Animated,
  Easing,
  Platform,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Theme, useTheme } from '../theme/theme';
import {
  apiGetModes,
  apiGetActiveMode,
  apiSetActiveMode,
  apiGetMyCodes,
  apiGetThreads,
  apiGetMessages,
  apiGetLastDevice,
  ModeInfo,
  McpCodeInfo,
  ThreadInfo,
  MessageInfo,
} from '../api/client';

interface DashboardScreenProps {
  onLogout: () => void;
  onSettings: () => void;
}

export default function DashboardScreen({ onLogout, onSettings }: DashboardScreenProps) {
  const { theme } = useTheme();
  const navigation = useNavigation<any>();
  const [refreshing, setRefreshing] = useState(false);
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [activeMode, setActiveMode] = useState<ModeInfo | null>(null);
  const [codes, setCodes] = useState<McpCodeInfo[]>([]);
  const [threads, setThreads] = useState<ThreadInfo[]>([]);
  const [messages, setMessages] = useState<MessageInfo[]>([]);
  const [activeThread, setActiveThread] = useState<number | null>(null);
  const [deviceId, setDeviceId] = useState('default');
  const [loadingMode, setLoadingMode] = useState<number | null>(null);

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

  const headerAnimStyle = {
    opacity: enterAnim,
    transform: [
      {
        translateY: enterAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [-8, 0],
        }),
      },
    ],
  };

  const contentAnimStyle = {
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

  const loadData = useCallback(async () => {
    try {
      const [modesData, codesData, lastDevice] = await Promise.all([
        apiGetModes(),
        apiGetMyCodes(),
        apiGetLastDevice(),
      ]);
      setModes(modesData);
      setCodes(codesData);

      const did = lastDevice.active_psid || 'default';
      setDeviceId(did);

      const active = await apiGetActiveMode(did);
      setActiveMode(active);

      const threadsData = await apiGetThreads(did);
      setThreads(threadsData);

      if (threadsData.length > 0 && !activeThread) {
        setActiveThread(threadsData[0].id);
        const msgs = await apiGetMessages(did, threadsData[0].id);
        setMessages(msgs.filter((m: MessageInfo) => m.role === 'user' || m.role === 'assistant'));
      }
    } catch (e: any) {
      console.warn('Load error:', e.message);
    }
  }, [activeThread]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  async function onRefresh() {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }

  async function selectMode(modeId: number) {
    setLoadingMode(modeId);
    try {
      await apiSetActiveMode(deviceId, modeId);
      const active = await apiGetActiveMode(deviceId);
      setActiveMode(active);
    } catch (e: any) {
      Alert.alert('Error', e.message);
    }
    setLoadingMode(null);
  }

  async function selectThread(threadId: number) {
    setActiveThread(threadId);
    try {
      const msgs = await apiGetMessages(deviceId, threadId);
      setMessages(msgs.filter((m: MessageInfo) => m.role === 'user' || m.role === 'assistant'));
    } catch (e: any) {
      Alert.alert('Error', e.message);
    }
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <Animated.View style={[styles.header, headerAnimStyle]}>
        <View>
          <Text style={styles.headerTitle}>SciG Mode</Text>
          <Text style={styles.headerSubtitle}>Device: {deviceId}</Text>
        </View>
        <View style={styles.headerButtons}>
          <TouchableOpacity style={styles.headerBtn} onPress={onSettings}>
            <Text style={styles.headerBtnText}>⚙️</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.headerBtn, styles.logoutBtn]} onPress={onLogout}>
            <Text style={styles.headerBtnText}>↪️</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>

      <ScrollView
        style={styles.scroll}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={theme.colors.accent} />
        }
      >
        <Animated.View style={contentAnimStyle}>
          {/* MCP Codes Status */}
          <View style={styles.section}>
            <View style={styles.sectionHeaderRow}>
              <Text style={styles.sectionTitle}>Koneksi MCP</Text>
              <TouchableOpacity
                style={styles.mcpManageBtn}
                onPress={() => navigation.navigate('McpToken')}
              >
                <Text style={styles.mcpManageBtnText}>🔗 Kelola Token</Text>
              </TouchableOpacity>
            </View>
            {codes.length === 0 ? (
              <View style={styles.mcpEmptyCard}>
                <Text style={styles.mutedText}>Belum ada code MCP.</Text>
                <TouchableOpacity
                  style={styles.mcpSetupBtn}
                  onPress={() => navigation.navigate('McpToken')}
                >
                  <Text style={styles.mcpSetupBtnText}>+ Setup MCP Endpoint</Text>
                </TouchableOpacity>
              </View>
            ) : (
              codes.map((c, i) => {
                const ok = !!c.is_connected;
                return (
                  <TouchableOpacity
                    key={i}
                    style={[styles.codeCard, ok ? styles.codeCardOnline : styles.codeCardOffline]}
                    onPress={() => navigation.navigate('McpToken')}
                    activeOpacity={0.7}
                  >
                    <View style={styles.codeRow}>
                      <View style={styles.codeLeftRow}>
                        <View style={[styles.statusDot, ok ? styles.dotOnline : styles.dotOffline]} />
                        <Text style={styles.codeText}>{c.code}</Text>
                      </View>
                      <View style={styles.codeRightRow}>
                        <View style={[styles.tokenBadge, c.has_token ? styles.tokenBadgeOn : styles.tokenBadgeOff]}>
                          <Text style={styles.tokenBadgeText}>
                            {c.has_token ? 'Token ✓' : 'No Token'}
                          </Text>
                        </View>
                        <Text style={[styles.codeStatusText, ok ? styles.codeStatusOn : styles.codeStatusOff]}>
                          {ok ? '🟢' : '🔴'}
                        </Text>
                      </View>
                    </View>
                    {c.last_error ? (
                      <Text style={styles.codeError} numberOfLines={1}>
                        ⚠️ {c.last_error}
                      </Text>
                    ) : null}
                  </TouchableOpacity>
                );
              })
            )}
          </View>

          {/* Active Mode */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Mode Aktif</Text>
            <View style={styles.activeModeCard}>
              <Text style={styles.activeModeTitle}>{activeMode?.title || '-'}</Text>
              <Text style={styles.activeModeName}>{activeMode?.name || '-'}</Text>
            </View>
          </View>

          {/* Mode Selector */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Pilih Mode</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {modes.map((m) => (
                <TouchableOpacity
                  key={m.id}
                  style={[
                    styles.modeChip,
                    activeMode?.id === m.id && styles.modeChipActive,
                  ]}
                  onPress={() => selectMode(m.id)}
                  disabled={loadingMode !== null}
                >
                  <Text
                    style={[
                      styles.modeChipText,
                      activeMode?.id === m.id && styles.modeChipTextActive,
                    ]}
                  >
                    {loadingMode === m.id ? '⏳' : ''} {m.title}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>

          {/* Chat Threads */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Chat Threads</Text>
            {threads.length === 0 ? (
              <Text style={styles.mutedText}>Belum ada thread.</Text>
            ) : (
              threads.slice(0, 10).map((t) => (
                <TouchableOpacity
                  key={t.id}
                  style={[
                    styles.threadCard,
                    activeThread === t.id && styles.threadCardActive,
                  ]}
                  onPress={() => selectThread(t.id)}
                >
                  <Text style={styles.threadTitle}>{t.title || 'New Chat'}</Text>
                  <Text style={styles.threadMeta}>
                    #{t.id} • {t.mode_title || t.mode_name || '-'}
                  </Text>
                </TouchableOpacity>
              ))
            )}
          </View>

          {/* Messages */}
          {activeThread && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Chat #{activeThread}</Text>
              <View style={styles.chatArea}>
                {messages.length === 0 ? (
                  <Text style={styles.mutedText}>Belum ada pesan.</Text>
                ) : (
                  messages.map((m, i) => (
                    <View
                      key={i}
                      style={[
                        styles.msgRow,
                        m.role === 'user' ? styles.msgUser : styles.msgAssistant,
                      ]}
                    >
                      <View
                        style={[
                          styles.msgBubble,
                          m.role === 'user' ? styles.msgBubbleUser : styles.msgBubbleAssistant,
                        ]}
                      >
                        <Text style={styles.msgLabel}>{m.role === 'user' ? 'User' : 'Xiaozhi'}</Text>
                        <Text style={styles.msgContent}>{m.content}</Text>
                        <Text style={styles.msgTime}>{m.created_at}</Text>
                      </View>
                    </View>
                  ))
                )}
              </View>
            </View>
          )}

          {/* Bottom spacing */}
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
    headerTitle: {
      fontSize: theme.fontSize.xl,
      fontWeight: '800',
      color: theme.colors.accentLight,
      fontFamily: theme.fonts.heading,
    },
    headerSubtitle: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: 2,
      fontFamily: theme.fonts.body,
    },
    headerButtons: { flexDirection: 'row', gap: theme.spacing.sm },
    headerBtn: {
      width: 40,
      height: 40,
      borderRadius: theme.radius.md,
      backgroundColor: theme.colors.surfaceLight,
      alignItems: 'center',
      justifyContent: 'center',
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.black : 'transparent',
    },
    logoutBtn: { backgroundColor: theme.isNeo ? '#fee2e2' : 'rgba(239,68,68,0.15)' },
    headerBtnText: { fontSize: 18 },
    scroll: { flex: 1 },
    section: { paddingHorizontal: theme.spacing.lg, marginTop: theme.spacing.xl },
    sectionTitle: {
      fontSize: theme.fontSize.md,
      fontWeight: '700',
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
      fontFamily: theme.fonts.heading,
    },
    mutedText: { fontSize: theme.fontSize.sm, color: theme.colors.textMuted, fontFamily: theme.fonts.body },

    // MCP Codes
    sectionHeaderRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    mcpManageBtn: {
      paddingHorizontal: theme.spacing.md,
      paddingVertical: 6,
      borderRadius: theme.radius.full,
      backgroundColor: theme.isNeo ? '#cffafe' : 'rgba(6,182,212,0.15)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(6,182,212,0.3)',
    },
    mcpManageBtnText: {
      fontSize: 11,
      fontWeight: '700',
      color: theme.isNeo ? theme.colors.black : '#67e8f9',
      fontFamily: theme.fonts.body,
    },
    mcpEmptyCard: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.15)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      padding: theme.spacing.lg,
      alignItems: 'center',
    },
    mcpSetupBtn: {
      marginTop: theme.spacing.md,
      paddingHorizontal: theme.spacing.lg,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.radius.full,
      backgroundColor: theme.isNeo ? '#a5f3fc' : '#0891b2',
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.black : 'transparent',
    },
    mcpSetupBtnText: {
      fontSize: theme.fontSize.xs,
      fontWeight: '700',
      color: theme.isNeo ? theme.colors.black : theme.colors.white,
      fontFamily: theme.fonts.heading,
    },
    codeCard: {
      borderRadius: theme.radius.md,
      padding: theme.spacing.md,
      marginBottom: theme.spacing.sm,
      borderWidth: theme.isNeo ? 2 : 1,
      ...theme.effects.cardShadow,
    },
    codeCardOnline: {
      backgroundColor: theme.isNeo ? '#ecfdf5' : 'rgba(16,185,129,0.06)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.2)',
    },
    codeCardOffline: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.2)',
      borderColor: theme.colors.panelBorder,
    },
    codeRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    codeLeftRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    codeRightRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
    codeText: {
      fontFamily: theme.fonts.mono,
      fontSize: theme.fontSize.sm,
      color: theme.colors.text,
      fontWeight: '600',
    },
    tokenBadge: {
      paddingHorizontal: 6,
      paddingVertical: 2,
      borderRadius: theme.radius.full,
      borderWidth: 1,
    },
    tokenBadgeOn: {
      backgroundColor: theme.isNeo ? '#cffafe' : 'rgba(6,182,212,0.1)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(6,182,212,0.2)',
    },
    tokenBadgeOff: {
      backgroundColor: theme.isNeo ? '#f3f4f6' : 'rgba(0,0,0,0.1)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(255,255,255,0.1)',
    },
    tokenBadgeText: {
      fontSize: 9,
      fontWeight: '700',
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },
    codeStatusText: { fontSize: 14 },
    codeStatusOn: {},
    codeStatusOff: {},
    statusDot: { width: 8, height: 8, borderRadius: 4 },
    dotOnline: { backgroundColor: theme.colors.emerald },
    dotOffline: { backgroundColor: theme.colors.red },
    codeError: {
      marginTop: 6,
      fontSize: 11,
      color: theme.isNeo ? '#dc2626' : '#fca5a5',
      fontFamily: theme.fonts.body,
    },
    codeStatus: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },

    // Active mode
    activeModeCard: {
      backgroundColor: theme.isNeo ? theme.colors.surface : 'rgba(59,130,246,0.1)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.panelBorder : 'rgba(59,130,246,0.3)',
      borderRadius: theme.radius.lg,
      padding: theme.spacing.lg,
      ...theme.effects.cardShadow,
    },
    activeModeTitle: {
      fontSize: theme.fontSize.lg,
      fontWeight: '700',
      color: theme.colors.accentLight,
      fontFamily: theme.fonts.heading,
    },
    activeModeName: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },

    // Mode chips
    modeChip: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.25)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.full,
      paddingHorizontal: theme.spacing.lg,
      paddingVertical: theme.spacing.sm,
      marginRight: theme.spacing.sm,
    },
    modeChipActive: {
      backgroundColor: theme.isNeo ? theme.colors.accentLight : 'rgba(59,130,246,0.2)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(59,130,246,0.5)',
    },
    modeChipText: { fontSize: theme.fontSize.sm, color: theme.colors.textSecondary, fontFamily: theme.fonts.body },
    modeChipTextActive: {
      color: theme.isNeo ? theme.colors.black : theme.colors.accentLight,
      fontWeight: '600',
      fontFamily: theme.fonts.heading,
    },

    // Threads
    threadCard: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.15)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      borderRadius: theme.radius.md,
      padding: theme.spacing.md,
      marginBottom: theme.spacing.sm,
      ...theme.effects.cardShadow,
    },
    threadCardActive: {
      backgroundColor: theme.isNeo ? theme.colors.surfaceLight : 'rgba(59,130,246,0.15)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(59,130,246,0.3)',
    },
    threadTitle: {
      fontSize: theme.fontSize.md,
      fontWeight: '600',
      color: theme.colors.text,
      fontFamily: theme.fonts.heading,
    },
    threadMeta: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textMuted,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },

    // Chat messages
    chatArea: {
      backgroundColor: theme.isNeo ? theme.colors.surface : theme.colors.surfaceLight,
      borderRadius: theme.radius.lg,
      padding: theme.spacing.md,
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.panelBorder : 'transparent',
    },
    msgRow: { marginVertical: 6 },
    msgUser: { alignItems: 'flex-end' },
    msgAssistant: { alignItems: 'flex-start' },
    msgBubble: {
      maxWidth: '80%',
      padding: theme.spacing.md,
      borderRadius: theme.radius.lg,
    },
    msgBubbleUser: {
      backgroundColor: theme.isNeo ? '#bbf7d0' : '#d7f0ff',
      borderBottomRightRadius: theme.radius.sm,
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.black : 'transparent',
    },
    msgBubbleAssistant: {
      backgroundColor: theme.isNeo ? theme.colors.panel : theme.colors.white,
      borderBottomLeftRadius: theme.radius.sm,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : theme.colors.panelBorder,
    },
    msgLabel: {
      fontSize: 11,
      fontWeight: '600',
      color: theme.colors.textSecondary,
      marginBottom: 4,
      fontFamily: theme.fonts.body,
    },
    msgContent: { fontSize: theme.fontSize.sm, color: theme.colors.text, fontFamily: theme.fonts.body },
    msgTime: {
      fontSize: 10,
      color: theme.colors.textMuted,
      textAlign: 'right',
      marginTop: 6,
      fontFamily: theme.fonts.mono,
    },
  });
