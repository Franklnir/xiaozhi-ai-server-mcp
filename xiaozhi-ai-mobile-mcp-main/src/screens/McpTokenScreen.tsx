import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  Animated,
  Easing,
  Linking,
} from 'react-native';
import { Theme, useTheme } from '../theme/theme';
import {
  apiGetMyCodes,
  apiTestMcpToken,
  apiCreateMyMcpToken,
  apiUpdateMyMcpToken,
  apiClearMyMcpToken,
  McpCodeInfo,
} from '../api/client';

interface McpTokenScreenProps {
  onBack: () => void;
}

export default function McpTokenScreen({ onBack }: McpTokenScreenProps) {
  const { theme } = useTheme();
  const [codes, setCodes] = useState<McpCodeInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const [selectedCodeId, setSelectedCodeId] = useState<number | null>(null);
  const [testStatus, setTestStatus] = useState<{ text: string; ok: boolean | null }>({
    text: 'Belum dites.',
    ok: null,
  });
  const [busy, setBusy] = useState(false);

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

  const loadCodes = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiGetMyCodes();
      setCodes(data || []);
    } catch (e: any) {
      console.warn('Load MCP codes error:', e.message);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadCodes();
  }, [loadCodes]);

  async function handleTest() {
    const token = tokenInput.trim();
    if (!token) {
      setTestStatus({ text: 'Token / URL kosong.', ok: false });
      return;
    }
    setBusy(true);
    setTestStatus({ text: 'Testing koneksi...', ok: null });
    try {
      const res = await apiTestMcpToken(token);
      setTestStatus({
        text: res.ok ? '✅ Connected! Token valid.' : `❌ Gagal: ${res.error || 'tidak bisa connect'}`,
        ok: !!res.ok,
      });
    } catch (e: any) {
      setTestStatus({ text: `❌ Error: ${e.message}`, ok: false });
    }
    setBusy(false);
  }

  async function handleSave() {
    if (testStatus.ok !== true) {
      Alert.alert('Test Dulu', 'Tolong test koneksi dulu sebelum simpan token.');
      return;
    }
    const token = tokenInput.trim();
    if (!token) {
      setTestStatus({ text: 'Token / URL kosong.', ok: false });
      return;
    }
    setBusy(true);
    try {
      if (selectedCodeId) {
        await apiUpdateMyMcpToken(selectedCodeId, token);
        Alert.alert('Berhasil', 'Token berhasil diperbarui. MCP worker akan reconnect otomatis.');
      } else {
        const created = await apiCreateMyMcpToken(token);
        Alert.alert('Berhasil', `Code baru dibuat: ${created.code || '-'}. MCP worker akan connect otomatis.`);
      }
      setTokenInput('');
      setSelectedCodeId(null);
      setTestStatus({ text: 'Tersimpan.', ok: true });
      await loadCodes();
    } catch (e: any) {
      Alert.alert('Error', e.message);
      setTestStatus({ text: `Error: ${e.message}`, ok: false });
    }
    setBusy(false);
  }

  async function handleClear() {
    if (!selectedCodeId) {
      Alert.alert('Pilih Code', 'Pilih code yang ingin dihapus tokennya terlebih dahulu.');
      return;
    }
    Alert.alert('Konfirmasi', 'Yakin hapus token dari code ini? MCP worker akan disconnect.', [
      { text: 'Batal', style: 'cancel' },
      {
        text: 'Hapus',
        style: 'destructive',
        onPress: async () => {
          setBusy(true);
          try {
            await apiClearMyMcpToken(selectedCodeId);
            Alert.alert('Berhasil', 'Token berhasil dihapus.');
            setTokenInput('');
            setSelectedCodeId(null);
            setTestStatus({ text: 'Token dihapus.', ok: true });
            await loadCodes();
          } catch (e: any) {
            Alert.alert('Error', e.message);
          }
          setBusy(false);
        },
      },
    ]);
  }

  function selectCode(code: McpCodeInfo) {
    setSelectedCodeId(code.id);
    setTokenInput('');
    setTestStatus({ text: `Code ${code.code} dipilih. Paste token baru untuk update.`, ok: null });
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backBtnText}>← Kembali</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>🔗 MCP Token</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        <Animated.View style={contentAnimStyle}>
          {/* Setup Guide Card */}
          <View style={styles.guideCard}>
            <Text style={styles.guideTitle}>📖 Cara Mendapatkan MCP Endpoint</Text>
            <View style={styles.guideSteps}>
              <Text style={styles.guideStep}>
                1. Login ke{' '}
                <Text
                  style={styles.guideLink}
                  onPress={() => Linking.openURL('https://xiaozhi.me')}
                >
                  xiaozhi.me
                </Text>
                {' → masuk Console'}
              </Text>
              <Text style={styles.guideStep}>
                2. Buka konfigurasi <Text style={styles.guideBold}>Smart Agent (智能体)</Text>
              </Text>
              <Text style={styles.guideStep}>
                3. Di halaman <Text style={styles.guideBold}>Role Config</Text>, lihat kanan bawah
              </Text>
              <Text style={styles.guideStep}>
                4. Copy URL <Text style={styles.guideBold}>MCP Endpoint</Text>
              </Text>
              <Text style={styles.guideStep}>
                5. Paste di bawah → <Text style={styles.guideBold}>Test → Simpan</Text>
              </Text>
            </View>
            <Text style={styles.guideNote}>
              ⚠️ Setiap smart agent punya token unik. Jika token berubah di xiaozhi.me, update juga di sini.
            </Text>
          </View>

          {/* My Codes List */}
          <View style={styles.card}>
            <View style={styles.cardHeaderRow}>
              <Text style={styles.cardTitle}>Status Koneksi MCP</Text>
              <TouchableOpacity onPress={loadCodes} style={styles.reloadBtn}>
                <Text style={styles.reloadBtnText}>{loading ? '...' : '🔄 Reload'}</Text>
              </TouchableOpacity>
            </View>

            {codes.length === 0 ? (
              <Text style={styles.emptyText}>
                Belum ada code MCP. Paste token dari xiaozhi.me di bawah untuk membuat yang baru.
              </Text>
            ) : (
              codes.map((c) => {
                const ok = !!c.is_connected;
                const isSelected = selectedCodeId === c.id;
                return (
                  <TouchableOpacity
                    key={c.id}
                    style={[
                      styles.codeCard,
                      ok ? styles.codeCardOnline : styles.codeCardOffline,
                      isSelected && styles.codeCardSelected,
                    ]}
                    onPress={() => selectCode(c)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.codeHeaderRow}>
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
                        <Text style={[styles.statusLabel, ok ? styles.statusOnline : styles.statusOffline]}>
                          {ok ? '🟢 Connected' : '🔴 Disconnected'}
                        </Text>
                      </View>
                    </View>

                    <View style={styles.codeMetaRow}>
                      {c.last_ok_at ? (
                        <Text style={styles.codeMeta}>Last OK: {String(c.last_ok_at)}</Text>
                      ) : null}
                      {c.last_err_at ? (
                        <Text style={styles.codeMeta}>Last Err: {String(c.last_err_at)}</Text>
                      ) : null}
                    </View>

                    {c.last_error ? (
                      <Text style={styles.codeError} numberOfLines={2}>
                        ⚠️ {c.last_error}
                      </Text>
                    ) : null}

                    {isSelected && (
                      <View style={styles.selectedIndicator}>
                        <Text style={styles.selectedText}>✓ Dipilih</Text>
                      </View>
                    )}
                  </TouchableOpacity>
                );
              })
            )}
          </View>

          {/* Token Manager */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Token Manager</Text>
            <Text style={styles.cardSubtitle}>
              {selectedCodeId
                ? `Updating code ID #${selectedCodeId}. Paste token baru dari xiaozhi.me.`
                : 'Paste MCP endpoint URL untuk membuat code baru.'}
            </Text>

            <TextInput
              style={styles.tokenInput}
              value={tokenInput}
              onChangeText={(text) => {
                setTokenInput(text);
                setTestStatus({ text: 'Belum dites.', ok: null });
              }}
              placeholder="wss://api.xiaozhi.me/mcp/?token=..."
              placeholderTextColor={theme.colors.textMuted}
              multiline
              numberOfLines={3}
              autoCapitalize="none"
              autoCorrect={false}
            />

            <View
              style={[
                styles.statusBox,
                testStatus.ok === true
                  ? styles.statusBoxOk
                  : testStatus.ok === false
                  ? styles.statusBoxErr
                  : styles.statusBoxNeutral,
              ]}
            >
              <Text style={styles.statusBoxText}>{testStatus.text}</Text>
            </View>

            <View style={styles.actionRow}>
              <TouchableOpacity
                style={[styles.actionBtn, styles.actionBtnTest]}
                onPress={handleTest}
                disabled={busy}
              >
                <Text style={styles.actionBtnText}>{busy ? '...' : '🔌 Test'}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.actionBtn, styles.actionBtnSave]}
                onPress={handleSave}
                disabled={busy}
              >
                <Text style={styles.actionBtnText}>{busy ? '...' : '💾 Simpan'}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.actionBtn, styles.actionBtnClear]}
                onPress={handleClear}
                disabled={busy}
              >
                <Text style={styles.actionBtnText}>{busy ? '...' : '🗑️ Hapus'}</Text>
              </TouchableOpacity>
            </View>

            {selectedCodeId && (
              <TouchableOpacity
                style={styles.deselectBtn}
                onPress={() => {
                  setSelectedCodeId(null);
                  setTokenInput('');
                  setTestStatus({ text: 'Mode: buat code baru.', ok: null });
                }}
              >
                <Text style={styles.deselectBtnText}>↩ Batal pilih code (buat baru)</Text>
              </TouchableOpacity>
            )}
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

    // Guide Card
    guideCard: {
      marginBottom: theme.spacing.md,
      padding: theme.spacing.lg,
      borderRadius: theme.radius.lg,
      backgroundColor: theme.isNeo ? '#ecfdf5' : 'rgba(6,182,212,0.08)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(6,182,212,0.2)',
    },
    guideTitle: {
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      color: theme.isNeo ? '#065f46' : '#67e8f9',
      fontFamily: theme.fonts.heading,
      marginBottom: theme.spacing.sm,
    },
    guideSteps: { gap: 4 },
    guideStep: {
      fontSize: theme.fontSize.xs,
      color: theme.isNeo ? '#064e3b' : theme.colors.textSecondary,
      lineHeight: 20,
      fontFamily: theme.fonts.body,
    },
    guideBold: {
      fontWeight: '700',
      color: theme.isNeo ? '#064e3b' : theme.colors.text,
    },
    guideLink: {
      color: theme.isNeo ? '#0891b2' : '#67e8f9',
      textDecorationLine: 'underline',
      fontWeight: '600',
    },
    guideNote: {
      marginTop: theme.spacing.sm,
      fontSize: 11,
      color: theme.isNeo ? '#6b7280' : theme.colors.textMuted,
      fontFamily: theme.fonts.body,
    },

    // Card
    card: {
      marginBottom: theme.spacing.md,
      padding: theme.spacing.lg,
      borderRadius: theme.radius.lg,
      backgroundColor: theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      ...theme.effects.cardShadow,
    },
    cardHeaderRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
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
      marginBottom: theme.spacing.md,
      fontFamily: theme.fonts.body,
    },
    reloadBtn: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: 6,
      borderRadius: theme.radius.md,
      backgroundColor: theme.colors.surfaceLight,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    reloadBtnText: {
      fontSize: 12,
      fontWeight: '600',
      color: theme.colors.text,
      fontFamily: theme.fonts.body,
    },
    emptyText: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textMuted,
      fontFamily: theme.fonts.body,
      lineHeight: 20,
    },

    // Code cards
    codeCard: {
      padding: theme.spacing.md,
      borderRadius: theme.radius.md,
      marginBottom: theme.spacing.sm,
      borderWidth: theme.isNeo ? 2 : 1,
    },
    codeCardOnline: {
      backgroundColor: theme.isNeo ? '#ecfdf5' : 'rgba(16,185,129,0.06)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.2)',
    },
    codeCardOffline: {
      backgroundColor: theme.isNeo ? theme.colors.panel : 'rgba(0,0,0,0.15)',
      borderColor: theme.colors.panelBorder,
    },
    codeCardSelected: {
      borderColor: theme.isNeo ? '#2563eb' : theme.colors.accentLight,
      borderWidth: 2,
    },
    codeHeaderRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    codeLeftRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    codeRightRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
    statusDot: { width: 8, height: 8, borderRadius: 4 },
    dotOnline: { backgroundColor: '#34d399' },
    dotOffline: { backgroundColor: '#f87171' },
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
      borderWidth: theme.isNeo ? 1.5 : 1,
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
      fontSize: 10,
      fontWeight: '700',
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },
    statusLabel: { fontSize: 11, fontWeight: '600', fontFamily: theme.fonts.body },
    statusOnline: { color: theme.isNeo ? '#065f46' : '#6ee7b7' },
    statusOffline: { color: theme.isNeo ? '#991b1b' : '#fca5a5' },
    codeMetaRow: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 8,
      marginTop: 6,
    },
    codeMeta: {
      fontSize: 10,
      color: theme.colors.textMuted,
      fontFamily: theme.fonts.mono,
    },
    codeError: {
      marginTop: 6,
      fontSize: 11,
      color: theme.isNeo ? '#dc2626' : '#fca5a5',
      fontFamily: theme.fonts.body,
    },
    selectedIndicator: {
      marginTop: 8,
      paddingVertical: 4,
      paddingHorizontal: 8,
      borderRadius: theme.radius.full,
      backgroundColor: theme.isNeo ? '#dbeafe' : 'rgba(59,130,246,0.15)',
      alignSelf: 'flex-start',
    },
    selectedText: {
      fontSize: 11,
      fontWeight: '700',
      color: theme.isNeo ? '#1e40af' : '#93c5fd',
      fontFamily: theme.fonts.body,
    },

    // Token Input
    tokenInput: {
      backgroundColor: theme.colors.surfaceLight,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.md,
      color: theme.colors.text,
      fontSize: theme.fontSize.sm,
      fontFamily: theme.fonts.mono,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      minHeight: 80,
      textAlignVertical: 'top',
    },

    // Status box
    statusBox: {
      marginTop: theme.spacing.sm,
      padding: theme.spacing.sm,
      borderRadius: theme.radius.md,
      borderWidth: theme.isNeo ? 2 : 1,
    },
    statusBoxOk: {
      backgroundColor: theme.isNeo ? '#dcfce7' : 'rgba(16,185,129,0.08)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.2)',
    },
    statusBoxErr: {
      backgroundColor: theme.isNeo ? '#fee2e2' : 'rgba(239,68,68,0.08)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(239,68,68,0.2)',
    },
    statusBoxNeutral: {
      backgroundColor: theme.isNeo ? '#f3f4f6' : 'rgba(255,255,255,0.05)',
      borderColor: theme.colors.panelBorder,
    },
    statusBoxText: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      fontFamily: theme.fonts.body,
    },

    // Action buttons
    actionRow: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
      marginTop: theme.spacing.md,
    },
    actionBtn: {
      flex: 1,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
      borderWidth: theme.isNeo ? 2 : 0,
      borderColor: theme.isNeo ? theme.colors.black : 'transparent',
    },
    actionBtnTest: {
      backgroundColor: theme.isNeo ? theme.colors.panel : theme.colors.surfaceLight,
    },
    actionBtnSave: {
      backgroundColor: theme.isNeo ? '#a5f3fc' : '#0891b2',
    },
    actionBtnClear: {
      backgroundColor: theme.isNeo ? '#fecdd3' : '#dc2626',
    },
    actionBtnText: {
      fontSize: theme.fontSize.xs,
      fontWeight: '700',
      color: theme.isNeo ? theme.colors.black : theme.colors.white,
      fontFamily: theme.fonts.heading,
    },
    deselectBtn: {
      marginTop: theme.spacing.sm,
      paddingVertical: theme.spacing.sm,
      alignItems: 'center',
    },
    deselectBtnText: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.accentLight,
      fontWeight: '600',
      fontFamily: theme.fonts.body,
    },
  });
