import React, { useEffect, useMemo, useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView, Alert } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { Theme, useTheme } from '../theme/theme';
import { apiGetDeviceDetail, apiSetDeviceAlias, apiUnpairDevice, DeviceInfo } from '../api/client';
import { deviceStore } from '../stores/deviceStore';

export default function DeviceDetailScreen() {
  const { theme } = useTheme();
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const deviceId = route.params?.deviceId as string;
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [myDeviceId, setMyDeviceId] = useState('');
  const [alias, setAlias] = useState('');

  const styles = useMemo(() => createStyles(theme), [theme]);

  async function loadDetail() {
    const detail = await apiGetDeviceDetail(deviceId);
    setDevice(detail);
    setAlias(detail.alias || detail.device_name || '');
  }

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const ownDeviceId = await deviceStore.getDeviceId();
        if (active) {
          setMyDeviceId(ownDeviceId);
        }
        await loadDetail();
      } catch (e: any) {
        if (active) {
          Alert.alert('Error', e.message || 'Gagal memuat device');
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [deviceId]);

  async function saveAlias() {
    if (!isOnline) {
      Alert.alert('Tunggu Perangkat Online', 'Nama device baru bisa diubah setelah perangkat tujuan tersambung.');
      return;
    }
    if (!alias.trim()) {
      Alert.alert('Error', 'Nama tidak boleh kosong');
      return;
    }
    await apiSetDeviceAlias(deviceId, alias.trim());
    await loadDetail();
    Alert.alert('OK', 'Nama device disimpan');
  }

  async function resetAlias() {
    await apiSetDeviceAlias(deviceId, '');
    await loadDetail();
    Alert.alert('OK', 'Nama device dikembalikan ke nama bawaan.');
  }

  function confirmUnpair() {
    Alert.alert(
      'Hapus Pair',
      'Device pair ini akan dihapus dari daftar monitor Anda. Data device pemilik aslinya tidak ikut terhapus.',
      [
        { text: 'Batal', style: 'cancel' },
        {
          text: 'Hapus Pair',
          style: 'destructive',
          onPress: async () => {
            try {
              await apiUnpairDevice(deviceId);
              Alert.alert('OK', 'Device pair berhasil dihapus dari monitor Anda.');
              navigation.goBack();
            } catch (e: any) {
              Alert.alert('Error', e.message || 'Gagal menghapus pair');
            }
          },
        },
      ],
    );
  }

  const addr = device?.address_full || [device?.address_street, device?.address_area, device?.address_city].filter(Boolean).join(', ');
  const isOnline = !!device?.is_online;
  const isOwnDevice = !!myDeviceId && deviceId === myDeviceId;
  const lastSeen = device?.last_seen_at
    ? new Date(device.last_seen_at).toLocaleString('id-ID')
    : 'Belum ada sinkron terbaru';

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Detail Device</Text>
        <Text style={styles.subtitle}>{deviceId}</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Nama Device</Text>
          <TextInput
            style={styles.input}
            value={alias}
            onChangeText={setAlias}
            placeholder="Nama lokasi"
            placeholderTextColor={theme.colors.textMuted}
          />
          <Text style={styles.meta}>
            {isOwnDevice
              ? 'Ini perangkat utama milik Anda. Nama tampilan bisa diatur dari sini.'
              : isOnline
              ? 'Device pair sudah online. Anda bisa ganti nama, reset nama, atau hapus pair dari sini.'
              : 'Tunggu sampai perangkat tujuan online dulu baru bisa diberi nama.'}
          </Text>
          <TouchableOpacity
            style={[styles.primaryBtn, !isOnline && styles.disabledBtn]}
            onPress={saveAlias}
            disabled={!isOnline}
          >
            <Text style={styles.btnText}>{isOnline ? 'Simpan Nama' : 'Menunggu Online'}</Text>
          </TouchableOpacity>
          <View style={styles.btnRow}>
            <TouchableOpacity style={styles.secondaryBtn} onPress={resetAlias}>
              <Text style={styles.secondaryBtnText}>Reset Nama</Text>
            </TouchableOpacity>
            {!isOwnDevice ? (
              <TouchableOpacity style={styles.dangerBtn} onPress={confirmUnpair}>
                <Text style={styles.dangerBtnText}>Hapus Pair</Text>
              </TouchableOpacity>
            ) : null}
          </View>
        </View>

        <View style={styles.card}>
          <View style={styles.rowBetween}>
            <Text style={styles.cardTitle}>Alamat</Text>
            <View style={[styles.statusPill, isOnline ? styles.statusOnline : styles.statusOffline]}>
              <Text style={styles.statusPillText}>{isOnline ? 'Online' : 'Offline'}</Text>
            </View>
          </View>
          <Text style={styles.value}>{addr || '-'}</Text>
          <Text style={styles.meta}>Update terakhir: {lastSeen}</Text>
          <Text style={styles.meta}>Koordinat mentah disembunyikan. Lokasi ditampilkan sebagai alamat detail.</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Status</Text>
          <Text style={styles.value}>Baterai: {device?.battery_level || '-'}%</Text>
          <Text style={styles.meta}>Charging: {device?.charging_type || '-'}</Text>
          <Text style={styles.meta}>Jaringan: {device?.network_type || '-'} / {device?.carrier || '-'}</Text>
        </View>
      </ScrollView>
    </View>
  );
}

const createStyles = (theme: Theme) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: theme.colors.bg },
    content: { padding: theme.spacing.lg },
    title: { fontSize: theme.fontSize.xl, fontWeight: '800', color: theme.colors.accentLight, fontFamily: theme.fonts.heading },
    subtitle: { fontSize: theme.fontSize.xs, color: theme.colors.textMuted, marginTop: 4, fontFamily: theme.fonts.mono },
    card: {
      marginTop: theme.spacing.md,
      padding: theme.spacing.lg,
      borderRadius: theme.radius.lg,
      backgroundColor: theme.colors.surface,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      ...theme.effects.cardShadow,
    },
    cardTitle: { fontSize: theme.fontSize.sm, color: theme.colors.textMuted, marginBottom: 8, fontFamily: theme.fonts.body },
    rowBetween: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      gap: theme.spacing.sm,
    },
    input: {
      backgroundColor: theme.colors.surfaceLight,
      borderRadius: theme.radius.md,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      color: theme.colors.text,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      fontFamily: theme.fonts.body,
    },
    primaryBtn: {
      marginTop: theme.spacing.md,
      backgroundColor: theme.colors.accent,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
    },
    btnRow: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
      marginTop: theme.spacing.sm,
    },
    disabledBtn: {
      opacity: 0.45,
    },
    btnText: { color: theme.colors.white, fontWeight: '700', fontFamily: theme.fonts.body },
    secondaryBtn: {
      flex: 1,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
      backgroundColor: theme.colors.surfaceLight,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    secondaryBtnText: {
      color: theme.colors.text,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    dangerBtn: {
      flex: 1,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
      backgroundColor: theme.isNeo ? '#fee2e2' : 'rgba(239,68,68,0.12)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(239,68,68,0.22)',
    },
    dangerBtnText: {
      color: theme.isNeo ? theme.colors.black : '#dc2626',
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    statusPill: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: 6,
      borderRadius: theme.radius.full,
      borderWidth: theme.isNeo ? 2 : 1,
    },
    statusOnline: {
      backgroundColor: theme.isNeo ? '#dcfce7' : 'rgba(16,185,129,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.24)',
    },
    statusOffline: {
      backgroundColor: theme.isNeo ? '#fef3c7' : 'rgba(245,158,11,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(245,158,11,0.24)',
    },
    statusPillText: {
      color: theme.colors.text,
      fontSize: 11,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    value: { color: theme.colors.text, fontSize: theme.fontSize.sm, fontFamily: theme.fonts.body },
    meta: { color: theme.colors.textMuted, fontSize: theme.fontSize.xs, marginTop: 4, fontFamily: theme.fonts.body },
  });
