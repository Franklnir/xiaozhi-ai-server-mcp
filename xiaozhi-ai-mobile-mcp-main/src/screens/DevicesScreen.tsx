import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Animated,
  Easing,
} from 'react-native';
import { useIsFocused, useNavigation } from '@react-navigation/native';
import { WebView } from 'react-native-webview';
import QRCode from 'react-native-qrcode-svg';
import { Theme, useTheme } from '../theme/theme';
import {
  apiClaimPairToken,
  apiCreatePairToken,
  apiGetDevices,
  DeviceInfo,
  DeviceLocationInfo,
  apiGetDeviceLocations,
} from '../api/client';
import { deviceStore } from '../stores/deviceStore';
import {
  isTrackingRunning,
  sendHeartbeat,
  startTracking,
  stopTracking,
  syncDeviceSnapshot,
  TRACKING_INTERVAL_LABEL,
  TRACKING_INTERVAL_MS,
} from '../services/deviceService';

function formatDeviceAddress(device?: DeviceInfo | null) {
  return (
    device?.address_full ||
    [device?.address_street, device?.address_area, device?.address_city].filter(Boolean).join(', ') ||
    'Alamat belum tersedia'
  );
}

function formatLocationAddress(location?: DeviceLocationInfo | null) {
  return (
    location?.address_full ||
    [location?.address_street, location?.address_area, location?.address_city].filter(Boolean).join(', ') ||
    'Alamat belum tersedia'
  );
}

function formatSeenAt(value?: string) {
  if (!value) {
    return 'Belum ada sinkron terbaru';
  }
  try {
    return new Date(value).toLocaleString('id-ID');
  } catch {
    return value;
  }
}

export default function DevicesScreen() {
  const { theme } = useTheme();
  const navigation = useNavigation<any>();
  const isFocused = useIsFocused();
  const [deviceId, setDeviceId] = useState('');
  const [deviceToken, setDeviceToken] = useState('');
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [pairToken, setPairToken] = useState('');
  const [pairExpires, setPairExpires] = useState('');
  const [manualToken, setManualToken] = useState('');
  const [selectedMapDeviceId, setSelectedMapDeviceId] = useState('');
  const [mapDeviceMenuOpen, setMapDeviceMenuOpen] = useState(false);
  const [locationHistory, setLocationHistory] = useState<DeviceLocationInfo[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [trackingEnabled, setTrackingEnabled] = useState(false);
  const [bootError, setBootError] = useState('');
  const [pageScrollEnabled, setPageScrollEnabled] = useState(true);

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

  const loadDevices = useCallback(async () => {
    try {
      const list = await apiGetDevices();
      setDevices(list || []);
      setSelectedMapDeviceId((current) => {
        const available = list || [];
        if (current && available.some((item) => item.device_id === current)) {
          return current;
        }
        if (deviceId && available.some((item) => item.device_id === deviceId)) {
          return deviceId;
        }
        return available[0]?.device_id || current;
      });
    } catch {
      // ignore
    }
  }, [deviceId]);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        setBootError('');
        const reg = await syncDeviceSnapshot();
        if (!active) return;
        setDeviceId(reg.device_id);
        setDeviceToken(reg.device_token);
        setSelectedMapDeviceId((current) => current || reg.device_id);
        const enabled = await isTrackingRunning();
        await deviceStore.setTrackingEnabled(enabled);
        if (!active) return;
        setTrackingEnabled(enabled);
        await loadDevices();
      } catch (e: any) {
        if (active) {
          setBootError(e?.message || 'Gagal menyiapkan perangkat.');
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [loadDevices]);

  useEffect(() => {
    if (isFocused) loadDevices();
  }, [isFocused, loadDevices]);

  useEffect(() => {
    const timer = setInterval(() => {
      loadDevices().catch(() => {});
    }, TRACKING_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [loadDevices]);

  useEffect(() => {
    if (!selectedMapDeviceId) {
      setLocationHistory([]);
      return;
    }

    let active = true;
    (async () => {
      setLoadingHistory(true);
      try {
        const history = await apiGetDeviceLocations(selectedMapDeviceId, 500);
        if (!active) return;
        setLocationHistory((history || []).slice().reverse());
      } catch {
        if (active) {
          setLocationHistory([]);
        }
      } finally {
        if (active) {
          setLoadingHistory(false);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, [selectedMapDeviceId]);

  async function toggleTracking() {
    try {
      if (trackingEnabled) {
        await stopTracking();
        setTrackingEnabled(false);
      } else {
        await startTracking();
        setTrackingEnabled(true);
      }
      await loadDevices();
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Gagal mengubah tracking');
    }
  }

  async function generatePairToken() {
    try {
      const res = await apiCreatePairToken(deviceId, deviceToken);
      setPairToken(res.pair_token || '');
      setPairExpires(res.expires_at || '');
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Gagal membuat QR');
    }
  }

  async function claimToken() {
    if (!manualToken.trim()) {
      Alert.alert('Error', 'Pair token wajib diisi');
      return;
    }
    try {
      const result = await apiClaimPairToken(manualToken.trim());
      setManualToken('');
      await loadDevices();
      if (result.device?.is_online) {
        Alert.alert('Pair Berhasil', 'Perangkat tujuan sudah tersambung. Sekarang Anda bisa beri nama device.', [
          { text: 'Nanti' },
          {
            text: 'Beri Nama',
            onPress: () =>
              navigation.navigate('DeviceDetail' as never, { deviceId: result.device_id } as never),
          },
        ]);
      } else {
        Alert.alert(
          'Pair Berhasil',
          'Perangkat sudah dipairkan. Tunggu sampai statusnya tersambung dulu, lalu beri nama dari detail device.',
        );
      }
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Gagal menambah device');
    }
  }

  async function manualHeartbeat() {
    try {
      await sendHeartbeat();
      await loadDevices();
      Alert.alert('OK', 'Data perangkat diperbarui');
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Gagal update data');
    }
  }

  const myDevice = devices.find((d) => d.device_id === deviceId);
  const myDeviceOnline = !!myDevice?.is_online;
  const selectedMapDevice =
    devices.find((item) => item.device_id === selectedMapDeviceId) ||
    myDevice ||
    devices[0] ||
    null;
  const selectedRoutePoints = locationHistory.filter(
    (point) => point.latitude != null && point.longitude != null,
  );
  const selectedRouteLabel = selectedMapDevice
    ? selectedMapDevice.device_id === deviceId
      ? 'Lokasi milik saya'
      : 'Lokasi device dipairkan'
    : '-';

  const mapHtml = useMemo(() => {
    const currentMarker =
      selectedMapDevice?.latitude != null && selectedMapDevice?.longitude != null
        ? {
            name: selectedMapDevice.alias || selectedMapDevice.device_name || selectedMapDevice.device_id,
            lat: selectedMapDevice.latitude,
            lon: selectedMapDevice.longitude,
            addr: formatDeviceAddress(selectedMapDevice),
            status: selectedMapDevice.is_online ? 'Online' : 'Offline',
          }
        : null;

    const route = selectedRoutePoints.map((point) => ({
      lat: point.latitude,
      lon: point.longitude,
      addr: formatLocationAddress(point),
      at: point.recorded_at,
    }));

    return `
      <!DOCTYPE html>
      <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
          <style>
            html, body, #map { height: 100%; margin: 0; padding: 0; }
          </style>
        </head>
        <body>
          <div id="map"></div>
          <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
          <script>
            const route = ${JSON.stringify(route)};
            const currentMarker = ${JSON.stringify(currentMarker)};
            const map = L.map('map', {
              zoomControl: true,
              dragging: true,
              scrollWheelZoom: true,
              doubleClickZoom: true,
              touchZoom: true,
              boxZoom: true
            }).setView([0,0], 2);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
              attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);
            L.control.scale({ imperial: false }).addTo(map);
            const layers = [];
            if (route.length > 1) {
              const polyline = L.polyline(route.map(p => [p.lat, p.lon]), {
                color: '#2563eb',
                weight: 5,
                opacity: 0.9
              }).addTo(map);
              layers.push(polyline);

              const start = route[0];
              const end = route[route.length - 1];
              const startMarker = L.circleMarker([start.lat, start.lon], {
                radius: 7,
                color: '#16a34a',
                fillColor: '#22c55e',
                fillOpacity: 1
              }).addTo(map).bindPopup('<b>Mulai</b><br>' + (start.addr || '-') + '<br>' + (start.at || '-'));
              const endMarker = L.circleMarker([end.lat, end.lon], {
                radius: 7,
                color: '#dc2626',
                fillColor: '#ef4444',
                fillOpacity: 1
              }).addTo(map).bindPopup('<b>Terakhir</b><br>' + (end.addr || '-') + '<br>' + (end.at || '-'));
              layers.push(startMarker);
              layers.push(endMarker);
            }

            if (currentMarker) {
              const marker = L.marker([currentMarker.lat, currentMarker.lon]).addTo(map).bindPopup(
                '<b>' + currentMarker.name + '</b><br>' + (currentMarker.addr || '-') + '<br><span style="color:#475569;font-size:12px;">' + currentMarker.status + '</span>'
              );
              layers.push(marker);
            }

            if (layers.length) {
              const bounds = L.featureGroup(layers).getBounds();
              map.fitBounds(bounds.pad(0.18), { maxZoom: 16 });
            }
          </script>
        </body>
      </html>
    `;
  }, [selectedMapDevice, selectedRoutePoints]);

  const animStyle = {
    opacity: enterAnim,
    transform: [
      {
        translateY: enterAnim.interpolate({ inputRange: [0, 1], outputRange: [12, 0] }),
      },
    ],
  };

  return (
    <View style={styles.container}>
      <ScrollView scrollEnabled={pageScrollEnabled}>
        <Animated.View style={animStyle}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Perangkat Saya</Text>
            <Text style={styles.headerSubtitle}>
              Snapshot lokasi dan data HP disinkronkan tiap {TRACKING_INTERVAL_LABEL}
            </Text>
          </View>

          {bootError ? (
            <View style={styles.warnCard}>
              <Text style={styles.warnTitle}>Perlu Perhatian</Text>
              <Text style={styles.warnText}>{bootError}</Text>
            </View>
          ) : null}

          <View style={styles.card}>
            <View style={styles.cardTopRow}>
              <View>
                <Text style={styles.cardTitle}>Perangkat Utama</Text>
                <Text style={styles.cardValue}>{deviceId || '-'}</Text>
              </View>
              <View style={[styles.statusPill, myDeviceOnline ? styles.statusOnlinePill : styles.statusOfflinePill]}>
                <Text style={styles.statusPillText}>{myDeviceOnline ? 'Online' : 'Offline'}</Text>
              </View>
            </View>
            <Text style={styles.cardMeta}>{formatDeviceAddress(myDevice)}</Text>
            <Text style={styles.cardMeta}>Update terakhir: {formatSeenAt(myDevice?.last_seen_at)}</Text>
            <Text style={styles.cardMeta}>
              Baterai: {myDevice?.battery_level ?? '-'}% • {myDevice?.battery_status || '-'}
            </Text>
            <Text style={styles.cardMeta}>
              Jaringan: {myDevice?.network_type || '-'} / {myDevice?.carrier || '-'}
            </Text>
            <View style={styles.infoBanner}>
              <Text style={styles.infoBannerText}>
                Izin background sekarang dipisah ke menu Akun & Pengaturan supaya flow izin lebih stabil dan tidak bikin force close.
              </Text>
            </View>
            <View style={styles.btnRow}>
              <TouchableOpacity style={styles.secondaryBtn} onPress={toggleTracking}>
                <Text style={styles.secondaryBtnText}>
                  {trackingEnabled ? 'Matikan Monitor' : 'Aktifkan Monitor'}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.primaryBtn} onPress={manualHeartbeat}>
                <Text style={styles.btnText}>Sync Now</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Generate QR</Text>
            <TouchableOpacity
              style={[styles.primaryBtn, !deviceToken && styles.disabledBtn]}
              onPress={generatePairToken}
              disabled={!deviceToken}
            >
              <Text style={styles.btnText}>Buat QR Pair</Text>
            </TouchableOpacity>
            {pairToken ? (
              <View style={styles.qrBox}>
                <QRCode value={pairToken} size={160} />
                <Text style={styles.qrText}>{pairToken}</Text>
                <Text style={styles.qrMeta}>Expired: {pairExpires}</Text>
              </View>
            ) : null}
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Tambah Device</Text>
            <Text style={styles.cardMeta}>
              Pair dulu perangkat tujuan. Setelah statusnya benar-benar tersambung, baru beri nama device dari halaman detail.
            </Text>
            <TextInput
              style={[styles.input, { marginTop: 10 }]}
              value={manualToken}
              onChangeText={setManualToken}
              placeholder="Pair token"
              placeholderTextColor={theme.colors.textMuted}
            />
            <View style={styles.btnRow}>
              <TouchableOpacity
                style={styles.secondaryBtn}
                onPress={() => navigation.navigate('QrScanner' as never)}
              >
                <Text style={styles.secondaryBtnText}>Scan QR</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.primaryBtn} onPress={claimToken}>
                <Text style={styles.btnText}>Tambah</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Peta Lokasi</Text>
            <Text style={styles.cardMeta}>
              Default menampilkan peta lokasi milik Anda. Buka dropdown di bawah untuk pindah ke device lain yang sudah dipairkan.
            </Text>
            <TouchableOpacity
              style={styles.dropdownTrigger}
              onPress={() => setMapDeviceMenuOpen((open) => !open)}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.dropdownLabel}>Pilih Lokasi Device</Text>
                <Text style={styles.dropdownValue}>
                  {selectedMapDevice
                    ? `${selectedMapDevice.alias || selectedMapDevice.device_name || selectedMapDevice.device_id}`
                    : 'Belum ada device'}
                </Text>
                <Text style={styles.dropdownHint}>{selectedRouteLabel}</Text>
              </View>
              <Text style={styles.dropdownChevron}>{mapDeviceMenuOpen ? '▲' : '▼'}</Text>
            </TouchableOpacity>
            {mapDeviceMenuOpen ? (
              <View style={styles.dropdownMenu}>
                {devices.map((item) => {
                  const active = item.device_id === selectedMapDeviceId;
                  return (
                    <TouchableOpacity
                      key={item.device_id}
                      style={[styles.dropdownItem, active && styles.dropdownItemActive]}
                      onPress={() => {
                        setSelectedMapDeviceId(item.device_id);
                        setMapDeviceMenuOpen(false);
                      }}
                    >
                      <View style={{ flex: 1 }}>
                        <Text style={styles.dropdownItemTitle}>
                          {item.alias || item.device_name || item.device_id}
                        </Text>
                        <Text style={styles.dropdownItemMeta}>
                          {item.device_id === deviceId ? 'Lokasi milik saya' : 'Device hasil pair'}
                        </Text>
                      </View>
                      <Text style={[styles.dropdownItemStatus, item.is_online ? styles.online : styles.offline]}>
                        {item.is_online ? 'Online' : 'Offline'}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            ) : null}
            <View style={styles.routeSummaryBox}>
              <Text style={styles.routeSummaryTitle}>
                {selectedMapDevice
                  ? `Rute: ${selectedMapDevice.alias || selectedMapDevice.device_name || selectedMapDevice.device_id}`
                  : 'Rute device'}
              </Text>
              <Text style={styles.cardMeta}>
                {loadingHistory
                  ? 'Memuat history perjalanan...'
                  : selectedRoutePoints.length > 1
                  ? `Jejak perjalanan terbaca ${selectedRoutePoints.length} titik.`
                  : 'Jejak perjalanan belum cukup. Gerakkan perangkat agar rutenya tergambar.'}
              </Text>
              <Text style={styles.cardMeta}>
                Lokasi terakhir: {selectedRoutePoints.length ? formatLocationAddress(selectedRoutePoints[selectedRoutePoints.length - 1]) : formatDeviceAddress(selectedMapDevice)}
              </Text>
            </View>
            <View
              style={styles.mapBox}
              onTouchStart={() => setPageScrollEnabled(false)}
              onTouchEnd={() => setPageScrollEnabled(true)}
              onTouchCancel={() => setPageScrollEnabled(true)}
            >
              <WebView
                originWhitelist={['*']}
                source={{ html: mapHtml }}
                javaScriptEnabled
                domStorageEnabled
                nestedScrollEnabled
              />
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Daftar Device</Text>
            <Text style={styles.cardMeta}>
              Buka detail device untuk ganti nama, reset nama bawaan, atau hapus pair untuk device lain.
            </Text>
            {devices.length === 0 ? (
              <Text style={styles.muted}>Belum ada device.</Text>
            ) : (
              devices.map((d) => (
                <TouchableOpacity
                  key={d.device_id}
                  style={styles.deviceRow}
                  onPress={() => navigation.navigate('DeviceDetail' as never, { deviceId: d.device_id } as never)}
                >
                  <View>
                    <Text style={styles.deviceName}>{d.alias || d.device_name || d.device_id}</Text>
                    <Text style={styles.deviceMeta}>{formatDeviceAddress(d)}</Text>
                  </View>
                  <Text style={[styles.deviceStatus, d.is_online ? styles.online : styles.offline]}>
                    {d.is_online ? 'Online' : 'Offline'}
                  </Text>
                </TouchableOpacity>
              ))
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
      paddingHorizontal: theme.spacing.lg,
      paddingTop: theme.spacing.xxl + 16,
      paddingBottom: theme.spacing.lg,
    },
    headerTitle: {
      fontSize: theme.fontSize.xl,
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
    warnCard: {
      marginHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.md,
      padding: theme.spacing.md,
      borderRadius: theme.radius.lg,
      backgroundColor: theme.isNeo ? '#fff7ed' : 'rgba(245, 158, 11, 0.12)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(245, 158, 11, 0.35)',
      ...theme.effects.cardShadow,
    },
    warnTitle: {
      color: theme.colors.text,
      fontWeight: '800',
      fontSize: theme.fontSize.sm,
      fontFamily: theme.fonts.heading,
    },
    warnText: {
      marginTop: 6,
      color: theme.colors.textSecondary,
      fontSize: theme.fontSize.sm,
      lineHeight: 20,
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
      marginBottom: 8,
    },
    cardTopRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      gap: theme.spacing.sm,
    },
    cardValue: {
      fontSize: theme.fontSize.md,
      fontWeight: '700',
      color: theme.colors.text,
      fontFamily: theme.fonts.heading,
    },
    cardMeta: {
      fontSize: theme.fontSize.xs,
      color: theme.colors.textSecondary,
      marginTop: 4,
      fontFamily: theme.fonts.body,
    },
    btnRow: { flexDirection: 'row', gap: theme.spacing.sm, marginTop: theme.spacing.md },
    primaryBtn: {
      flex: 1,
      backgroundColor: theme.colors.accent,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
    },
    disabledBtn: {
      opacity: 0.6,
    },
    secondaryBtn: {
      flex: 1,
      backgroundColor: theme.colors.surfaceLight,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      alignItems: 'center',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    btnText: { color: theme.colors.white, fontWeight: '700', fontFamily: theme.fonts.body },
    secondaryBtnText: { color: theme.colors.text, fontWeight: '700', fontFamily: theme.fonts.body },
    statusPill: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: 6,
      borderRadius: theme.radius.full,
      borderWidth: theme.isNeo ? 2 : 1,
    },
    statusOnlinePill: {
      backgroundColor: theme.isNeo ? '#dcfce7' : 'rgba(16,185,129,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(16,185,129,0.25)',
    },
    statusOfflinePill: {
      backgroundColor: theme.isNeo ? '#fef3c7' : 'rgba(245,158,11,0.12)',
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(245,158,11,0.25)',
    },
    statusPillText: {
      color: theme.colors.text,
      fontSize: 11,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    infoBanner: {
      marginTop: theme.spacing.md,
      padding: theme.spacing.sm,
      borderRadius: theme.radius.md,
      backgroundColor: theme.isNeo ? '#fff7ed' : 'rgba(59,130,246,0.08)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(59,130,246,0.18)',
    },
    infoBannerText: {
      color: theme.colors.textSecondary,
      fontSize: theme.fontSize.xs,
      lineHeight: 18,
      fontFamily: theme.fonts.body,
    },
    qrBox: { alignItems: 'center', marginTop: theme.spacing.md },
    qrText: { marginTop: theme.spacing.sm, color: theme.colors.text, fontFamily: theme.fonts.mono, fontSize: 12 },
    qrMeta: { marginTop: 4, color: theme.colors.textMuted, fontSize: 11 },
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
    dropdownTrigger: {
      marginTop: theme.spacing.md,
      flexDirection: 'row',
      alignItems: 'center',
      gap: theme.spacing.sm,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.radius.md,
      backgroundColor: theme.colors.surfaceLight,
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
    },
    dropdownLabel: {
      color: theme.colors.textMuted,
      fontSize: 10,
      fontFamily: theme.fonts.body,
    },
    dropdownValue: {
      marginTop: 2,
      color: theme.colors.text,
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      fontFamily: theme.fonts.heading,
    },
    dropdownHint: {
      marginTop: 2,
      color: theme.colors.textSecondary,
      fontSize: 10,
      fontFamily: theme.fonts.body,
    },
    dropdownChevron: {
      color: theme.colors.text,
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    dropdownMenu: {
      marginTop: theme.spacing.sm,
      borderRadius: theme.radius.md,
      overflow: 'hidden',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.colors.panelBorder,
      backgroundColor: theme.colors.surfaceLight,
    },
    dropdownItem: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: theme.spacing.sm,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.md,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.panelBorder,
    },
    dropdownItemActive: {
      backgroundColor: theme.isNeo ? '#fff7ed' : 'rgba(59,130,246,0.08)',
    },
    dropdownItemTitle: {
      color: theme.colors.text,
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      fontFamily: theme.fonts.heading,
    },
    dropdownItemMeta: {
      marginTop: 2,
      color: theme.colors.textSecondary,
      fontSize: 10,
      fontFamily: theme.fonts.body,
    },
    dropdownItemStatus: {
      fontSize: theme.fontSize.xs,
      fontWeight: '700',
      fontFamily: theme.fonts.body,
    },
    routeSummaryBox: {
      marginTop: theme.spacing.md,
      padding: theme.spacing.sm,
      borderRadius: theme.radius.md,
      backgroundColor: theme.isNeo ? '#fff7ed' : 'rgba(59,130,246,0.08)',
      borderWidth: theme.isNeo ? 2 : 1,
      borderColor: theme.isNeo ? theme.colors.black : 'rgba(59,130,246,0.18)',
    },
    routeSummaryTitle: {
      color: theme.colors.text,
      fontSize: theme.fontSize.sm,
      fontWeight: '700',
      fontFamily: theme.fonts.heading,
    },
    mapBox: { height: 300, borderRadius: theme.radius.lg, overflow: 'hidden', marginTop: theme.spacing.md },
    muted: { color: theme.colors.textMuted, fontFamily: theme.fonts.body },
    deviceRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: theme.spacing.sm,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.panelBorder,
    },
    deviceName: { fontSize: theme.fontSize.sm, color: theme.colors.text, fontFamily: theme.fonts.heading },
    deviceMeta: { fontSize: theme.fontSize.xs, color: theme.colors.textMuted, fontFamily: theme.fonts.body },
    deviceStatus: { fontSize: theme.fontSize.xs, fontWeight: '700', fontFamily: theme.fonts.body },
    online: { color: theme.colors.emerald },
    offline: { color: theme.colors.amberDark },
  });
