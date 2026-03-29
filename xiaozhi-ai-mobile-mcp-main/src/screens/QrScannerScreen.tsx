import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Camera, CameraType } from 'react-native-camera-kit';
import { useNavigation } from '@react-navigation/native';
import { apiClaimPairToken } from '../api/client';
import { Theme, useTheme } from '../theme/theme';

export default function QrScannerScreen() {
  const { theme } = useTheme();
  const navigation = useNavigation();
  const [scanned, setScanned] = useState(false);

  function extractToken(raw: string) {
    if (!raw) return '';
    if (raw.includes('token=')) {
      const match = raw.match(/token=([^&]+)/i);
      return match ? decodeURIComponent(match[1]) : raw;
    }
    return raw.trim();
  }

  async function onReadCode(event: any) {
    if (scanned) return;
    setScanned(true);
    const raw = event.nativeEvent?.codeStringValue || '';
    const token = extractToken(raw);
    if (!token) {
      Alert.alert('Error', 'QR tidak valid');
      setScanned(false);
      return;
    }
    try {
      await apiClaimPairToken(token);
      Alert.alert('OK', 'Device berhasil ditambahkan');
      navigation.goBack();
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Gagal menambah device');
      setScanned(false);
    }
  }

  return (
    <View style={styles(theme).container}>
      <Camera
        style={styles(theme).camera}
        cameraType={CameraType.Back}
        scanBarcode
        onReadCode={onReadCode}
        showFrame
        frameColor={theme.colors.accent}
        laserColor={theme.colors.accent}
      />
      <View style={styles(theme).overlay}>
        <Text style={styles(theme).title}>Scan QR Device</Text>
        <TouchableOpacity style={styles(theme).closeBtn} onPress={() => navigation.goBack()}>
          <Text style={styles(theme).closeText}>Tutup</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = (theme: Theme) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: theme.colors.bg },
    camera: { flex: 1 },
    overlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      paddingTop: 48,
      paddingHorizontal: 16,
      alignItems: 'center',
    },
    title: {
      color: '#fff',
      fontSize: 18,
      fontWeight: '700',
      marginBottom: 12,
    },
    closeBtn: {
      backgroundColor: 'rgba(0,0,0,0.4)',
      paddingHorizontal: 16,
      paddingVertical: 8,
      borderRadius: 12,
    },
    closeText: { color: '#fff', fontWeight: '600' },
  });
