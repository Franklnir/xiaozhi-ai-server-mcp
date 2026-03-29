import { Linking, PermissionsAndroid, Platform } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import NetInfo from '@react-native-community/netinfo';
import Geolocation from 'react-native-geolocation-service';
import BackgroundService from 'react-native-background-actions';
import { apiDeviceHeartbeat, apiRegisterDevice } from '../api/client';
import { deviceStore } from '../stores/deviceStore';

const sleep = (time: number) => new Promise((resolve) => setTimeout(resolve, time));
export const TRACKING_INTERVAL_MS = 5 * 1000;
export const TRACKING_INTERVAL_LABEL = '5 detik';
let trackingStartPromise: Promise<void> | null = null;
const ANDROID_VERSION = Platform.OS === 'android' ? Number(Platform.Version) : 0;

type AndroidPermission = Parameters<typeof PermissionsAndroid.check>[0];
type AndroidPermissionStatus =
  (typeof PermissionsAndroid.RESULTS)[keyof typeof PermissionsAndroid.RESULTS];

const FINE_LOCATION: AndroidPermission = PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION;
const COARSE_LOCATION: AndroidPermission = PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION;
const BACKGROUND_LOCATION: AndroidPermission = PermissionsAndroid.PERMISSIONS.ACCESS_BACKGROUND_LOCATION;
const POST_NOTIFICATIONS: AndroidPermission = PermissionsAndroid.PERMISSIONS.POST_NOTIFICATIONS;

export interface TrackingPermissionSummary {
  ready: boolean;
  missing: string[];
  needsSettings: boolean;
  locationGranted: boolean;
  backgroundGranted: boolean;
  notificationsGranted: boolean;
}

export interface AppEntryPermissionSummary {
  ready: boolean;
  missing: string[];
  locationGranted: boolean;
}

export interface BackgroundActivationResult {
  summary: TrackingPermissionSummary;
  trackingEnabled: boolean;
  openedSettings: boolean;
}

function buildAppEntrySummary(locationGranted: boolean): AppEntryPermissionSummary {
  return {
    ready: locationGranted,
    missing: locationGranted ? [] : ['Lokasi perangkat'],
    locationGranted,
  };
}

function isAndroidPermissionRequired(permission: AndroidPermission): boolean {
  if (permission === BACKGROUND_LOCATION) {
    return Platform.OS === 'android' && ANDROID_VERSION >= 29;
  }
  if (permission === POST_NOTIFICATIONS) {
    return Platform.OS === 'android' && ANDROID_VERSION >= 33;
  }
  return Platform.OS === 'android';
}

function buildPermissionSummary(params: {
  locationGranted: boolean;
  backgroundGranted: boolean;
  notificationsGranted: boolean;
  needsSettings?: boolean;
}): TrackingPermissionSummary {
  const missing: string[] = [];
  if (!params.locationGranted) {
    missing.push('Lokasi perangkat');
  }
  if (!params.backgroundGranted) {
    missing.push('Lokasi latar belakang');
  }
  if (!params.notificationsGranted) {
    missing.push('Notifikasi foreground service');
  }
  return {
    ready: missing.length === 0,
    missing,
    needsSettings: params.needsSettings || false,
    locationGranted: params.locationGranted,
    backgroundGranted: params.backgroundGranted,
    notificationsGranted: params.notificationsGranted,
  };
}

async function checkAndroidPermission(permission: AndroidPermission): Promise<boolean> {
  if (!isAndroidPermissionRequired(permission)) return true;
  return PermissionsAndroid.check(permission);
}

export async function getTrackingPermissionSummary(): Promise<TrackingPermissionSummary> {
  if (Platform.OS !== 'android') {
    return buildPermissionSummary({
      locationGranted: true,
      backgroundGranted: true,
      notificationsGranted: true,
    });
  }

  const fineGranted = await checkAndroidPermission(FINE_LOCATION);
  const coarseGranted = await checkAndroidPermission(COARSE_LOCATION);
  const backgroundGranted = await checkAndroidPermission(BACKGROUND_LOCATION);
  const notificationsGranted = await checkAndroidPermission(POST_NOTIFICATIONS);
  const locationGranted = fineGranted || coarseGranted;
  const needsSettings = ANDROID_VERSION >= 30 && locationGranted && !backgroundGranted;

  return buildPermissionSummary({
    locationGranted,
    backgroundGranted,
    notificationsGranted,
    needsSettings,
  });
}

export async function getAppEntryPermissionSummary(): Promise<AppEntryPermissionSummary> {
  if (Platform.OS !== 'android') {
    return buildAppEntrySummary(true);
  }

  const fineGranted = await checkAndroidPermission(FINE_LOCATION);
  const coarseGranted = await checkAndroidPermission(COARSE_LOCATION);
  return buildAppEntrySummary(fineGranted || coarseGranted);
}

function buildPermissionError(summary: TrackingPermissionSummary): string {
  if (summary.ready) {
    return '';
  }
  const tail = summary.needsSettings
    ? ' Sebagian izin harus diaktifkan manual dari Pengaturan Aplikasi.'
    : '';
  return `Izin wajib belum lengkap: ${summary.missing.join(', ')}.${tail}`;
}

async function requestAndroidPermission(
  permission: AndroidPermission,
): Promise<AndroidPermissionStatus> {
  if (!isAndroidPermissionRequired(permission)) {
    return PermissionsAndroid.RESULTS.GRANTED;
  }

  try {
    return await PermissionsAndroid.request(permission);
  } catch {
    return PermissionsAndroid.RESULTS.DENIED;
  }
}

async function ensureLocationPermission(): Promise<{
  granted: boolean;
  neverAskAgain: boolean;
}> {
  const existingFine = await checkAndroidPermission(FINE_LOCATION);
  const existingCoarse = await checkAndroidPermission(COARSE_LOCATION);

  if (existingFine || existingCoarse) {
    return {
      granted: true,
      neverAskAgain: false,
    };
  }

  const fineResult = await requestAndroidPermission(FINE_LOCATION);
  let fineGranted = await checkAndroidPermission(FINE_LOCATION);
  let coarseGranted = await checkAndroidPermission(COARSE_LOCATION);

  if (!fineGranted && !coarseGranted && fineResult !== PermissionsAndroid.RESULTS.NEVER_ASK_AGAIN) {
    await requestAndroidPermission(COARSE_LOCATION);
    fineGranted = await checkAndroidPermission(FINE_LOCATION);
    coarseGranted = await checkAndroidPermission(COARSE_LOCATION);
  }

  return {
    granted: fineGranted || coarseGranted,
    neverAskAgain: fineResult === PermissionsAndroid.RESULTS.NEVER_ASK_AGAIN,
  };
}

async function ensureNotificationPermission(): Promise<{
  granted: boolean;
  neverAskAgain: boolean;
}> {
  const alreadyGranted = await checkAndroidPermission(POST_NOTIFICATIONS);
  if (alreadyGranted) {
    return {
      granted: true,
      neverAskAgain: false,
    };
  }

  const result = await requestAndroidPermission(POST_NOTIFICATIONS);
  const granted = await checkAndroidPermission(POST_NOTIFICATIONS);

  return {
    granted,
    neverAskAgain: result === PermissionsAndroid.RESULTS.NEVER_ASK_AGAIN,
  };
}

async function ensureBackgroundLocationPermission(locationGranted: boolean): Promise<{
  granted: boolean;
  needsSettings: boolean;
}> {
  if (!isAndroidPermissionRequired(BACKGROUND_LOCATION)) {
    return {
      granted: true,
      needsSettings: false,
    };
  }

  const alreadyGranted = await checkAndroidPermission(BACKGROUND_LOCATION);
  if (alreadyGranted || !locationGranted) {
    return {
      granted: alreadyGranted,
      needsSettings: false,
    };
  }
  return {
    granted: false,
    needsSettings: true,
  };
}

export async function requestAppEntryPermissions(): Promise<AppEntryPermissionSummary> {
  if (Platform.OS !== 'android') {
    return getAppEntryPermissionSummary();
  }

  const location = await ensureLocationPermission();
  return buildAppEntrySummary(location.granted);
}

export async function requestTrackingPermissions(): Promise<TrackingPermissionSummary> {
  if (Platform.OS !== 'android') {
    return getTrackingPermissionSummary();
  }

  let needsSettings = false;
  const location = await ensureLocationPermission();
  needsSettings = needsSettings || location.neverAskAgain;

  const notifications = await ensureNotificationPermission();
  needsSettings = needsSettings || notifications.neverAskAgain;

  const background = await ensureBackgroundLocationPermission(location.granted);
  needsSettings = needsSettings || background.needsSettings;

  return buildPermissionSummary({
    locationGranted: location.granted,
    backgroundGranted: background.granted,
    notificationsGranted: notifications.granted,
    needsSettings,
  });
}

export async function openAppSettings() {
  await Linking.openSettings();
}

function buildTrackingNotificationMeta(isOnline: boolean, note?: string) {
  return {
    taskTitle: isOnline ? 'xiaozhiscig monitor online' : 'xiaozhiscig monitor offline',
    taskDesc: note
      ? `${isOnline ? 'Online' : 'Offline'} • ${note}`
      : `${isOnline ? 'Online' : 'Offline'} • sinkron tiap ${TRACKING_INTERVAL_LABEL}`,
  };
}

async function updateTrackingNotification(isOnline: boolean, note?: string) {
  if (!BackgroundService.isRunning()) {
    return;
  }
  try {
    await BackgroundService.updateNotification(buildTrackingNotificationMeta(isOnline, note));
  } catch {
    // ignore notification update failures
  }
}

async function safeCall<T>(factory: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await factory();
  } catch {
    return fallback;
  }
}

async function getCurrentLocation(): Promise<{ latitude: number; longitude: number } | null> {
  const summary = await getTrackingPermissionSummary();
  if (!summary.locationGranted) return null;

  return new Promise((resolve) => {
    Geolocation.getCurrentPosition(
      (pos) => {
        resolve({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
        });
      },
      () => resolve(null),
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: TRACKING_INTERVAL_MS,
      },
    );
  });
}

async function collectDeviceStatus() {
  const [batteryLevel, powerState, carrier, totalMem, usedMem, totalDisk, freeDisk, netInfo] =
    await Promise.all([
      safeCall(() => DeviceInfo.getBatteryLevel(), 0),
      safeCall(() => DeviceInfo.getPowerState(), null),
      safeCall(() => DeviceInfo.getCarrier(), ''),
      safeCall(() => DeviceInfo.getTotalMemory(), 0),
      safeCall(() => DeviceInfo.getUsedMemory(), 0),
      safeCall(() => DeviceInfo.getTotalDiskCapacity(), 0),
      safeCall(() => DeviceInfo.getFreeDiskStorage(), 0),
      safeCall(() => NetInfo.fetch(), null as any),
    ]);

  const batteryTemp: number | undefined = undefined;

  const location = await getCurrentLocation();
  const batteryPercent = Math.round((batteryLevel || 0) * 100);

  let networkType = netInfo.type?.toUpperCase() || 'UNKNOWN';
  if (netInfo.type === 'cellular' && netInfo.details && 'cellularGeneration' in netInfo.details) {
    const gen = (netInfo.details as any).cellularGeneration;
    networkType = gen ? String(gen).toUpperCase() : 'CELLULAR';
  }

  const batteryStatus = powerState?.batteryState || 'unknown';
  const chargingType = batteryStatus === 'charging' || batteryStatus === 'full' ? 'Charging' : 'Not Charging';

  return {
    latitude: location?.latitude,
    longitude: location?.longitude,
    battery_level: batteryPercent,
    battery_status: batteryStatus,
    charging_type: chargingType,
    battery_temp: batteryTemp,
    network_type: networkType,
    carrier: carrier || 'Unknown',
    ram_used: usedMem,
    ram_total: totalMem,
    storage_used: totalDisk && freeDisk ? totalDisk - freeDisk : undefined,
    storage_total: totalDisk,
  };
}

export async function registerDevice() {
  const deviceId = await DeviceInfo.getUniqueId();
  const deviceName = await safeCall(() => DeviceInfo.getDeviceName(), 'SciG Device');
  const model = DeviceInfo.getModel();
  const osVersion = DeviceInfo.getSystemVersion();
  const storedToken = await deviceStore.getDeviceToken();

  const result = await apiRegisterDevice({
    device_id: deviceId,
    device_name: deviceName,
    device_token: storedToken || undefined,
    platform: Platform.OS,
    model,
    os_version: osVersion,
  });

  await deviceStore.setDeviceId(result.device_id);
  await deviceStore.setDeviceToken(result.device_token);

  return result;
}

export async function syncDeviceSnapshot() {
  const reg = await registerDevice();
  try {
    await sendHeartbeat();
  } catch {
    // Keep the device registered even if the first foreground sync has not reached the backend yet.
  }
  return reg;
}

export async function sendHeartbeat() {
  let deviceId = await deviceStore.getDeviceId();
  let deviceToken = await deviceStore.getDeviceToken();
  if (!deviceId || !deviceToken) {
    const reg = await registerDevice();
    deviceId = reg.device_id;
    deviceToken = reg.device_token;
  }

  const status = await collectDeviceStatus();
  return apiDeviceHeartbeat({
    device_id: deviceId,
    device_token: deviceToken,
    ...status,
  });
}

const trackingOptions = {
  taskName: 'SciG Tracking',
  taskTitle: 'xiaozhiscig monitor latar belakang',
  taskDesc: `Offline • sinkron tiap ${TRACKING_INTERVAL_LABEL}`,
  taskIcon: {
    name: 'ic_launcher',
    type: 'mipmap',
  },
  color: '#3b82f6',
  parameters: {},
};

const trackingTask = async () => {
  while (BackgroundService.isRunning()) {
    try {
      await sendHeartbeat();
      await updateTrackingNotification(true, `sinkron tiap ${TRACKING_INTERVAL_LABEL}`);
    } catch {
      await updateTrackingNotification(false, 'cek koneksi, lokasi, atau izin notifikasi');
    }
    await sleep(TRACKING_INTERVAL_MS);
  }
};

export async function startTracking(options: { skipPermissionCheck?: boolean } = {}) {
  if (trackingStartPromise) {
    return trackingStartPromise;
  }

  trackingStartPromise = (async () => {
    if (!options.skipPermissionCheck) {
      const summary = await requestTrackingPermissions();
      if (!summary.ready) {
        throw new Error(buildPermissionError(summary));
      }
    }

    try {
      let synced = false;
      try {
        await syncDeviceSnapshot();
        synced = true;
      } catch {
        synced = false;
      }
      if (!BackgroundService.isRunning()) {
        await BackgroundService.start(trackingTask, trackingOptions);
      }
      await updateTrackingNotification(
        synced,
        synced ? `sinkron tiap ${TRACKING_INTERVAL_LABEL}` : 'menunggu sinkron pertama',
      );
    } catch {
      await deviceStore.setTrackingEnabled(false);
      throw new Error(
        'Tracking latar belakang belum bisa dinyalakan. Aktifkan izin lokasi latar belakang dari Pengaturan Aplikasi, izinkan notifikasi, lalu coba lagi.',
      );
    }

    await deviceStore.setTrackingEnabled(true);
  })();

  try {
    await trackingStartPromise;
  } finally {
    trackingStartPromise = null;
  }
}

export async function stopTracking() {
  if (BackgroundService.isRunning()) {
    await BackgroundService.stop();
  }
  await deviceStore.setTrackingEnabled(false);
}

export async function isTrackingRunning(): Promise<boolean> {
  return BackgroundService.isRunning();
}

export async function activateBackgroundMonitoring(): Promise<BackgroundActivationResult> {
  if (Platform.OS !== 'android') {
    await startTracking();
    return {
      summary: await getTrackingPermissionSummary(),
      trackingEnabled: await isTrackingRunning(),
      openedSettings: false,
    };
  }

  let needsSettings = false;
  let openedSettings = false;

  const location = await ensureLocationPermission();
  needsSettings = needsSettings || location.neverAskAgain;

  const notifications = await ensureNotificationPermission();
  needsSettings = needsSettings || notifications.neverAskAgain;

  let backgroundGranted = await checkAndroidPermission(BACKGROUND_LOCATION);
  if (isAndroidPermissionRequired(BACKGROUND_LOCATION) && location.granted && !backgroundGranted) {
    if (ANDROID_VERSION <= 29) {
      const result = await requestAndroidPermission(BACKGROUND_LOCATION);
      backgroundGranted = await checkAndroidPermission(BACKGROUND_LOCATION);
      needsSettings = needsSettings || result === PermissionsAndroid.RESULTS.NEVER_ASK_AGAIN;
    } else {
      needsSettings = true;
      openedSettings = true;
      await openAppSettings().catch(() => {});
    }
  }

  const summary = buildPermissionSummary({
    locationGranted: location.granted,
    backgroundGranted,
    notificationsGranted: notifications.granted,
    needsSettings,
  });

  if (summary.ready) {
    await startTracking({ skipPermissionCheck: true });
  } else {
    await deviceStore.setTrackingEnabled(false);
  }

  return {
    summary,
    trackingEnabled: await isTrackingRunning(),
    openedSettings,
  };
}

export async function bootstrapTrackingAfterLogin(): Promise<AppEntryPermissionSummary> {
  const appSummary = await requestAppEntryPermissions();
  await deviceStore.setTrackingEnabled(false);

  if (!appSummary.ready) {
    return appSummary;
  }

  await syncDeviceSnapshot().catch(() => {});
  return appSummary;
}

export async function resumeTrackingIfEnabled() {
  const enabled = await deviceStore.isTrackingEnabled();
  if (!enabled) {
    return false;
  }

  const summary = await getTrackingPermissionSummary();
  if (!summary.ready) {
    await deviceStore.setTrackingEnabled(false);
    return false;
  }

  await startTracking({ skipPermissionCheck: true });
  return true;
}
