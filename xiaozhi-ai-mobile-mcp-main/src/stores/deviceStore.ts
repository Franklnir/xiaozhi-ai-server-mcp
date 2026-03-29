import EncryptedStorage from 'react-native-encrypted-storage';

const KEY_DEVICE_ID = 'scig_device_id';
const KEY_DEVICE_TOKEN = 'scig_device_token';
const KEY_TRACKING = 'scig_tracking_enabled';

export const deviceStore = {
  async getDeviceId(): Promise<string> {
    return (await EncryptedStorage.getItem(KEY_DEVICE_ID)) || '';
  },

  async setDeviceId(deviceId: string) {
    await EncryptedStorage.setItem(KEY_DEVICE_ID, deviceId);
  },

  async getDeviceToken(): Promise<string> {
    return (await EncryptedStorage.getItem(KEY_DEVICE_TOKEN)) || '';
  },

  async setDeviceToken(token: string) {
    await EncryptedStorage.setItem(KEY_DEVICE_TOKEN, token);
  },

  async isTrackingEnabled(): Promise<boolean> {
    const val = await EncryptedStorage.getItem(KEY_TRACKING);
    return val === '1';
  },

  async setTrackingEnabled(enabled: boolean) {
    await EncryptedStorage.setItem(KEY_TRACKING, enabled ? '1' : '0');
  },

  async clear() {
    await EncryptedStorage.removeItem(KEY_DEVICE_ID);
    await EncryptedStorage.removeItem(KEY_DEVICE_TOKEN);
    await EncryptedStorage.removeItem(KEY_TRACKING);
  },
};
