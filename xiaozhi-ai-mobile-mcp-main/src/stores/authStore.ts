import EncryptedStorage from 'react-native-encrypted-storage';
import { DEFAULT_SERVER_URL } from '../config/appConfig';
import { validateServerUrl } from '../utils/serverUrl';

const TOKEN_KEY = 'scig_access_token';
const SERVER_URL_KEY = 'scig_server_url';
const LANG_KEY = 'scig_lang';

type AuthListener = (token: string | null) => void;
const listeners = new Set<AuthListener>();

function notify(token: string | null) {
  listeners.forEach((listener) => listener(token));
}

/**
 * Secure auth store using EncryptedStorage.
 * Stores access token and server URL securely on device.
 */
export const authStore = {
  async getToken(): Promise<string | null> {
    try {
      return await EncryptedStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },

  async setToken(token: string): Promise<void> {
    await EncryptedStorage.setItem(TOKEN_KEY, token);
    notify(token);
  },

  async removeToken(): Promise<void> {
    try {
      await EncryptedStorage.removeItem(TOKEN_KEY);
      notify(null);
    } catch {
      // ignore
    }
  },

  async getServerUrl(): Promise<string> {
    try {
      const url = await EncryptedStorage.getItem(SERVER_URL_KEY);
      if (!url) return DEFAULT_SERVER_URL;
      return validateServerUrl(url);
    } catch {
      return DEFAULT_SERVER_URL;
    }
  },

  async setServerUrl(url: string): Promise<void> {
    const normalized = validateServerUrl(url);
    await EncryptedStorage.setItem(SERVER_URL_KEY, normalized);
  },

  async clear(): Promise<void> {
    try {
      await EncryptedStorage.removeItem(TOKEN_KEY);
      notify(null);
    } catch {
      // ignore
    }
  },

  async getLanguage(): Promise<string> {
    try {
      return (await EncryptedStorage.getItem(LANG_KEY)) || 'id';
    } catch {
      return 'id';
    }
  },

  async setLanguage(lang: string): Promise<void> {
    await EncryptedStorage.setItem(LANG_KEY, lang);
  },

  subscribe(listener: AuthListener): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
};
