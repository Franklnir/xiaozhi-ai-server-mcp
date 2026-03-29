import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosResponse } from 'axios';
import { authStore } from '../stores/authStore';
import { validateServerUrl } from '../utils/serverUrl';

let apiClient: AxiosInstance | null = null;
let apiBaseURL = '';

async function resolveBaseURL(serverUrl?: string): Promise<string> {
  const raw = (serverUrl || '').trim();
  if (raw) {
    return validateServerUrl(raw);
  }
  return authStore.getServerUrl();
}

/**
 * Initialize or re-initialize the API client with the given server URL.
 */
export async function initApiClient(): Promise<AxiosInstance> {
  const baseURL = await resolveBaseURL();
  apiBaseURL = baseURL;

  apiClient = axios.create({
    baseURL,
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'X-Client': 'mobile',
    },
    withCredentials: true,
  });

  // Request interceptor: attach token
  apiClient.interceptors.request.use(
    async (config: InternalAxiosRequestConfig) => {
      const token = await authStore.getToken();
      if (config.headers) {
        config.headers['X-Client'] = 'mobile';
        if (token) {
          config.headers.Cookie = `access_token=${token}`;
        }
      }
      return config;
    },
    (error) => Promise.reject(error),
  );

  // Response interceptor: handle 401
  apiClient.interceptors.response.use(
    (response: AxiosResponse) => response,
    async (error) => {
      if (error.response?.status === 401) {
        await authStore.removeToken();
      }
      return Promise.reject(error);
    },
  );

  return apiClient;
}

/**
 * Get the current API client instance. Initializes if needed.
 */
export async function getApi(): Promise<AxiosInstance> {
  const baseURL = await authStore.getServerUrl();
  if (!apiClient || apiBaseURL !== baseURL) {
    return initApiClient();
  }
  return apiClient;
}

// ─── API Functions ───────────────────────────────────────

export interface LoginResult {
  success: boolean;
  token?: string;
  error?: string;
}

export interface RegisterResult {
  success: boolean;
  token?: string;
  error?: string;
}

function extractRegisterError(html?: string): string | null {
  if (!html) return null;
  const match = html.match(/<div class="mt-5[^>]*>([^<]+)<\/div>/i);
  return match?.[1]?.trim() || null;
}

/**
 * Login to the backend. The backend uses cookie-based auth,
 * so we POST form data and extract the cookie from the redirect response.
 */
export async function apiLogin(
  serverUrl: string | undefined,
  username: string,
  password: string,
): Promise<LoginResult> {
  try {
    const normalizedServerUrl = await resolveBaseURL(serverUrl);
    // Save server URL for future API calls
    await authStore.setServerUrl(normalizedServerUrl);

    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await axios.post(`${normalizedServerUrl}/login`, formData.toString(), {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Accept: 'application/json',
        'X-Client': 'mobile',
      },
      maxRedirects: 0,
      validateStatus: (status) => status >= 200 && status < 500,
      withCredentials: true,
    });

    const data = response.data;
    if (data && typeof data === 'object') {
      const token = data.access_token || data.token;
      if (typeof token === 'string' && token.length > 0) {
        await authStore.setToken(token);
        await initApiClient();
        return { success: true, token };
      }
      if (data.error) {
        return { success: false, error: data.error };
      }
    }

    // Extract access_token from Set-Cookie header
    const cookies = response.headers['set-cookie'];
    if (cookies) {
      for (const cookie of cookies) {
        const match = cookie.match(/access_token=([^;]+)/);
        if (match) {
          const token = match[1];
          await authStore.setToken(token);
          await initApiClient();
          return { success: true, token };
        }
      }
    }

    return { success: false, error: 'Token tidak ditemukan di response' };
  } catch (error: any) {
    const msg = error.response?.data?.error || error.message || 'Login gagal';
    return { success: false, error: msg };
  }
}

/**
 * Register a new account. Backend returns HTML; on success it sets cookie + redirect.
 */
export async function apiRegister(
  serverUrl: string | undefined,
  username: string,
  password: string,
  confirmPassword: string,
  code?: string,
): Promise<RegisterResult> {
  try {
    const normalizedServerUrl = await resolveBaseURL(serverUrl);
    await authStore.setServerUrl(normalizedServerUrl);

    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);
    formData.append('confirm_password', confirmPassword);
    if (typeof code === 'string') {
      formData.append('code', code);
    }

    const response = await axios.post(`${normalizedServerUrl}/register`, formData.toString(), {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Accept: 'application/json',
        'X-Client': 'mobile',
      },
      maxRedirects: 0,
      validateStatus: (status) => status >= 200 && status < 500,
      withCredentials: true,
    });

    const data = response.data;
    if (data && typeof data === 'object') {
      const token = data.access_token || data.token;
      if (typeof token === 'string' && token.length > 0) {
        await authStore.setToken(token);
        await initApiClient();
        return { success: true, token };
      }
      if (data.error) {
        return { success: false, error: data.error };
      }
    }

    const cookies = response.headers['set-cookie'];
    if (cookies) {
      for (const cookie of cookies) {
        const match = cookie.match(/access_token=([^;]+)/);
        if (match) {
          const token = match[1];
          await authStore.setToken(token);
          await initApiClient();
          return { success: true, token };
        }
      }
    }

    const html = typeof response.data === 'string' ? response.data : '';
    return { success: false, error: extractRegisterError(html) || 'Register gagal' };
  } catch (error: any) {
    const msg = error.response?.data?.error || error.message || 'Register gagal';
    return { success: false, error: msg };
  }
}

export async function apiGetPublicSettings(serverUrl?: string): Promise<PublicSettings> {
  const baseURL = await resolveBaseURL(serverUrl);
  const res = await axios.get(`${baseURL}/api/public/settings`, { timeout: 10000 });
  return res.data;
}

export interface ModeInfo {
  id: number;
  name: string;
  title: string;
  introduction: string;
}

export interface DeviceInfo {
  device_id: string;
  alias?: string;
  device_name?: string;
  platform?: string;
  model?: string;
  os_version?: string;
  last_seen_at?: string;
  latitude?: number;
  longitude?: number;
  address_street?: string;
  address_area?: string;
  address_city?: string;
  address_full?: string;
  battery_level?: number;
  battery_status?: string;
  charging_type?: string;
  battery_temp?: number;
  network_type?: string;
  signal_strength?: number;
  carrier?: string;
  ram_used?: number;
  ram_total?: number;
  storage_used?: number;
  storage_total?: number;
  is_online?: number;
}

export interface DeviceLocationInfo {
  id: number;
  device_id: string;
  latitude?: number;
  longitude?: number;
  address_street?: string;
  address_area?: string;
  address_city?: string;
  address_full?: string;
  recorded_at?: string;
}

export interface DeviceRegisterResult {
  device_id: string;
  device_token: string;
  device_name?: string;
}

export interface McpCodeInfo {
  id: number;
  code: string;
  has_token?: number;
  is_connected?: number;
  created_at?: string;
  last_ok_at?: string;
  last_err_at?: string;
  last_error?: string;
}

export interface SocialLink {
  label: string;
  url: string;
}

export interface PublicSettings {
  register_requires_code: boolean;
  social_links: SocialLink[];
}

export interface ThreadInfo {
  id: number;
  title: string;
  updated_at: string;
  mode_name?: string;
  mode_title?: string;
}

export interface MessageInfo {
  id: number;
  role: string;
  content: string;
  created_at: string;
}

export async function apiGetConfig() {
  const api = await getApi();
  const res = await api.get('/api/config');
  return res.data;
}

export async function apiGetModes(): Promise<ModeInfo[]> {
  const api = await getApi();
  const res = await api.get('/api/modes');
  return res.data;
}

export async function apiSaveMode(mode: ModeInfo) {
  const api = await getApi();
  const res = await api.post('/api/modes', {
    name: mode.name,
    title: mode.title,
    introduction: mode.introduction,
  });
  return res.data;
}

export async function apiRenderPrompt(
  modeId: number | null,
  vars: Record<string, any>,
  deviceId?: string,
  overrides?: { title?: string; introduction?: string },
) {
  const api = await getApi();
  const res = await api.post('/api/render-prompt', {
    mode_id: modeId,
    vars,
    device_id: deviceId || null,
    title: overrides?.title,
    introduction: overrides?.introduction,
  });
  return res.data;
}

export async function apiGetActiveMode(deviceId: string): Promise<ModeInfo> {
  const api = await getApi();
  const res = await api.get(`/api/mode?device_id=${encodeURIComponent(deviceId)}`);
  return res.data;
}

export async function apiSetActiveMode(deviceId: string, modeId: number) {
  const api = await getApi();
  const res = await api.post('/api/mode', { device_id: deviceId, mode_id: modeId });
  return res.data;
}

export async function apiGetMyCodes(): Promise<McpCodeInfo[]> {
  const api = await getApi();
  const res = await api.get('/api/mcp/my-codes');
  return res.data;
}

export async function apiTestMcpToken(token: string): Promise<{ ok: boolean; error?: string }> {
  const api = await getApi();
  const res = await api.post('/api/mcp/test-token', { token });
  return res.data;
}

export async function apiCreateMyMcpToken(token: string) {
  const api = await getApi();
  const res = await api.post('/api/mcp/my-codes/create', { token });
  return res.data;
}

export async function apiUpdateMyMcpToken(codeId: number, token: string) {
  const api = await getApi();
  const res = await api.post('/api/mcp/my-codes/update', { code_id: codeId, token });
  return res.data;
}

export async function apiClearMyMcpToken(codeId: number) {
  const api = await getApi();
  const res = await api.post('/api/mcp/my-codes/clear', { code_id: codeId });
  return res.data;
}

export async function apiGetThreads(deviceId: string): Promise<ThreadInfo[]> {
  const api = await getApi();
  const res = await api.get(`/api/chats?device_id=${encodeURIComponent(deviceId)}`);
  return res.data;
}

export async function apiGetMessages(
  deviceId: string,
  threadId: number,
  limit = 200,
): Promise<MessageInfo[]> {
  const api = await getApi();
  const res = await api.get(
    `/api/chats/${threadId}/messages?device_id=${encodeURIComponent(deviceId)}&limit=${limit}`,
  );
  return res.data;
}

export async function apiGetLastDevice() {
  const api = await getApi();
  const res = await api.get('/api/last-device');
  return res.data;
}

export async function apiGetDeviceSettings(deviceId: string) {
  const api = await getApi();
  const res = await api.get(`/api/device/settings?device_id=${encodeURIComponent(deviceId)}`);
  return res.data;
}

export async function apiSetDeviceSettings(deviceId: string, source: string, target: string) {
  const api = await getApi();
  const res = await api.post('/api/device/settings', { device_id: deviceId, source, target });
  return res.data;
}

export async function apiRegisterDevice(payload: {
  device_id: string;
  device_name?: string;
  device_token?: string;
  platform?: string;
  model?: string;
  os_version?: string;
}): Promise<DeviceRegisterResult> {
  const api = await getApi();
  const res = await api.post('/api/devices/register', payload);
  return res.data;
}

export async function apiDeviceHeartbeat(payload: {
  device_id: string;
  device_token: string;
  latitude?: number;
  longitude?: number;
  battery_level?: number;
  battery_status?: string;
  charging_type?: string;
  battery_temp?: number;
  network_type?: string;
  signal_strength?: number;
  carrier?: string;
  ram_used?: number;
  ram_total?: number;
  storage_used?: number;
  storage_total?: number;
}): Promise<{ ok: boolean; device_id: string }> {
  const api = await getApi();
  const res = await api.post('/api/devices/heartbeat', payload);
  return res.data;
}

export async function apiGetDevices(): Promise<DeviceInfo[]> {
  const api = await getApi();
  const res = await api.get('/api/devices');
  return res.data;
}

export async function apiGetDeviceDetail(deviceId: string): Promise<DeviceInfo> {
  const api = await getApi();
  const res = await api.get(`/api/devices/${encodeURIComponent(deviceId)}`);
  return res.data;
}

export async function apiGetDeviceLocations(
  deviceId: string,
  limit = 400,
): Promise<DeviceLocationInfo[]> {
  const api = await getApi();
  const res = await api.get(
    `/api/devices/${encodeURIComponent(deviceId)}/locations?limit=${encodeURIComponent(String(limit))}`,
  );
  return res.data;
}

export async function apiSetDeviceAlias(deviceId: string, alias: string) {
  const api = await getApi();
  const res = await api.post('/api/devices/alias', { device_id: deviceId, alias });
  return res.data;
}

export async function apiUnpairDevice(deviceId: string) {
  const api = await getApi();
  const res = await api.post('/api/devices/unpair', { device_id: deviceId });
  return res.data;
}

export async function apiCreatePairToken(deviceId: string, deviceToken: string) {
  const api = await getApi();
  const res = await api.post('/api/devices/pair-token', { device_id: deviceId, device_token: deviceToken });
  return res.data;
}

export async function apiClaimPairToken(pairToken: string): Promise<{
  ok: boolean;
  device_id: string;
  device: DeviceInfo;
}> {
  const api = await getApi();
  const res = await api.post('/api/devices/pair-claim', { pair_token: pairToken });
  return res.data;
}
