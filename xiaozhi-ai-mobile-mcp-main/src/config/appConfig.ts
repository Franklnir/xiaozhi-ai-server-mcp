import { validateServerUrl } from '../utils/serverUrl';

type SharedAppConfig = {
  appName?: string;
  appVersion?: string;
  defaultServerUrl?: string;
};

let sharedAppConfig: SharedAppConfig = {};

try {
  sharedAppConfig = require('../../mobile_release.json') as SharedAppConfig;
} catch {
  sharedAppConfig = {};
}

export const APP_PUBLIC_NAME = (sharedAppConfig.appName || 'xiaozhiscig').trim();
export const APP_PUBLIC_VERSION = (sharedAppConfig.appVersion || '0.6').trim();

// Ganti satu nilai ini sebelum build production. User akhir tidak perlu mengetik URL lagi.
const DEFAULT_SERVER_URL_RAW = (sharedAppConfig.defaultServerUrl || 'https://xiaozhiscig.biz.id').trim();

export const DEFAULT_SERVER_URL = validateServerUrl(DEFAULT_SERVER_URL_RAW);
