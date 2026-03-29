const LOCAL_HOSTS = new Set([
  'localhost',
  '127.0.0.1',
  '10.0.2.2',
  '10.0.3.2',
  '::1',
]);

export const SERVER_URL_PLACEHOLDER = 'https://api.your-domain.example';

function isPrivateIpv4(hostname: string): boolean {
  if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostname)) return true;
  if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(hostname)) return true;
  const match = hostname.match(/^172\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/);
  if (!match) return false;
  const secondOctet = Number(match[1]);
  return secondOctet >= 16 && secondOctet <= 31;
}

function isLocalHostname(hostname: string): boolean {
  const host = hostname.trim().toLowerCase();
  return (
    LOCAL_HOSTS.has(host) ||
    host.endsWith('.local') ||
    host.endsWith('.lan') ||
    host.endsWith('.internal')
  );
}

function isPrivateIpv6(hostname: string): boolean {
  const host = hostname.trim().toLowerCase();
  return host.startsWith('fc') || host.startsWith('fd') || host.startsWith('fe80:');
}

function isLocalOrLanHost(hostname: string): boolean {
  const host = hostname.trim().toLowerCase();
  return isLocalHostname(host) || isPrivateIpv4(host) || isPrivateIpv6(host);
}

function parseHostPort(authority: string): { hostname: string; hostPort: string } {
  const value = authority.trim();
  if (!value) {
    throw new Error('Hostname server tidak valid.');
  }

  if (value.startsWith('[')) {
    const closingIndex = value.indexOf(']');
    if (closingIndex <= 1) {
      throw new Error('Hostname server tidak valid.');
    }
    const hostname = value.slice(1, closingIndex).trim().toLowerCase();
    const remainder = value.slice(closingIndex + 1);
    if (!hostname) {
      throw new Error('Hostname server tidak valid.');
    }
    if (!remainder) {
      return { hostname, hostPort: `[${hostname}]` };
    }
    if (!remainder.startsWith(':')) {
      throw new Error('Port server tidak valid.');
    }
    const port = remainder.slice(1);
    if (!/^\d{1,5}$/.test(port)) {
      throw new Error('Port server tidak valid.');
    }
    const portNumber = Number(port);
    if (portNumber < 1 || portNumber > 65535) {
      throw new Error('Port server tidak valid.');
    }
    return { hostname, hostPort: `[${hostname}]:${portNumber}` };
  }

  const parts = value.split(':');
  if (parts.length === 1) {
    const hostname = value.toLowerCase();
    return { hostname, hostPort: hostname };
  }

  const maybePort = parts[parts.length - 1];
  if (!/^\d{1,5}$/.test(maybePort)) {
    throw new Error('Gunakan format host:port yang valid.');
  }
  const portNumber = Number(maybePort);
  if (portNumber < 1 || portNumber > 65535) {
    throw new Error('Port server tidak valid.');
  }

  const hostname = parts.slice(0, -1).join(':').trim().toLowerCase();
  if (!hostname) {
    throw new Error('Hostname server tidak valid.');
  }
  return { hostname, hostPort: `${hostname}:${portNumber}` };
}

export function validateServerUrl(input: string): string {
  const raw = (input || '').trim();
  if (!raw) {
    throw new Error('Server URL wajib diisi.');
  }

  const match = raw.match(/^(https?):\/\/([^/?#]+)(?:[/?#].*)?$/i);
  if (!match) {
    throw new Error('Server URL tidak valid. Gunakan format http:// atau https://');
  }

  const protocol = match[1].toLowerCase();
  const { hostname, hostPort } = parseHostPort(match[2]);

  if (protocol === 'http' && !isLocalOrLanHost(hostname)) {
    throw new Error('HTTP hanya boleh untuk localhost/LAN. Untuk VPS atau production wajib HTTPS.');
  }

  return `${protocol}://${hostPort}`;
}
