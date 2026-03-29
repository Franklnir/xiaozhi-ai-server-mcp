import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { ColorSchemeName, Platform, useColorScheme, ViewStyle } from 'react-native';
import EncryptedStorage from 'react-native-encrypted-storage';

export type ThemeName = 'default' | 'dark' | 'light' | 'neo';

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
};

export const FontSize = {
  xs: 11,
  sm: 13,
  md: 15,
  lg: 18,
  xl: 22,
  xxl: 28,
};

const RadiusDefault = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 999,
};

const RadiusNeo = {
  sm: 4,
  md: 6,
  lg: 8,
  xl: 12,
  full: 16,
};

const THEME_KEY = 'scig_theme_mode';

const baseFonts = {
  heading: Platform.OS === 'ios' ? 'Georgia' : 'serif',
  body: Platform.OS === 'ios' ? 'Avenir' : 'sans-serif',
  mono: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
};

const neoFonts = {
  heading: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
  body: Platform.OS === 'ios' ? 'Georgia' : 'serif',
  mono: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
};

const darkColors = {
  bg: '#0a1224',
  panel: 'rgba(15,23,42,0.78)',
  panelBorder: 'rgba(255,255,255,0.10)',
  surface: '#0f172a',
  surfaceLight: '#1e293b',

  text: '#e5e7eb',
  textSecondary: '#94a3b8',
  textMuted: '#64748b',

  accent: '#3b82f6',
  accentLight: '#60a5fa',
  accentDark: '#2563eb',

  emerald: '#10b981',
  emeraldDark: '#059669',

  teal: '#14b8a6',
  tealDark: '#0d9488',

  red: '#ef4444',
  redDark: '#dc2626',

  amber: '#f59e0b',
  amberDark: '#d97706',

  purple: '#8b5cf6',
  purpleDark: '#7c3aed',

  white: '#ffffff',
  black: '#000000',

  gradientBlueStart: 'rgba(37,99,235,0.35)',
  gradientTealStart: 'rgba(20,184,166,0.25)',
};

const lightColors = {
  bg: '#f8fafc',
  panel: '#ffffff',
  panelBorder: '#e2e8f0',
  surface: '#f1f5f9',
  surfaceLight: '#e2e8f0',

  text: '#0f172a',
  textSecondary: '#475569',
  textMuted: '#94a3b8',

  accent: '#2563eb',
  accentLight: '#3b82f6',
  accentDark: '#1d4ed8',

  emerald: '#10b981',
  emeraldDark: '#059669',

  teal: '#14b8a6',
  tealDark: '#0d9488',

  red: '#ef4444',
  redDark: '#dc2626',

  amber: '#f59e0b',
  amberDark: '#d97706',

  purple: '#8b5cf6',
  purpleDark: '#7c3aed',

  white: '#ffffff',
  black: '#000000',

  gradientBlueStart: 'rgba(37,99,235,0.10)',
  gradientTealStart: 'rgba(20,184,166,0.10)',
};

const neoColors = {
  bg: '#fef3c7',
  panel: '#ffffff',
  panelBorder: '#111827',
  surface: '#fde047',
  surfaceLight: '#a7f3d0',

  text: '#111827',
  textSecondary: '#1f2937',
  textMuted: '#374151',

  accent: '#ff2d55',
  accentLight: '#ff7aa2',
  accentDark: '#c9184a',

  emerald: '#16a34a',
  emeraldDark: '#166534',

  teal: '#0ea5e9',
  tealDark: '#0369a1',

  red: '#ef4444',
  redDark: '#b91c1c',

  amber: '#f59e0b',
  amberDark: '#b45309',

  purple: '#8b5cf6',
  purpleDark: '#6d28d9',

  white: '#ffffff',
  black: '#000000',

  gradientBlueStart: 'rgba(59,130,246,0.20)',
  gradientTealStart: 'rgba(16,185,129,0.20)',
};

const neoShadow: ViewStyle = {
  shadowColor: '#000000',
  shadowOffset: { width: 4, height: 4 },
  shadowOpacity: 0.25,
  shadowRadius: 0,
  elevation: 6,
};

const softShadow: ViewStyle = {
  shadowColor: '#0f172a',
  shadowOffset: { width: 0, height: 6 },
  shadowOpacity: 0.08,
  shadowRadius: 10,
  elevation: 2,
};

export type Theme = {
  name: ThemeName;
  isDark: boolean;
  isNeo: boolean;
  colors: typeof darkColors;
  spacing: typeof Spacing;
  radius: typeof RadiusDefault;
  fontSize: typeof FontSize;
  fonts: typeof baseFonts;
  effects: {
    cardShadow: ViewStyle;
    buttonShadow: ViewStyle;
  };
};

const darkTheme: Theme = {
  name: 'dark',
  isDark: true,
  isNeo: false,
  colors: darkColors,
  spacing: Spacing,
  radius: RadiusDefault,
  fontSize: FontSize,
  fonts: baseFonts,
  effects: {
    cardShadow: softShadow,
    buttonShadow: softShadow,
  },
};

const lightTheme: Theme = {
  name: 'light',
  isDark: false,
  isNeo: false,
  colors: lightColors,
  spacing: Spacing,
  radius: RadiusDefault,
  fontSize: FontSize,
  fonts: baseFonts,
  effects: {
    cardShadow: softShadow,
    buttonShadow: softShadow,
  },
};

const neoTheme: Theme = {
  name: 'neo',
  isDark: false,
  isNeo: true,
  colors: neoColors,
  spacing: Spacing,
  radius: RadiusNeo,
  fontSize: FontSize,
  fonts: neoFonts,
  effects: {
    cardShadow: neoShadow,
    buttonShadow: neoShadow,
  },
};

function resolveTheme(name: ThemeName, systemScheme: ColorSchemeName): Theme {
  if (name === 'neo') return neoTheme;
  if (name === 'light') return lightTheme;
  if (name === 'dark') return darkTheme;
  return systemScheme === 'light' ? lightTheme : darkTheme;
}

type ThemeContextValue = {
  theme: Theme;
  themeName: ThemeName;
  setThemeName: (name: ThemeName) => void;
  ready: boolean;
};

const ThemeContext = createContext<ThemeContextValue>({
  theme: darkTheme,
  themeName: 'default',
  setThemeName: () => {},
  ready: false,
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemScheme = useColorScheme();
  const [themeName, setThemeNameState] = useState<ThemeName>('default');
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const stored = await EncryptedStorage.getItem(THEME_KEY);
        if (stored === 'default' || stored === 'dark' || stored === 'light' || stored === 'neo') {
          setThemeNameState(stored);
        }
      } catch {
        // ignore
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const theme = useMemo(() => resolveTheme(themeName, systemScheme), [themeName, systemScheme]);

  function setThemeName(name: ThemeName) {
    setThemeNameState(name);
    EncryptedStorage.setItem(THEME_KEY, name).catch(() => {});
  }

  return (
    <ThemeContext.Provider value={{ theme, themeName, setThemeName, ready }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
