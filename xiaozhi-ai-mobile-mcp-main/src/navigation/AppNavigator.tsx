import React, { useEffect, useMemo, useState } from 'react';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import LoginScreen from '../screens/LoginScreen';
import HomeScreen from '../screens/HomeScreen';
import ModeEditorScreen from '../screens/ModeEditorScreen';
import DevicesScreen from '../screens/DevicesScreen';
import MessagesScreen from '../screens/MessagesScreen';
import AccountScreen from '../screens/AccountScreen';
import DeviceDetailScreen from '../screens/DeviceDetailScreen';
import QrScannerScreen from '../screens/QrScannerScreen';
import McpTokenScreen from '../screens/McpTokenScreen';
import PermissionGateScreen from '../screens/PermissionGateScreen';
import { authStore } from '../stores/authStore';
import { resumeTrackingIfEnabled } from '../services/deviceService';
import { useTheme } from '../theme/theme';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function MainTabs() {
  const { theme } = useTheme();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.panelBorder,
          borderTopWidth: theme.isNeo ? 2 : 1,
          height: 62,
          paddingBottom: 8,
          paddingTop: 6,
        },
        tabBarActiveTintColor: theme.colors.accentLight,
        tabBarInactiveTintColor: theme.colors.textMuted,
        tabBarLabelStyle: {
          fontSize: 11,
          fontFamily: theme.fonts.body,
        },
        tabBarIcon: ({ color, size }) => {
          const iconMap: Record<string, string> = {
            Home: 'home-variant',
            Mode: 'pencil-box-outline',
            Devices: 'map-marker-radius-outline',
            Messages: 'message-text-outline',
            Account: 'account-cog-outline',
          };
          const name = iconMap[route.name] || 'circle';
          return <Icon name={name} size={size || 22} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Mode" component={ModeEditorScreen} />
      <Tab.Screen name="Devices" component={DevicesScreen} />
      <Tab.Screen name="Messages" component={MessagesScreen} />
      <Tab.Screen name="Account" component={AccountScreen} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  const { theme } = useTheme();
  const [token, setToken] = useState<string | null>(null);
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [permissionsReady, setPermissionsReady] = useState(false);

  useEffect(() => {
    (async () => {
      const t = await authStore.getToken();
      setToken(t);
      setPermissionsReady(false);
      setCheckingAuth(false);
    })();

    const unsub = authStore.subscribe((t) => {
      setToken(t);
      setPermissionsReady(false);
    });

    return () => {
      unsub();
    };
  }, []);

  const navTheme = useMemo(
    () => ({
      ...DefaultTheme,
      colors: {
        ...DefaultTheme.colors,
        background: theme.colors.bg,
        card: theme.colors.surface,
        text: theme.colors.text,
        primary: theme.colors.accent,
        border: theme.colors.panelBorder,
      },
    }),
    [theme],
  );

  useEffect(() => {
    if (!token || !permissionsReady) {
      return;
    }

    const timer = setTimeout(() => {
      resumeTrackingIfEnabled().catch(() => {});
    }, 1600);

    return () => clearTimeout(timer);
  }, [token, permissionsReady]);

  if (checkingAuth) return null;

  return (
    <NavigationContainer theme={navTheme}>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {token ? (
          permissionsReady ? (
            <>
              <Stack.Screen name="Main" component={MainTabs} />
              <Stack.Screen name="DeviceDetail" component={DeviceDetailScreen} />
              <Stack.Screen name="QrScanner" component={QrScannerScreen} />
              <Stack.Screen name="McpToken" component={McpTokenScreen} />
            </>
          ) : (
            <Stack.Screen name="PermissionGate">
              {() => <PermissionGateScreen onReady={() => setPermissionsReady(true)} />}
            </Stack.Screen>
          )
        ) : (
          <Stack.Screen name="Login" component={LoginScreen} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
