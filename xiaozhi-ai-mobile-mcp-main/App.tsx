import React from 'react';
import { StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import AppNavigator from './src/navigation/AppNavigator';
import { ThemeProvider, useTheme } from './src/theme/theme';

interface ErrorBoundaryState {
  hasError: boolean;
}

class AppErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
  };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error('Unhandled app error:', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.errorWrap}>
          <Text style={styles.errorTitle}>Aplikasi Perlu Dimuat Ulang</Text>
          <Text style={styles.errorText}>
            Kami menahan error JavaScript agar aplikasi tidak langsung crash. Tekan tombol di bawah untuk memuat ulang tampilan.
          </Text>
          <TouchableOpacity style={styles.errorBtn} onPress={() => this.setState({ hasError: false })}>
            <Text style={styles.errorBtnText}>Coba Lagi</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

function AppShell() {
  const { theme } = useTheme();

  return (
    <>
      <StatusBar
        barStyle={theme.isDark ? 'light-content' : 'dark-content'}
        backgroundColor={theme.colors.bg}
      />
      <AppNavigator />
    </>
  );
}

export default function App(): React.JSX.Element {
  return (
    <AppErrorBoundary>
      <ThemeProvider>
        <AppShell />
      </ThemeProvider>
    </AppErrorBoundary>
  );
}

const styles = StyleSheet.create({
  errorWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: '#111827',
  },
  errorTitle: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 12,
    textAlign: 'center',
  },
  errorText: {
    color: '#d1d5db',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  errorBtn: {
    marginTop: 18,
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: '#2563eb',
  },
  errorBtnText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 15,
  },
});
