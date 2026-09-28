import { Component, type ReactNode } from "react";
import { BrowserRouter, Route, Routes, Link } from "react-router-dom";
import { ScanProvider } from "./ScanProvider";
import { ScannerPage } from "../pages/ScannerPage/ScannerPage";
import { ScanResultPage } from "../pages/ScanResultPage/ScanResultPage";
import { WinePage } from "../pages/WinePage/WinePage";
class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <main className="empty">
        <h1>Не удалось открыть страницу</h1>
        <p>Обновите страницу, чтобы попробовать ещё раз.</p>
        <button className="primary" onClick={() => window.location.reload()}>
          Обновить
        </button>
      </main>
    ) : (
      this.props.children
    );
  }
}
export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <ScanProvider>
          <Routes>
            <Route path="/" element={<ScannerPage />} />
            <Route path="/scan/:scanId" element={<ScanResultPage />} />
            <Route path="/wine/:slug" element={<WinePage />} />
            <Route
              path="*"
              element={
                <main className="empty">
                  <h1>Страница не найдена</h1>
                  <Link className="primary" to="/">
                    К сканеру
                  </Link>
                </main>
              }
            />
          </Routes>
        </ScanProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
