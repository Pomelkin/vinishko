import {
  createContext,
  useContext,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { useStore } from "zustand";
import { useLocation, useNavigate } from "react-router-dom";
import {
  DEMO_SLOW_MS,
  ScanController,
  SLOW_MS,
} from "../features/scanning/controller";
import { service, isMockMode, type DemoScenario } from "../shared/api/service";
import { MockRecognitionService } from "../shared/api/mock-recognition-service";
import { prepareImage } from "../features/image-input/image";
import { Camera } from "../features/image-input/Camera";
import { Icon } from "../shared/ui/Icon";
import { cleanupSessions } from "../shared/storage/sessions";
import s from "./Shell.module.css";
export const scanController = new ScanController(
  service,
  undefined,
  isMockMode ? DEMO_SLOW_MS : SLOW_MS,
);
interface Actions {
  upload: () => void;
  camera: () => void;
  demo: (scenario: DemoScenario) => void;
}
const Context = createContext<Actions | null>(null);
export const useScan = () => useContext(Context)!;
export function ScanProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate(),
    location = useLocation(),
    file = useRef<HTMLInputElement>(null),
    capture = useRef<HTMLInputElement>(null);
  const { phase, error, storageNotice } = useStore(scanController.store);
  useEffect(() => {
    void cleanupSessions();
  }, []);
  useEffect(() => () => scanController.cancel(), [location.pathname]);
  const showResult = (
    session: Awaited<ReturnType<ScanController["start"]>>,
  ) => {
    if (session) navigate(`/scan/${session.scanId}`);
  };
  const upload = () => {
    if (phase === "camera") scanController.cancel();
    file.current?.click();
  };
  const native = () => {
    scanController.cancel();
    capture.current?.click();
  };
  const camera = () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext)
      native();
    else scanController.setPhase("camera");
  };
  const useImage = (blob: Blob) => {
    scanController.cancel();
    void scanController
      .start(() => prepareImage(blob), "upload")
      .then(showResult);
  };
  const demo = (scenario: DemoScenario) => {
    const mock = service;
    if (mock instanceof MockRecognitionService)
      void scanController
        .start(() => mock.createDemo(scenario), "demo")
        .then(showResult);
  };
  const busy = ["preparing", "recognizing", "slow", "error"].includes(phase);
  useEffect(() => {
    if (!busy) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = overflow;
    };
  }, [busy]);
  return (
    <Context.Provider value={{ upload, camera, demo }}>
      <div inert={busy || phase === "camera"}>
        {children}
        {storageNotice && (
          <div className={s.storage} role="status">
            {storageNotice}
          </div>
        )}
      </div>
      <input
        ref={file}
        className="sr-only"
        tabIndex={-1}
        type="file"
        accept="image/jpeg,image/png,image/webp,.heic,.heif"
        aria-label="Выбрать фото"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) useImage(f);
        }}
      />
      <input
        ref={capture}
        className="sr-only"
        tabIndex={-1}
        type="file"
        accept="image/*"
        capture="environment"
        aria-label="Камера телефона"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) useImage(f);
        }}
      />
      {phase === "camera" && (
        <Camera
          onClose={() => scanController.cancel()}
          onGallery={upload}
          onNative={native}
          onUse={useImage}
        />
      )}
      {busy && (
        <Process
          phase={phase}
          error={error}
          onCancel={() => {
            scanController.cancel();
            navigate("/");
          }}
          onRetry={() => void scanController.retry().then(showResult)}
          onSearch={() => {
            scanController.cancel();
            navigate("/?search=1");
          }}
        />
      )}
      {isMockMode && (
        <span className="sr-only">
          Демонстрационный режим. Реальное распознавание не подключено.
        </span>
      )}
    </Context.Provider>
  );
}
function Process({
  phase,
  error,
  onCancel,
  onRetry,
  onSearch,
}: {
  phase: string;
  error: string | null;
  onCancel: () => void;
  onRetry: () => void;
  onSearch: () => void;
}) {
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement;
    first.current?.focus();
    return () => {
      if (prev?.isConnected) prev.focus();
    };
  }, []);
  useEffect(() => {
    first.current?.focus();
  }, [phase === "slow" || phase === "error"]);
  const issue = phase === "slow" || phase === "error";
  return (
    <section
      className={s.process}
      role="dialog"
      aria-modal="true"
      aria-label="Сканирование фотографии"
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
        if (e.key === "Tab") {
          const buttons = [
            ...e.currentTarget.querySelectorAll<HTMLButtonElement>("button"),
          ];
          const first = buttons[0],
            last = buttons.at(-1);
          if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last?.focus();
          } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first?.focus();
          }
        }
      }}
    >
      {!issue && (
        <header>
          <button ref={first} className="text-button" onClick={onCancel}>
            <Icon name="close" />
            Отменить
          </button>
        </header>
      )}
      {issue ? (
        <div className={s.issue} aria-live="polite">
          <h2>
            {phase === "slow"
              ? "Сканирование заняло больше времени чем обычно"
              : "Не удалось завершить сканирование"}
          </h2>
          <p>
            {phase === "slow"
              ? "Можно подождать ещё, попробовать повторить позже или воспользоваться поиском по названию"
              : error}
          </p>
          <div className="stack">
            <button
              ref={first}
              className="primary"
              onClick={
                phase === "slow" ? () => scanController.keepWaiting() : onRetry
              }
            >
              {phase === "slow" ? "Подождать еще" : "Повторить"}
            </button>
            <button className="secondary" onClick={onSearch}>
              Найти по названию
            </button>
            <button className="secondary" onClick={onCancel}>
              Отменить
            </button>
          </div>
        </div>
      ) : (
        <div className={s.loading} role="status">
          <img src="/assets/loader.svg" alt="" />
          <p>
            {phase === "preparing"
              ? "Подготавливаем фотографию"
              : "Еще немного времени"}
          </p>
        </div>
      )}
    </section>
  );
}
