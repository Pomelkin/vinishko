import { createStore } from "zustand/vanilla";
import {
  RecognitionSchema,
  type RecognitionService,
} from "../../shared/api/contracts";
import { messageFor, UserError } from "../../shared/api/errors";
import { requestId } from "../../shared/api/request-id";
import {
  saveSession,
  storageWarning,
  type ScanSession,
} from "../../shared/storage/sessions";
import type { PreparedImage } from "../image-input/image";
export type ScanPhase =
  | "idle"
  | "camera"
  | "preparing"
  | "recognizing"
  | "slow"
  | "success"
  | "error";
interface ScanState {
  phase: ScanPhase;
  error: string | null;
  session: ScanSession | null;
  storageNotice: string | null;
  operationId: string | null;
}
/** Через столько ожидания ответа показывается «Сканирование заняло больше времени чем обычно». */
export const SLOW_MS = 60_000;
/** В демо короче: сценарий «Долгое ожидание» длится 14 с и должен успеть показать это окно. */
export const DEMO_SLOW_MS = 6_000;
export class ScanController {
  readonly store = createStore<ScanState>(() => ({
    phase: "idle",
    error: null,
    session: null,
    storageNotice: null,
    operationId: null,
  }));
  private abort: AbortController | null = null;
  private slowTimer: ReturnType<typeof setTimeout> | undefined;
  private previous: {
    prepared: PreparedImage;
    source: "demo" | "upload";
  } | null = null;
  private input: {
    prepare: () => Promise<PreparedImage>;
    source: "demo" | "upload";
  } | null = null;
  constructor(
    private service: RecognitionService,
    private persist = saveSession,
    private slowMs = SLOW_MS,
  ) {}
  setPhase(phase: "idle" | "camera") {
    this.store.setState({ phase });
  }
  private armSlow(id: string) {
    clearTimeout(this.slowTimer);
    this.slowTimer = setTimeout(() => {
      if (this.store.getState().operationId === id)
        this.store.setState({ phase: "slow" });
    }, this.slowMs);
  }
  cancel() {
    this.abort?.abort();
    this.abort = null;
    clearTimeout(this.slowTimer);
    this.store.setState({ phase: "idle", operationId: null, error: null });
  }
  keepWaiting() {
    const id = this.store.getState().operationId;
    if (id && this.abort && !this.abort.signal.aborted) {
      this.store.setState({ phase: "recognizing" });
      this.armSlow(id);
    }
  }
  retry() {
    const previous = this.previous;
    if (previous)
      return this.start(async () => previous.prepared, previous.source);
    if (this.input) return this.start(this.input.prepare, this.input.source);
    return Promise.resolve(null);
  }
  async start(
    prepare: () => Promise<PreparedImage>,
    source: "demo" | "upload",
  ): Promise<ScanSession | null> {
    if (
      ["preparing", "recognizing", "slow"].includes(this.store.getState().phase)
    )
      return null;
    this.previous = null;
    this.input = { prepare, source };
    this.abort?.abort();
    const abort = new AbortController();
    this.abort = abort;
    const id = requestId();
    this.store.setState({ phase: "preparing", error: null, operationId: id });
    try {
      const prepared = await prepare();
      if (this.store.getState().operationId !== id || abort.signal.aborted)
        return null;
      this.previous = { prepared, source };
      this.store.setState({ phase: "recognizing" });
      this.armSlow(id);
      const raw = await this.service.recognize(prepared.blob, {
        signal: abort.signal,
      });
      if (this.store.getState().operationId !== id || abort.signal.aborted)
        return null;
      const parsed = RecognitionSchema.safeParse(raw);
      if (
        !parsed.success ||
        parsed.data.image.width !== prepared.width ||
        parsed.data.image.height !== prepared.height
      )
        throw new UserError(
          "Сервис вернул некорректный ответ. Попробуйте ещё раз.",
        );
      clearTimeout(this.slowTimer);
      const session: ScanSession = {
        scanId: id,
        imageBlob: prepared.blob,
        imageDimensions: { width: prepared.width, height: prepared.height },
        recognitionResult: parsed.data,
        selectedDetectionId: null,
        sheetView: "closed",
        selectedSimilarWineSlug: null,
        relevantScrollPositions: { page: 0, similar: 0, sheet: 0 },
        createdAt: Date.now(),
        source,
      };
      const saved = await this.persist(session);
      if (this.store.getState().operationId !== id || abort.signal.aborted)
        return null;
      this.abort = null;
      this.store.setState({
        phase: "success",
        session,
        storageNotice: saved ? null : storageWarning,
        operationId: null,
      });
      return session;
    } catch (error) {
      if (this.store.getState().operationId !== id || abort.signal.aborted)
        return null;
      clearTimeout(this.slowTimer);
      this.abort = null;
      this.store.setState({
        phase: "error",
        error: messageFor(error),
        operationId: null,
      });
      return null;
    }
  }
}
