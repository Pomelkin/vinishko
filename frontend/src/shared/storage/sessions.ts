import { RecognitionSchema, type RecognitionResponse } from "../api/contracts";
export type SheetView = "closed" | "match" | "similar" | "candidate";
export interface ScanSession {
  scanId: string;
  imageBlob: Blob;
  imageDimensions: { width: number; height: number };
  recognitionResult: RecognitionResponse;
  selectedDetectionId: string | null;
  sheetView: SheetView;
  selectedSimilarWineSlug: string | null;
  relevantScrollPositions: { page: number; similar: number; sheet: number };
  createdAt: number;
  source: "demo" | "upload";
}
const memory = new Map<string, ScanSession>();
const MAX_AGE = 7 * 24 * 60 * 60 * 1000,
  MAX_SESSIONS = 10;
let pending: Promise<void> = Promise.resolve();
export const storageWarning =
  "Браузер не разрешил сохранить фото. Оно доступно в этой вкладке, но может исчезнуть после перезагрузки.";
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("svoe-vino-scans", 1);
    request.onupgradeneeded = () =>
      request.result.createObjectStore("sessions", { keyPath: "scanId" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("Storage blocked"));
  });
}
async function transaction<T>(
  mode: IDBTransactionMode,
  action: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction("sessions", mode);
      const request = action(tx.objectStore("sessions"));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}
export function saveSession(session: ScanSession): Promise<boolean> {
  memory.set(session.scanId, session);
  // Serialize writes so rapid sheet transitions cannot persist an older snapshot last.
  const write = pending.then(async () => {
    try {
      await transaction("readwrite", (s) => s.put(session));
      return true;
    } catch {
      return false;
    }
  });
  pending = write.then(() => undefined);
  return write;
}
export async function loadSession(id: string): Promise<ScanSession | null> {
  await pending;
  const cached = memory.get(id);
  if (cached && Date.now() - cached.createdAt < MAX_AGE) return cached;
  try {
    const stored = await transaction<ScanSession | undefined>("readonly", (s) =>
      s.get(id),
    );
    if (
      !stored ||
      Date.now() - stored.createdAt > MAX_AGE ||
      !(stored.imageBlob instanceof Blob)
    )
      return null;
    const parsed = RecognitionSchema.safeParse(stored.recognitionResult);
    if (!parsed.success) return null;
    const result = { ...stored, recognitionResult: parsed.data };
    memory.set(id, result);
    return result;
  } catch {
    return null;
  }
}
export async function cleanupSessions(): Promise<void> {
  try {
    const all = await transaction<ScanSession[]>("readonly", (s) => s.getAll());
    const sorted = all.sort((a, b) => b.createdAt - a.createdAt);
    for (const [i, s] of sorted.entries())
      if (i >= MAX_SESSIONS || Date.now() - s.createdAt > MAX_AGE) {
        await transaction("readwrite", (store) => store.delete(s.scanId));
        memory.delete(s.scanId);
      }
  } catch {
    /* Storage is optional; saveSession surfaces failures when they matter. */
  }
}
export function clearMemoryCache() {
  memory.clear();
}
