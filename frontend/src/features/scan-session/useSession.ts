import { useCallback, useEffect, useRef, useState } from "react";
import {
  loadSession,
  saveSession,
  storageWarning,
  type ScanSession,
} from "../../shared/storage/sessions";
export function useSession(id: string) {
  const [session, setSession] = useState<ScanSession | null>(null),
    [loading, setLoading] = useState(true),
    [warning, setWarning] = useState<string | null>(null);
  const ref = useRef<ScanSession | null>(null);
  useEffect(() => {
    let active = true;
    setLoading(true);
    loadSession(id).then((value) => {
      if (active) {
        ref.current = value;
        setSession(value);
        setLoading(false);
      }
    });
    return () => {
      active = false;
    };
  }, [id]);
  const update = useCallback(async (patch: Partial<ScanSession>) => {
    if (!ref.current) return;
    const value = { ...ref.current, ...patch };
    ref.current = value;
    setSession(value);
    if (!(await saveSession(value))) setWarning(storageWarning);
  }, []);
  return { session, loading, warning, update };
}
