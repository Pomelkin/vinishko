import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  UnknownWineSchema,
  type BottleDetection,
} from "../../shared/api/contracts";
import { parse, request } from "../../shared/api/client";
import { isAbort, messageFor } from "../../shared/api/errors";
import { SommelierChat } from "../sommelier/SommelierChat";

type Attributes = z.infer<typeof UnknownWineSchema>;
async function cropBottle(
  image: Blob,
  detection?: BottleDetection,
): Promise<Blob> {
  const bitmap = await createImageBitmap(image);
  try {
    const points = detection
      ? (detection.polygons || [detection.polygon]).flat()
      : [
          [0, 0],
          [1, 1],
        ];
    const x = Math.max(
      0,
      Math.floor(Math.min(...points.map((p) => p[0])) * bitmap.width) - 12,
    );
    const y = Math.max(
      0,
      Math.floor(Math.min(...points.map((p) => p[1])) * bitmap.height) - 12,
    );
    const width = Math.min(
      bitmap.width - x,
      Math.ceil(Math.max(...points.map((p) => p[0])) * bitmap.width) + 12 - x,
    );
    const height = Math.min(
      bitmap.height - y,
      Math.ceil(Math.max(...points.map((p) => p[1])) * bitmap.height) + 12 - y,
    );
    const scale = Math.min(1, 1600 / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas unavailable");
    ctx.drawImage(
      bitmap,
      x,
      y,
      width,
      height,
      0,
      0,
      canvas.width,
      canvas.height,
    );
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("Image encoding failed")),
        "image/jpeg",
        0.9,
      ),
    );
  } finally {
    bitmap.close();
  }
}

export function UnknownWine({
  image,
  detection,
  contextId,
  onResult,
}: {
  image: Blob;
  detection?: BottleDetection;
  contextId: string;
  onResult?: (value: Attributes) => void;
}) {
  const [result, setResult] = useState<Attributes | null>(
    detection?.unknownWine || null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(
    detection?.unknownWineError || null,
  );
  const flight = useRef<AbortController | null>(null);
  useEffect(() => () => flight.current?.abort(), []);
  const predict = async () => {
    if (flight.current) return;
    const controller = new AbortController();
    flight.current = controller;
    setBusy(true);
    setError(null);
    try {
      const blob = await cropBottle(image, detection);
      if (controller.signal.aborted) return;
      const body = new FormData();
      body.append("image", blob, "bottle.jpg");
      const attributes = parse(
        UnknownWineSchema,
        await request("/whatis", {
          method: "POST",
          body,
          signal: controller.signal,
        }),
      );
      if (!controller.signal.aborted) {
        setResult(attributes);
        onResult?.(attributes);
      }
    } catch (e) {
      if (!isAbort(e) && !controller.signal.aborted) setError(messageFor(e));
    } finally {
      flight.current = null;
      if (!controller.signal.aborted) setBusy(false);
    }
  };
  return (
    <section>
      <h3>Что можно узнать по этикетке</h3>
      <p className="muted">
        Определим категорию и винодельню по фото. Это не подтверждает совпадение
        с каталогом.
      </p>
      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}
      {result ? (
        <>
          <p>
            Категория: <strong>{result.category}</strong>
            <br />
            Винодельня: <strong>{result.brand}</strong>
          </p>
          <SommelierChat
            key={contextId}
            contextId={contextId}
            wine={{
              Категория: result.category,
              Винодельня: result.brand,
              Примечание:
                "Вино не определено в каталоге; признаки предположены по фото",
            }}
          />
        </>
      ) : (
        <button
          className="secondary"
          disabled={busy}
          onClick={() => void predict()}
        >
          {busy ? "Изучаем этикетку…" : "Определить признаки"}
        </button>
      )}
    </section>
  );
}
