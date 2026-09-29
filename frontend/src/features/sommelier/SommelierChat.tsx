import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import { ApiError, parse, request } from "../../shared/api/client";
import { isAbort, messageFor } from "../../shared/api/errors";
import s from "./SommelierChat.module.css";

type Card = Record<string, string | number | boolean | null>;
const Message = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string(),
  suggestions: z.array(z.string()).nullable(),
});
const Turn = z.object({ session_id: z.string(), message: Message });
const History = z.object({
  session_id: z.string(),
  messages: z.array(Message),
});
type ChatMessage = z.infer<typeof Message>;

function uuid7() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  let stamp = Date.now();
  for (let i = 5; i >= 0; i--) {
    bytes[i] = stamp % 256;
    stamp = Math.floor(stamp / 256);
  }
  bytes[6] = (bytes[6] & 15) | 112;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join(
    "",
  );
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function SommelierChat({
  contextId,
  wine,
  candidates = [],
}: {
  contextId: string;
  wine: Card;
  candidates?: Card[];
}) {
  const storageKey = `vino-sommelier:${contextId}`;
  const [id, setId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uncertain, setUncertain] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const flight = useRef<AbortController | null>(null);
  const remember = (value: string | null) => {
    setId(value);
    try {
      if (value) localStorage.setItem(storageKey, value);
      else localStorage.removeItem(storageKey);
    } catch {
      setWarning(
        "История доступна в этой вкладке; браузер запретил сохранение номера диалога.",
      );
    }
  };
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(storageKey);
    } catch {
      /* История остаётся на сервере. */
    }
    if (
      saved &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
        saved,
      )
    ) {
      setId(saved);
      void run((signal) => load(saved, signal));
    }
    return () => {
      flight.current?.abort();
    };
  }, [storageKey]);

  // Истории на сервере нет — диалог не успел создаться или удалён: молча начинаем заново, без ошибки.
  async function load(session: string, signal: AbortSignal) {
    try {
      const data = parse(
        History,
        await request(`/sommelier/sessions/${session}`, { signal }),
      );
      if (!signal.aborted) {
        setMessages(data.messages);
        setText("");
      }
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 404)) throw e;
      if (!signal.aborted) {
        remember(null);
        setMessages([]);
      }
    }
  }
  async function run(action: (signal: AbortSignal) => Promise<void>) {
    if (flight.current && !flight.current.signal.aborted) return;
    const controller = new AbortController();
    flight.current = controller;
    setBusy(true);
    setError(null);
    try {
      await action(controller.signal);
      if (!controller.signal.aborted) setUncertain(false);
    } catch (e) {
      if (!isAbort(e) && !controller.signal.aborted) {
        if (e instanceof ApiError && e.status === 404) {
          remember(null);
          setMessages([]);
          setUncertain(false);
        } else {
          setUncertain(true);
        }
        setError(messageFor(e));
      }
    } finally {
      if (flight.current === controller) {
        flight.current = null;
        if (!controller.signal.aborted) setBusy(false);
      }
    }
  }
  const start = () =>
    run(async (signal) => {
      const session = uuid7();
      remember(session);
      const data = parse(
        Turn,
        await request(`/sommelier/sessions/${session}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal,
          body: JSON.stringify({ wine, candidates: candidates.slice(0, 5) }),
        }),
      );
      if (!signal.aborted) setMessages([data.message]);
    });
  const send = (content: string) =>
    run(async (signal) => {
      const data = parse(
        Turn,
        await request(`/sommelier/sessions/${id}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal,
          body: JSON.stringify({ content: content.trim() }),
        }),
      );
      if (!signal.aborted) {
        setMessages((previous) => [
          ...previous,
          { role: "user", content: content.trim(), suggestions: null },
          data.message,
        ]);
        setText("");
      }
    });
  const restore = (session: string) => run((signal) => load(session, signal));
  const remove = () =>
    run(async (signal) => {
      await request(`/sommelier/sessions/${id}`, { method: "DELETE", signal });
      if (!signal.aborted) {
        remember(null);
        setMessages([]);
        setText("");
      }
    });
  // Ответ не дошёл, но запрос мог выполниться на сервере: история подтягивается сама, без кнопки.
  // Неудача оставляет uncertain как есть, поэтому повтор один; дальше поможет перезагрузка — история загрузится при открытии.
  useEffect(() => {
    if (uncertain && id) void restore(id);
  }, [uncertain, id]);
  const suggestions = messages.at(-1)?.suggestions || [];
  return (
    <section className={s.chat} aria-label="AI-сомелье">
      <p className="eyebrow">К вашему вину</p>
      <h2>Спросите сомелье</h2>
      <p className="muted">
        Сочетания с блюдами, температура подачи и особенности вина.
      </p>
      {warning && <p className="notice">{warning}</p>}
      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}
      <div className={s.messages} role="log" aria-live="polite">
        {messages.map((message, i) => (
          <div
            key={i}
            className={message.role === "user" ? s.user : s.assistant}
          >
            <strong>{message.role === "user" ? "Вы" : "AI-сомелье"}</strong>
            <p>{message.content}</p>
          </div>
        ))}
      </div>
      {busy && !uncertain && <p role="status">Сомелье готовит ответ…</p>}
      {!id ? (
        <button
          className="primary"
          disabled={busy}
          onClick={() => void start()}
        >
          Получить совет
        </button>
      ) : (
        <>
          {!uncertain && !!messages.length && (
            <>
              <div className={s.suggestions}>
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    className="secondary"
                    disabled={busy}
                    onClick={() => void send(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  if (text.trim() && !busy) void send(text);
                }}
              >
                <label htmlFor={`question-${contextId}`}>Ваш вопрос</label>
                <textarea
                  id={`question-${contextId}`}
                  value={text}
                  maxLength={4000}
                  disabled={busy}
                  onChange={(event) => setText(event.target.value)}
                  placeholder="С чем подать это вино?"
                />
                <button className="primary" disabled={busy || !text.trim()}>
                  Отправить
                </button>
              </form>
            </>
          )}
          {uncertain && (
            <p className="muted" role="status">
              {busy
                ? "Связь прервалась, обновляем диалог…"
                : "Не удалось обновить диалог. Обновите страницу — история загрузится сама."}
            </p>
          )}
          <div className={s.actions}>
            <button
              className="text-button"
              disabled={busy}
              onClick={() => void remove()}
            >
              Удалить диалог
            </button>
          </div>
        </>
      )}
    </section>
  );
}
