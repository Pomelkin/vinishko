export class UserError extends Error {}
export function messageFor(error: unknown): string {
  if (error instanceof UserError) return error.message;
  if (typeof navigator !== "undefined" && !navigator.onLine)
    return "Нет подключения к сети. Проверьте интернет и попробуйте ещё раз.";
  return "Не удалось получить результат. Попробуйте ещё раз — фотография сохранена.";
}
export const isAbort = (e: unknown) =>
  e instanceof Error && e.name === "AbortError";
