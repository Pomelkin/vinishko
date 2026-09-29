import { z } from "zod";
import { UserError } from "./errors";

export const apiBase = (import.meta.env.VITE_API_BASE_URL || "/api").replace(
  /\/$/,
  "",
);
export class ApiError extends UserError {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function request(
  path: string,
  init: RequestInit = {},
  base = apiBase,
): Promise<unknown> {
  if (!navigator.onLine)
    throw new UserError("Нет подключения к сети. Проверьте интернет.");
  const response = await fetch(`${base}${path}`, init);
  if (!response.ok) {
    const messages: Record<number, string> = {
      404: "Запись не найдена.",
      409: "Диалог уже существует. Загрузите историю.",
      413: "Фотография слишком большая. Выберите снимок меньшего размера.",
      415: "Формат фото не поддерживается. Выберите JPEG, PNG или WebP.",
      422: "Проверьте введённые данные.",
      502: "Сервис не смог получить ответ. Попробуйте позже.",
      503: "Сервис пока недоступен. Попробуйте позже.",
      504: "Сервис не успел ответить. Для диалога загрузите историю перед повтором.",
    };
    throw new ApiError(
      messages[response.status] ||
        "Не удалось выполнить запрос. Попробуйте позже.",
      response.status,
    );
  }
  if (response.status === 204) return undefined;
  try {
    return await response.json();
  } catch {
    throw new UserError("Сервис вернул некорректный ответ.");
  }
}
export function parse<T>(schema: z.ZodType<T>, data: unknown): T {
  const result = schema.safeParse(data);
  if (!result.success) throw new UserError("Сервис вернул некорректный ответ.");
  return result.data;
}
