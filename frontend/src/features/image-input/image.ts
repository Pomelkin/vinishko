import { UserError } from "../../shared/api/errors";
export const MAX_BYTES = 20 * 1024 * 1024;
export interface PreparedImage {
  blob: Blob;
  width: number;
  height: number;
}
export async function decodeImage(blob: Blob): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } catch {
    throw new UserError(
      "Не удалось прочитать фото. Возможно, файл повреждён. Выберите другое изображение.",
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}
export async function prepareImage(
  file: Blob,
  internalDemo = false,
): Promise<PreparedImage> {
  if (
    !internalDemo &&
    !["image/jpeg", "image/png", "image/webp"].includes(file.type)
  ) {
    throw new UserError(
      "Выберите фото в формате JPEG, PNG или WebP. HEIC/HEIF нужно сначала экспортировать в JPEG или PNG.",
    );
  }
  if (file.size > MAX_BYTES)
    throw new UserError(
      "Фото слишком большое. Выберите файл размером до 20 МБ.",
    );
  if (!file.size) throw new UserError("Файл пуст. Выберите другое фото.");
  // Modern browser image decoding applies EXIF orientation before drawing.
  const image = await decodeImage(file);
  if (image.naturalWidth * image.naturalHeight > 80_000_000)
    throw new UserError(
      "Слишком высокое разрешение фотографии. Уменьшите её и попробуйте снова.",
    );
  const scale = Math.min(
    1,
    2400 / Math.max(image.naturalWidth, image.naturalHeight),
  );
  const width = Math.round(image.naturalWidth * scale),
    height = Math.round(image.naturalHeight * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx)
    throw new UserError(
      "Браузер не смог обработать фото. Попробуйте другой браузер.",
    );
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (b) =>
        b ? resolve(b) : reject(new UserError("Не удалось сохранить фото.")),
      "image/jpeg",
      0.92,
    ),
  );
  canvas.width = 0;
  canvas.height = 0;
  return { blob, width, height };
}
