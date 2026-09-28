import { useEffect, useRef, useState } from "react";
import { Icon } from "../../shared/ui/Icon";
import { useBlobUrl } from "../../shared/ui/useBlobUrl";
import s from "./Camera.module.css";
export function Camera({
  onClose,
  onUse,
  onGallery,
  onNative,
}: {
  onClose: () => void;
  onUse: (blob: Blob) => void;
  onGallery: () => void;
  onNative: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null),
    stream = useRef<MediaStream | null>(null),
    dialog = useRef<HTMLDialogElement>(null);
  const [shot, setShot] = useState<Blob | null>(null),
    [error, setError] = useState<string | null>(null),
    [ready, setReady] = useState(false),
    [generation, setGeneration] = useState(0);
  const url = useBlobUrl(shot);
  const stop = () => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  };
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current?.showModal();
    return () => {
      stop();
      document.body.style.overflow = old;
      previous?.focus?.();
    };
  }, []);
  useEffect(() => {
    let active = true;
    setReady(false);
    setError(null);
    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      })
      .then(async (media) => {
        if (!active) {
          media.getTracks().forEach((t) => t.stop());
          return;
        }
        stream.current = media;
        if (video.current) {
          video.current.srcObject = media;
          try {
            await video.current.play();
            if (active) setReady(true);
          } catch {
            if (active)
              setError(
                "Не удалось запустить превью. Откройте камеру телефона или загрузите фото.",
              );
            stop();
          }
        }
      })
      .catch((e) => {
        if (active)
          setError(
            e.name === "NotAllowedError"
              ? "Доступ к камере не разрешён. Разрешите его в настройках браузера или загрузите фото."
              : e.name === "NotFoundError"
                ? "Камера не найдена. Можно загрузить готовую фотографию."
                : "Камера сейчас недоступна. Попробуйте камеру телефона или загрузите фото.",
          );
      });
    return () => {
      active = false;
      stop();
    };
  }, [generation]);
  const capture = () => {
    if (!video.current?.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.current.videoWidth;
    canvas.height = video.current.videoHeight;
    canvas.getContext("2d")?.drawImage(video.current, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          setShot(blob);
          stop();
        } else setError("Не удалось сделать снимок. Попробуйте снова.");
      },
      "image/jpeg",
      0.94,
    );
  };
  return (
    <dialog
      ref={dialog}
      className={s.camera}
      aria-label="Камера"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <header>
        <button className="text-button" onClick={onClose}>
          <Icon name="close" />
          Отменить
        </button>
        <span>Сканер вина</span>
      </header>
      <div className={s.preview}>
        {error ? (
          <div className={s.error} role="alert">
            <Icon name="camera" size={40} />
            <p>{error}</p>
            <button className="secondary" onClick={onNative}>
              Открыть камеру телефона
            </button>
            <button className="primary" onClick={onGallery}>
              Загрузить фото
            </button>
          </div>
        ) : shot && url ? (
          <img src={url} alt="Сделанный снимок" />
        ) : (
          <>
            <video ref={video} muted playsInline />
            <div className={s.guide} />
            <p className={s.hint}>
              {ready ? "Поместите бутылку в кадр" : "Подключаем камеру…"}
            </p>
          </>
        )}
      </div>
      {!error && (
        <footer>
          {shot ? (
            <>
              <button
                className="secondary"
                onClick={() => {
                  setShot(null);
                  setGeneration((v) => v + 1);
                }}
              >
                Переснять
              </button>
              <button className="primary" onClick={() => onUse(shot)}>
                Использовать фото
              </button>
            </>
          ) : (
            <>
              <button
                className="icon-button"
                aria-label="Загрузить из галереи"
                onClick={onGallery}
              >
                <Icon name="upload" />
              </button>
              <button
                className={s.shutter}
                aria-label="Сделать снимок"
                disabled={!ready}
                onClick={capture}
              />
              <span className={s.spacer} />
            </>
          )}
        </footer>
      )}
    </dialog>
  );
}
