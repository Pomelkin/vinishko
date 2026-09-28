import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Header } from "../../app/Shell";
import { useScan } from "../../app/ScanProvider";
import { useSession } from "../../features/scan-session/useSession";
import { useBlobUrl } from "../../shared/ui/useBlobUrl";
import { BottlePhoto } from "../../features/bottle-selection/BottlePhoto";
import { Sheet } from "../../shared/ui/Sheet";
import { WineCard, WinePreview } from "../../entities/wine/WineCard";
import { Icon } from "../../shared/ui/Icon";
import type { WineSummary } from "../../shared/api/contracts";
import type { ScanSession } from "../../shared/storage/sessions";
import s from "./ScanResultPage.module.css";
export function ScanResultPage() {
  const { scanId = "" } = useParams();
  return <ScanResult key={scanId} scanId={scanId} />;
}
function ScanResult({ scanId }: { scanId: string }) {
  const navigate = useNavigate(),
    scan = useScan();
  const { session, loading, warning, update } = useSession(scanId),
    url = useBlobUrl(session?.imageBlob);
  const [saving, setSaving] = useState(false),
    restore = useRef(false),
    scroll = useRef({ page: 0, similar: 0, sheet: 0 });
  const current = useRef(session);
  current.current = session;
  const scrollTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const saveScroll = () => {
    clearTimeout(scrollTimer.current);
    scrollTimer.current = setTimeout(
      () => void update({ relevantScrollPositions: { ...scroll.current } }),
      150,
    );
  };
  useEffect(() => () => clearTimeout(scrollTimer.current), []);
  useEffect(() => {
    if (session && !restore.current) {
      restore.current = true;
      scroll.current = session.relevantScrollPositions;
      requestAnimationFrame(() =>
        window.scrollTo(0, session.relevantScrollPositions.page),
      );
    }
  }, [session]);
  useEffect(() => {
    const listener = () => {
      scroll.current.page = window.scrollY;
      saveScroll();
    };
    window.addEventListener("scroll", listener, { passive: true });
    const save = () => {
      if (current.current)
        void update({ relevantScrollPositions: { ...scroll.current } });
    };
    const visibility = () => {
      if (document.visibilityState === "hidden") save();
    };
    window.addEventListener("pagehide", save);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("scroll", listener);
      window.removeEventListener("pagehide", save);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [update]);
  if (loading)
    return (
      <div className="page-loading" role="status">
        Возвращаемся к вашему фото…
      </div>
    );
  if (!session)
    return (
      <div className={s.page}>
        <Header />
        <main className="empty">
          <h1>Фотография не сохранилась</h1>
          <p className="muted">
            Сессия могла устареть или данные браузера были очищены. Загрузите
            фото ещё раз.
          </p>
          <button className="primary" onClick={scan.upload}>
            Загрузить фото
          </button>
          <br />
          <button className="text-button" onClick={() => navigate("/")}>
            К сканеру
          </button>
        </main>
      </div>
    );
  const detections = session.recognitionResult.detections;
  const selected = detections.find((d) => d.id === session.selectedDetectionId);
  const open = session.sheetView !== "closed" && !!selected;
  const wine: WineSummary | undefined =
    session.sheetView === "candidate"
      ? selected?.similar.find(
          (c) => c.slug === session.selectedSimilarWineSlug,
        )?.wine
      : selected?.match?.wine;
  const change = async (patch: Partial<ScanSession>) =>
    update({ ...patch, relevantScrollPositions: { ...scroll.current } });
  const select = (id: string) => {
    const d = detections.find((d) => d.id === id);
    if (!d) return;
    scroll.current.similar = 0;
    scroll.current.sheet = 0;
    void change({
      selectedDetectionId: id,
      sheetView: d.status === "matched" ? "match" : "similar",
      selectedSimilarWineSlug: null,
    });
  };
  const close = () => void change({ sheetView: "closed" });
  const details = async () => {
    if (!wine || saving) return;
    setSaving(true);
    await change({});
    navigate(`/wine/${wine.slug}`, { state: { from: `/scan/${scanId}` } });
  };
  const listScroll =
    session.sheetView === "similar"
      ? scroll.current.similar
      : scroll.current.sheet;
  return (
    <div className={s.page}>
      <main>
        <div className={s.topline}>
          <button className="text-button" onClick={() => navigate("/")}>
            <Icon name="back" size={18} />К сканеру
          </button>
          <span className="pill">
            {session.source === "demo" ? "Демопример" : "Ваше фото"}
          </span>
        </div>
        <h1>{detections.length ? "Выберите бутылку" : "Ваш снимок"}</h1>
        <p className={s.hint}>
          {detections.length
            ? "Коснитесь контура, чтобы узнать больше о вине"
            : "Фотография сохранена в этом браузере"}
        </p>
        {url && (
          <BottlePhoto
            url={url}
            width={session.imageDimensions.width}
            height={session.imageDimensions.height}
            detections={detections}
            selected={session.selectedDetectionId}
            onSelect={select}
          />
        )}
        {warning && (
          <p className="notice" role="status">
            {warning}
          </p>
        )}
        {detections.length ? (
          <>
            <div className={s.bottles} aria-label="Обнаруженные бутылки">
              {detections.map((d, i) => (
                <button
                  key={d.id}
                  className={
                    d.id === session.selectedDetectionId
                      ? s.activeBottle
                      : s.bottle
                  }
                  onClick={() => select(d.id)}
                  aria-label={`Бутылка ${i + 1}: ${d.status === "matched" ? d.match.wine.name : "не найдена в каталоге"}`}
                  aria-pressed={d.id === session.selectedDetectionId}
                >
                  <span>{i + 1}</span>
                  <div>
                    <strong>
                      {d.status === "matched"
                        ? d.match.wine.name
                        : "Нет в каталоге"}
                    </strong>
                    <small>
                      {d.status === "matched"
                        ? "Посмотреть вино"
                        : d.similar.length
                          ? "Есть похожие варианты"
                          : "Посмотреть варианты поиска"}
                    </small>
                  </div>
                  <Icon name="chevron" size={18} />
                </button>
              ))}
            </div>
            <p className={s.localNote}>
              {session.source === "demo"
                ? "Подготовленный пример · контуры и результаты заданы заранее"
                : "Результат распознавания фотографии"}
            </p>
          </>
        ) : (
          <div className="empty">
            <h2>
              {session.source === "upload" &&
              session.recognitionResult.metrics.isMock
                ? "Фото готово к распознаванию"
                : "Бутылки не обнаружены"}
            </h2>
            <p className="muted">
              {session.source === "upload" &&
              session.recognitionResult.metrics.isMock
                ? "В деморежиме мы проверяем загрузку и сохранение фото. Чтобы увидеть выделение бутылок, откройте подготовленный пример."
                : "Попробуйте снять бутылку целиком при хорошем освещении или найдите вино по названию."}
            </p>
            {session.recognitionResult.metrics.isMock && (
              <button className="primary" onClick={() => scan.demo("mixed")}>
                Открыть демофото
              </button>
            )}
            <button
              className="text-button"
              onClick={() => navigate("/?search=1")}
            >
              Найти по названию
            </button>
          </div>
        )}
        <button className={`secondary ${s.newPhoto}`} onClick={scan.upload}>
          <Icon name="upload" size={18} />
          Загрузить другое фото
        </button>
      </main>
      {open && (
        <Sheet
          compact
          title={wine ? wine.name : "Эту бутылку не удалось найти в каталоге"}
          onClose={close}
          viewKey={session.sheetView}
          initialScroll={listScroll}
          onScroll={(y) => {
            if (session.sheetView === "similar") scroll.current.similar = y;
            else scroll.current.sheet = y;
            saveScroll();
          }}
          footer={
            wine ? (
              <button
                className="primary"
                disabled={saving}
                onClick={() => void details()}
              >
                Подробнее о вине <Icon name="arrow" size={18} />
              </button>
            ) : undefined
          }
        >
          {wine ? (
            <>
              {session.sheetView === "candidate" && (
                <button
                  className={s.backToSimilar}
                  onClick={() => {
                    scroll.current.sheet = 0;
                    void change({
                      sheetView: "similar",
                      selectedSimilarWineSlug: null,
                    });
                  }}
                >
                  <Icon name="back" size={16} />
                  Назад к похожим
                </button>
              )}
              <WinePreview
                wine={wine}
                label={
                  session.sheetView === "candidate"
                    ? "Похожее вино"
                    : "Вино найдено"
                }
              />
            </>
          ) : (
            <>
              <p className="eyebrow">
                Бутылка {detections.indexOf(selected!) + 1}
              </p>
              <h2 className={s.unmatchedTitle}>
                Эту бутылку не удалось найти в каталоге
              </h2>
              <p className="muted">
                {selected!.similar.length
                  ? "Но есть вина, с которыми стоит познакомиться. Это рекомендации, а не точное совпадение."
                  : "Похожих вин пока нет. Сделайте ещё одно фото или воспользуйтесь поиском по названию."}
              </p>
              {selected!.similar.length ? (
                <>
                  <div className="section-title">
                    <h2>Похожие вина</h2>
                    <span>{selected!.similar.length} варианта</span>
                  </div>
                  <div className="grid">
                    {selected!.similar.map((c) => (
                      <WineCard
                        key={c.slug}
                        wine={c.wine}
                        onClick={() => {
                          scroll.current.sheet = 0;
                          void change({
                            sheetView: "candidate",
                            selectedSimilarWineSlug: c.slug,
                          });
                        }}
                      />
                    ))}
                  </div>
                </>
              ) : (
                <div className="stack">
                  <button
                    className="primary"
                    onClick={() => {
                      close();
                      scan.upload();
                    }}
                  >
                    Повторить фото
                  </button>
                  <button
                    className="secondary"
                    onClick={() => navigate("/?search=1")}
                  >
                    Найти по названию
                  </button>
                </div>
              )}
            </>
          )}
        </Sheet>
      )}
    </div>
  );
}
