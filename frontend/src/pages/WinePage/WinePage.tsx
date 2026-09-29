import { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { Header, SearchBar, Footer } from "../../app/Shell";
import { service, isMockMode } from "../../shared/api/service";
import { isAbort, messageFor } from "../../shared/api/errors";
import type { WineDetails } from "../../shared/api/contracts";
import { WineImage, WineCard, Ratings } from "../../entities/wine/WineCard";
import { Icon } from "../../shared/ui/Icon";
import { Sheet } from "../../shared/ui/Sheet";
import { SommelierChat } from "../../features/sommelier/SommelierChat";
import s from "./WinePage.module.css";
const pairings: Record<string, { image: string; text: string }> = {
  "Мясо и стейки": {
    image: "/assets/steak.webp",
    text: "Попробуйте стейк на гриле с розмарином. Насыщенность блюда поддержит ягодные и пряные оттенки вина.",
  },
  "Запечённые овощи": {
    image: "/assets/vegetables.webp",
    text: "Баклажан, сладкий перец и грибы из духовки. Добавьте оливковое масло и немного пряных трав.",
  },
  Сыры: {
    image: "/assets/cheese.webp",
    text: "Соберите небольшую тарелку полутвёрдых выдержанных сыров. Дайте сыру согреться до комнатной температуры.",
  },
  Рыба: {
    image: "/assets/vegetables.webp",
    text: "Белая рыба с лимоном и зеленью подчеркнёт свежесть белого вина. Выбирайте лёгкий соус.",
  },
  Птица: {
    image: "/assets/bbq.webp",
    text: "Запечённая птица с травами и овощами подойдёт к мягким ягодным винам.",
  },
};
export function WinePage() {
  const { slug = "" } = useParams(),
    navigate = useNavigate(),
    location = useLocation();
  const [wine, setWine] = useState<WineDetails | null>(null),
    [error, setError] = useState<string | null>(null),
    [pairing, setPairing] = useState<string | null>(null);
  const from =
    typeof location.state?.from === "string" &&
    /^\/scan\/[a-zA-Z0-9-]+$/.test(location.state.from)
      ? location.state.from
      : "/";
  useEffect(() => {
    const abort = new AbortController();
    setWine(null);
    setError(null);
    window.scrollTo(0, 0);
    service
      .getWine(slug, { signal: abort.signal })
      .then((w) => {
        if (!abort.signal.aborted) {
          setWine(w);
          document.title = `${w.name} — Своё Вино`;
        }
      })
      .catch((e) => {
        if (!isAbort(e)) setError(messageFor(e));
      });
    return () => abort.abort();
  }, [slug]);
  return (
    <div className={s.page}>
      <Header />
      <main>
        <button className={s.back} onClick={() => navigate(from)}>
          <Icon name="back" size={17} />
          {from === "/" ? "Свои вина" : "К вашему фото"}
        </button>
        {error ? (
          <div className="empty" role="alert">
            <h1>Вино пока недоступно</h1>
            <p className="muted">{error}</p>
            <button className="primary" onClick={() => navigate("/?search=1")}>
              Найти по названию
            </button>
          </div>
        ) : !wine ? (
          <p className="page-loading" role="status">
            Открываем вино…
          </p>
        ) : (
          <>
            <section className={s.hero}>
              <div className={s.title}>
                <h1>{wine.name}</h1>
                <p>{wine.producer}</p>
              </div>
              <div className={s.bottle}>
                <WineImage wine={wine} large />
              </div>
              <div className={s.meta}>
                <span>
                  {[wine.color, wine.category?.toLowerCase()]
                    .filter(Boolean)
                    .join(" ")}
                </span>
                {wine.vintage && <span>{wine.vintage}</span>}
              </div>
            </section>
            <section
              className={s.characteristics}
              aria-label="Характеристики вина"
            >
              <div className={s.feature}>
                <img src="/assets/region.webp" alt="" />
                <div>
                  <span>Регион</span>
                  <strong>{wine.region ?? "Не указан"}</strong>
                </div>
              </div>
              <div className={s.feature}>
                <img src="/assets/grapes.webp" alt="" />
                <div>
                  <span>Сорт винограда</span>
                  <strong>
                    {wine.grapeVarieties.length
                      ? wine.grapeVarieties.join(", ")
                      : "Не указан"}
                  </strong>
                </div>
              </div>
              <div className={s.fact}>
                <span>Температура подачи</span>
                <strong>{wine.servingTemperature ?? "Не указана"}</strong>
              </div>
              <div className={s.fact}>
                <span>Крепость вина</span>
                <strong>
                  {wine.alcoholPercent !== null
                    ? `${wine.alcoholPercent.toLocaleString("ru")}%`
                    : "Не указана"}
                </strong>
              </div>
            </section>
            {wine.description && (
              <section className={s.description}>
                <p className="eyebrow">Характер вина</p>
                <h2>Ближе к вкусу</h2>
                <p>{wine.description}</p>
              </section>
            )}
            {(wine.ratings.community || wine.ratings.roskachestvo) && (
              <section className={s.ratingSection}>
                <h2>Рейтинги</h2>
                <Ratings wine={wine} />
              </section>
            )}
            {!!wine.foodPairings.length && (
              <section className={s.pairings}>
                <p className="eyebrow">Идеальная пара</p>
                <h2>К чему подать</h2>
                <p className="muted">Выберите блюдо — подскажем сочетание</p>
                <div>
                  {wine.foodPairings.map((p) => (
                    <button key={p} onClick={() => setPairing(p)}>
                      {pairings[p] && <img src={pairings[p].image} alt="" />}
                      <span>{p}</span>
                      <Icon name="chevron" size={16} />
                    </button>
                  ))}
                </div>
              </section>
            )}
            {!!wine.similarWines.length && (
              <section className={s.similar}>
                <h2>Похожие вина</h2>
                <div className="grid">
                  {wine.similarWines.map((w) => (
                    <WineCard
                      key={w.slug}
                      wine={w}
                      onClick={() =>
                        navigate(`/wine/${w.slug}`, { state: { from } })
                      }
                    />
                  ))}
                </div>
              </section>
            )}
            {isMockMode && (
              <p className={s.disclaimer}>
                Демонстрационная карточка. Фотографии и доступные рейтинги взяты
                из предоставленных макетов; часть описаний и сочетаний — пример
                наполнения. Неизвестные значения не подставляются.
              </p>
            )}
            {!isMockMode && wine.catalog && (
              <SommelierChat
                key={wine.slug}
                contextId={wine.slug}
                wine={wine.catalog}
                candidates={wine.candidateCatalogs}
              />
            )}
          </>
        )}
      </main>
      <Footer />
      <SearchBar />
      {pairing && (
        <Sheet title={`Сочетание: ${pairing}`} onClose={() => setPairing(null)}>
          <p className="eyebrow">К вашему вину</p>
          <h2>{pairing}</h2>
          {pairings[pairing] && (
            <img
              className={s.pairingImage}
              src={pairings[pairing].image}
              alt=""
            />
          )}
          <p className="muted">
            {pairings[pairing]?.text ??
              `Попробуйте ${pairing.toLowerCase()} с этим вином.`}
          </p>
          {wine?.servingTemperature && (
            <p className="notice">
              Подавайте вино при {wine.servingTemperature}.
            </p>
          )}
        </Sheet>
      )}
    </div>
  );
}
