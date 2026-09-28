import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Header, SearchBar, Footer } from "../../app/Shell";
import { useScan } from "../../app/ScanProvider";
import {
  demoScenarios,
  isMockMode,
  service,
  type DemoScenario,
} from "../../shared/api/service";
import { messageFor, isAbort } from "../../shared/api/errors";
import type { WineSummary } from "../../shared/api/contracts";
import { WineCard } from "../../entities/wine/WineCard";
import { Icon } from "../../shared/ui/Icon";
import { Sheet } from "../../shared/ui/Sheet";
import s from "./ScannerPage.module.css";
export function ScannerPage() {
  const scan = useScan(),
    navigate = useNavigate(),
    [params, setParams] = useSearchParams();
  const [query, setQuery] = useState(""),
    [wines, setWines] = useState<WineSummary[]>([]),
    [error, setError] = useState<string | null>(null),
    [loading, setLoading] = useState(true);
  const [filterOpen, setFilterOpen] = useState(false),
    [color, setColor] = useState("Все"),
    [region, setRegion] = useState("Все"),
    [scenario, setScenario] = useState<DemoScenario>("mixed");
  const [searchOpen, setSearchOpen] = useState(params.has("search")),
    [demoOpen, setDemoOpen] = useState(params.has("demo"));
  const input = useRef<HTMLInputElement>(null),
    catalog = useRef<HTMLElement>(null);
  useEffect(() => {
    if (params.has("search")) {
      setSearchOpen(true);
      requestAnimationFrame(() => {
        catalog.current?.scrollIntoView({ block: "start" });
        input.current?.focus();
      });
    }
    if (params.has("demo")) setDemoOpen(true);
  }, [params]);
  useEffect(() => {
    const abort = new AbortController();
    setLoading(true);
    setError(null);
    service
      .searchWines(query, { signal: abort.signal })
      .then((data) => {
        if (!abort.signal.aborted) {
          setWines(data);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!isAbort(e)) {
          setError(messageFor(e));
          setLoading(false);
        }
      });
    return () => abort.abort();
  }, [query]);
  const visible = wines.filter(
    (w) =>
      (color === "Все" || w.color === color) &&
      (region === "Все" || w.region === region),
  );
  const active = color !== "Все" || region !== "Все";
  const closeDemo = () => {
    setDemoOpen(false);
    if (params.has("demo")) {
      params.delete("demo");
      setParams(params, { replace: true });
    }
  };
  return (
    <div className={s.page}>
      <Header />
      <main>
        <section className={s.hero}>
          <h1>Свои вина</h1>
          <p>
            Сфотографируйте этикетку Российского вина
            <br className={s.desktopBreak} /> или загрузите фото, чтобы найти
            его
          </p>
          <div className={s.scanner}>
            <img
              src="/assets/scanner.svg"
              alt="Иллюстрация сканирования бутылки"
            />
            <button className="primary" onClick={scan.camera}>
              Сканировать
            </button>
            <button className="text-button" onClick={scan.upload}>
              Загрузить фото
            </button>
          </div>
        </section>
        <section ref={catalog} className={s.catalog} aria-label="Каталог вин">
          <div className={s.catalogTools}>
            <button className={s.filter} onClick={() => setFilterOpen(true)}>
              <Icon name="filter" size={18} />
              Фильтр{active && <span className={s.dot} />}
            </button>
            {isMockMode && (
              <button className={s.demoLink} onClick={() => setDemoOpen(true)}>
                Попробовать на примере <Icon name="arrow" size={16} />
              </button>
            )}
          </div>
          {searchOpen && (
            <div className={s.searchField}>
              <label className="sr-only" htmlFor="wine-search">
                Название, сорт или винодельня
              </label>
              <input
                id="wine-search"
                ref={input}
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Название, сорт или винодельня"
              />
              <button
                className="icon-button"
                aria-label="Закрыть поиск"
                onClick={() => {
                  setSearchOpen(false);
                  setQuery("");
                  setParams({}, { replace: true });
                }}
              >
                <Icon name="close" />
              </button>
            </div>
          )}
          {active && (
            <div className={s.applied}>
              <span>
                {[color, region].filter((x) => x !== "Все").join(" · ")}
              </span>
              <button
                className="text-button"
                onClick={() => {
                  setColor("Все");
                  setRegion("Все");
                }}
              >
                Сбросить
              </button>
            </div>
          )}
          {error ? (
            <div role="alert" className="notice">
              {error}
            </div>
          ) : loading ? (
            <p role="status" className="muted">
              Загружаем каталог…
            </p>
          ) : visible.length ? (
            <>
              <div className="grid">
                {visible.map((w) => (
                  <WineCard
                    key={w.slug}
                    wine={w}
                    onClick={() =>
                      navigate(`/wine/${w.slug}`, { state: { from: "/" } })
                    }
                  />
                ))}
              </div>
              {query && (
                <p className={s.count} role="status">
                  Найдено вин: {visible.length}
                </p>
              )}
            </>
          ) : (
            <div className="empty">
              <h2>Пока ничего не нашлось</h2>
              <p className="muted">
                Попробуйте другое название или измените фильтры.
              </p>
              <button
                className="secondary"
                onClick={() => {
                  setQuery("");
                  setColor("Все");
                  setRegion("Все");
                }}
              >
                Сбросить поиск
              </button>
            </div>
          )}
        </section>
        {isMockMode && (
          <p className={s.demoNote}>
            Демонстрационный каталог. Распознавание показано на подготовленных
            примерах; загруженные фото остаются в вашем браузере.
          </p>
        )}
      </main>
      <Footer />
      <SearchBar />
      {filterOpen && (
        <Sheet
          title="Фильтры каталога"
          onClose={() => setFilterOpen(false)}
          footer={
            <button className="primary" onClick={() => setFilterOpen(false)}>
              Показать вина
            </button>
          }
        >
          <h2>Выберите своё</h2>
          <div className={s.filters}>
            <label>
              Цвет
              <select value={color} onChange={(e) => setColor(e.target.value)}>
                {["Все", "Красное", "Белое", "Розовое", "Оранжевое"].map(
                  (v) => (
                    <option key={v}>{v}</option>
                  ),
                )}
              </select>
            </label>
            <label>
              Регион
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
              >
                {["Все", "Крым", "Кубань"].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </label>
          </div>
          <button
            className="text-button"
            onClick={() => {
              setColor("Все");
              setRegion("Все");
            }}
          >
            Сбросить фильтры
          </button>
        </Sheet>
      )}
      {demoOpen && isMockMode && (
        <Sheet
          title="Демонстрационные сценарии"
          onClose={closeDemo}
          footer={
            <button
              className="primary"
              onClick={() => {
                closeDemo();
                scan.demo(scenario);
              }}
            >
              Открыть демофото <Icon name="arrow" size={18} />
            </button>
          }
        >
          <p className="eyebrow">Знакомство со сканером</p>
          <h2>Попробуйте на примере</h2>
          <img
            className={s.demoPhoto}
            src="/assets/demo-multiple.svg"
            alt="Демонстрационная композиция из двух фотографий бутылок"
          />
          <p className="muted">
            Выберите бутылку на фото, откройте карточку и вернитесь к снимку.
          </p>
          <label className={s.scenario}>
            Сценарий
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value as DemoScenario)}
            >
              {demoScenarios.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <p className={s.demoDisclaimer}>
            Это подготовленная композиция из каталожных фотографий с
            размеченными контурами. Результаты заданы заранее, реальное
            распознавание не выполняется.
          </p>
        </Sheet>
      )}
    </div>
  );
}
