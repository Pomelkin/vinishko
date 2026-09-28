import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Icon } from "../shared/ui/Icon";
import { Sheet } from "../shared/ui/Sheet";
import { useScan } from "./ScanProvider";
import { isMockMode } from "../shared/api/service";
import s from "./Shell.module.css";
export function Header() {
  const [menu, setMenu] = useState(false);
  const navigate = useNavigate();
  return (
    <>
      <header className={s.header}>
        <Link to="/" aria-label="Своё Вино — главная">
          <img src="/assets/logo.svg" alt="Своё Вино. От Россельхозбанка" />
        </Link>
        <div>
          <button
            className="icon-button"
            aria-label="Поиск вина"
            onClick={() => navigate("/?search=1")}
          >
            <Icon name="search" />
          </button>
          <button
            className={s.menuButton}
            aria-label="Открыть меню"
            onClick={() => setMenu(true)}
          >
            <Icon name="menu" />
          </button>
        </div>
      </header>
      {menu && (
        <Sheet title="Меню" onClose={() => setMenu(false)}>
          <h2>Найти своё вино</h2>
          <div className="stack">
            <button
              className="secondary"
              onClick={() => {
                setMenu(false);
                navigate("/?search=1");
              }}
            >
              Каталог и поиск
            </button>
            {isMockMode && (
              <button
                className="secondary"
                onClick={() => {
                  setMenu(false);
                  navigate("/?demo=1");
                }}
              >
                Демонстрационные сценарии
              </button>
            )}
          </div>
          <p className={`${s.menuText} muted`}>
            Сфотографируйте бутылку или выберите снимок. Нажмите на выделенный
            силуэт, чтобы узнать больше о вине.
          </p>
          {isMockMode && (
            <p className="notice">
              Это демонстрация интерфейса. Фотографии остаются в вашем браузере.
              Распознавание доступно на подготовленных примерах.
            </p>
          )}
        </Sheet>
      )}
    </>
  );
}
export function SearchBar() {
  const navigate = useNavigate(),
    scan = useScan();
  return (
    <nav className={s.searchBar} aria-label="Быстрый поиск">
      <button
        aria-label="Открыть поиск"
        className={s.searchIcon}
        onClick={() => navigate("/?search=1")}
      >
        <Icon name="search" size={26} />
      </button>
      <button className={s.searchText} onClick={() => navigate("/?search=1")}>
        Найти своё вино
      </button>
      <button
        className="icon-button"
        aria-label="Открыть сканер"
        onClick={scan.camera}
      >
        <Icon name="camera" />
      </button>
    </nav>
  );
}
export function Footer() {
  return (
    <footer className={s.footer}>
      <img src="/assets/logo.svg" alt="Своё Вино" />
      <p>Знакомьтесь с российским вином</p>
      {isMockMode && <p>Демонстрационный каталог · 18+</p>}
      <small>Чрезмерное употребление алкоголя вредит вашему здоровью</small>
    </footer>
  );
}
