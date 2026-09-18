import { useState } from "react";
import type { WineSummary } from "../../shared/api/contracts";
import s from "./Wine.module.css";
export function WineImage({
  wine,
  large = false,
  crop = false,
}: {
  wine: WineSummary;
  large?: boolean;
  crop?: boolean;
}) {
  const [broken, setBroken] = useState(false),
    [square, setSquare] = useState(false);
  return wine.imageUrl && !broken ? (
    <img
      className={large ? s.largeImage : s.image}
      style={crop && square ? { objectFit: "cover" } : undefined}
      src={wine.imageUrl}
      alt={wine.name}
      loading={large ? "eager" : "lazy"}
      onLoad={(e) =>
        setSquare(
          e.currentTarget.naturalWidth >= e.currentTarget.naturalHeight * 0.8,
        )
      }
      onError={() => setBroken(true)}
    />
  ) : (
    <div className={s.placeholder}>
      <img src="/assets/loader.svg" alt="" />
      <span>Фото появится позже</span>
    </div>
  );
}
export function WineCard({
  wine,
  onClick,
}: {
  wine: WineSummary;
  onClick: () => void;
}) {
  return (
    <button className={s.card} onClick={onClick}>
      {wine.ratings.community && (
        <span
          className={s.rating}
          aria-label={`Народный рейтинг ${wine.ratings.community.value} из ${wine.ratings.community.scaleMax}`}
        >
          <img src="/assets/rating.svg" alt="" />
          {wine.ratings.community.value.toFixed(2)}
        </span>
      )}
      <div className={s.imageArea}>
        <WineImage wine={wine} />
      </div>
      <span className={s.name}>{wine.name}</span>
      <span className={s.producer}>{wine.producer}</span>
    </button>
  );
}
export function Ratings({ wine }: { wine: WineSummary }) {
  return (
    <div className={s.ratings}>
      {wine.ratings.community && (
        <p>
          Народный рейтинг{" "}
          <strong>
            {wine.ratings.community.value.toFixed(2)} /{" "}
            {wine.ratings.community.scaleMax}
          </strong>
        </p>
      )}
      {wine.ratings.roskachestvo && (
        <p>
          Роскачество{" "}
          <strong>
            {wine.ratings.roskachestvo.value} /{" "}
            {wine.ratings.roskachestvo.scaleMax}
          </strong>
        </p>
      )}
    </div>
  );
}
export function WinePreview({
  wine,
  label,
}: {
  wine: WineSummary;
  label: string;
}) {
  return (
    <>
      <p className="eyebrow">{label}</p>
      <div className={s.preview}>
        <div className={s.previewPhoto}>
          <WineImage wine={wine} crop />
        </div>
        <div>
          <h2>{wine.name}</h2>
          <p className="accent">{wine.producer}</p>
          <p className="muted">
            {[wine.region, wine.vintage].filter(Boolean).join(" · ")}
          </p>
          <p>
            {[wine.color, wine.category?.toLocaleLowerCase("ru")]
              .filter(Boolean)
              .join(" ")}
          </p>
        </div>
      </div>
      {wine.grapeVarieties.length > 0 && (
        <p className={s.grapes}>{wine.grapeVarieties.join(", ")}</p>
      )}
      {wine.shortDescription && (
        <p className={s.description}>{wine.shortDescription}</p>
      )}
      <Ratings wine={wine} />
    </>
  );
}
