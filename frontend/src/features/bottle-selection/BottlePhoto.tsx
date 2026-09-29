import { useId, useRef } from "react";
import type { BottleDetection, Point } from "../../shared/api/contracts";
import { center, hitTest } from "../../shared/geometry/polygons";
import s from "./BottlePhoto.module.css";
export function BottlePhoto({
  url,
  width,
  height,
  detections,
  selected,
  onSelect,
}: {
  url: string;
  width: number;
  height: number;
  detections: BottleDetection[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const prefix = useId().replace(/:/g, ""),
    svg = useRef<SVGSVGElement>(null);
  const points = (polygon: Point[]) =>
    polygon.map(([x, y]) => `${x * width},${y * height}`).join(" ");
  return (
    <div
      className={`${s.photo} ${selected ? s.hasSelection : ""}`}
      data-testid="bottle-photo"
    >
      <svg
        ref={svg}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        aria-label={`Фотография. Бутылок: ${detections.length}`}
        onClick={(e) => {
          const matrix = svg.current?.getScreenCTM();
          if (!matrix) return;
          const local = new DOMPoint(e.clientX, e.clientY).matrixTransform(
            matrix.inverse(),
          );
          const detection = hitTest(detections, [
            local.x / width,
            local.y / height,
          ]);
          if (detection) onSelect(detection.id);
        }}
      >
        <defs>
          {detections.map((d) => (
            <clipPath key={d.id} id={`${prefix}-${d.id}`}>
              {(d.polygons || [d.polygon]).map((p, i) => (
                <polygon key={i} points={points(p)} />
              ))}
            </clipPath>
          ))}
        </defs>
        <image href={url} width={width} height={height} />
        <rect
          width={width}
          height={height}
          fill="#16110e"
          opacity={selected ? 0.48 : 0.12}
          className={s.dimmer}
        />
        {detections.map((d, index) => {
          const contours = d.polygons || [d.polygon];
          const [cx, cy] = center(contours.flat());
          const active = d.id === selected;
          return (
            <g
              key={d.id}
              className={active ? s.selected : s.bottle}
              style={{
                transformOrigin: `${cx * width}px ${cy * height}px`,
                transform: active ? "scale(1.045)" : "scale(1)",
              }}
            >
              <image
                href={url}
                width={width}
                height={height}
                clipPath={`url(#${prefix}-${d.id})`}
                opacity={active || !selected ? 1 : 0.62}
              />
              {contours.map((p, i) => (
                <polygon
                  key={i}
                  points={points(p)}
                  fill="transparent"
                  stroke={active ? "#fffdf3" : "#fffdf3bb"}
                  strokeWidth={active ? 2.5 : 1.5}
                  vectorEffect="non-scaling-stroke"
                  className={s.outline}
                />
              ))}
              <g
                transform={`translate(${cx * width},${Math.max(...d.polygon.map((p) => p[1])) * height + 35})`}
                aria-hidden="true"
              >
                <circle r="22" fill={active ? "#8f3d42" : "#fefdfa"} />
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={active ? "white" : "#2c2a28"}
                  fontSize="23"
                  fontFamily="system-ui"
                >
                  {index + 1}
                </text>
              </g>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
