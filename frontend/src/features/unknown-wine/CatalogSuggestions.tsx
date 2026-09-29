import { useEffect, useRef } from "react";
import {
  SuggestionsSchema,
  type BottleDetection,
  type Suggestions,
} from "../../shared/api/contracts";
import { parse, request } from "../../shared/api/client";
/**
 * Бутылке, которую отверг поиск, похожих из поиска нет: когда whatis определил категорию или
 * винодельню, один раз запрашивает вина каталога с ними и отдаёт их в onLoaded для сессии.
 * Ничего не рисует.
 */
export function CatalogSuggestions({
  detection,
  onLoaded,
}: {
  detection: BottleDetection;
  onLoaded: (value: Suggestions) => void;
}) {
  const loaded = useRef(onLoaded);
  loaded.current = onLoaded;
  const attributes = detection.unknownWine;
  const pending =
    !detection.similar.length &&
    !!attributes &&
    detection.suggestions === undefined;
  useEffect(() => {
    if (!pending || !attributes) return;
    const controller = new AbortController();
    const query = new URLSearchParams({
      seed: detection.id,
      brand: attributes.brand,
      category: attributes.category,
    });
    request(`/suggestions?${query}`, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted)
          loaded.current(parse(SuggestionsSchema, data));
      })
      // Похожие необязательны: при ошибке экран остаётся без них, повтор — при следующем открытии.
      .catch(() => undefined);
    return () => controller.abort();
  }, [pending, detection.id, attributes?.brand, attributes?.category]);
  return null;
}
