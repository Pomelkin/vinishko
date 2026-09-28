"use strict";

const MAX_WHATIS_BYTES = 10 * 1024 * 1024;
const SESSION_STORAGE_KEY = "vinishko.session";
const $ = id => document.getElementById(id);
let photo = null, previewUrl = null, result = null, busy = false, activeSession = null;
let messages = [], selectedBottle = null;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function uuid7() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  let timestamp = BigInt(Date.now());
  for (let i = 5; i >= 0; i--) { bytes[i] = Number(timestamp & 255n); timestamp >>= 8n; }
  bytes[6] = (bytes[6] & 15) | 112;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join("-");
}

function rememberSession(id) {
  $("session-id").value = id || "";
  try { id ? localStorage.setItem(SESSION_STORAGE_KEY, id) : localStorage.removeItem(SESSION_STORAGE_KEY); }
  catch { /* Browser storage is optional; the session ID remains available to copy. */ }
}

function controls() {
  document.querySelectorAll("button, input, textarea").forEach(node => { node.disabled = busy; });
  $("recognize").disabled = busy || !photo;
  $("predict").disabled = busy || !photo;
  $("send").disabled = busy || !activeSession;
  $("message").disabled = busy || !activeSession;
  $("delete").disabled = busy || !$("session-id").value.trim();
  $("restore").disabled = busy || !$("session-id").value.trim();
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { throw new Error("API вернул невалидный ответ."); }
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : JSON.stringify(data?.detail || data);
    throw new Error(response.status + ": " + detail);
  }
  return data;
}

function jsonRequest(path, body) {
  return request(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
}

async function run(label, action) {
  if (busy) return;
  busy = true;
  controls();
  $("notice").hidden = false;
  $("notice").className = "loading";
  $("notice").textContent = label;
  try {
    await action();
    $("notice").className = "";
    $("notice").textContent = "Готово";
  } catch (error) {
    $("notice").className = "error";
    $("notice").textContent = error.message || "Не удалось выполнить запрос.";
  } finally { busy = false; controls(); }
}

function uploadBody() {
  if (!photo) throw new Error("Сначала выберите фотографию.");
  const data = new FormData();
  data.append("image", photo);
  return data;
}

function fields(card) {
  const list = element("dl");
  for (const [name, value] of Object.entries(card || {})) {
    if (value === null || value === "") continue;
    list.append(element("dt", name), element("dd", String(value)));
  }
  return list;
}

function featureCard(value) {
  const node = element("div", undefined, "features");
  node.append(element("strong", "Признаки вина"), fields({"Категория": value.category, "Винодельня": value.brand}));
  return node;
}

function catalogImage(candidate) {
  const image = element("img", undefined, "catalog-image");
  image.alt = candidate.catalog?.["Название вина"] || "Бутылка из каталога";
  image.loading = "lazy";
  if (candidate.image_url?.startsWith("/catalog/images/")) image.src = candidate.image_url;
  image.onerror = () => { image.hidden = true; };
  return image;
}

function button(text, action, className = "secondary") {
  const node = element("button", text, className);
  node.type = "button";
  node.onclick = action;
  return node;
}

function selectBottle(id) {
  selectedBottle = id;
  document.querySelectorAll("[data-bottle]").forEach(node => node.classList.toggle("selected", node.dataset.bottle === id));
  document.querySelector('[data-bottle="' + CSS.escape(id) + '"].bottle')?.scrollIntoView({behavior: "smooth", block: "nearest"});
}

function drawMasks() {
  const svg = $("masks");
  svg.replaceChildren();
  if (!result) return;
  svg.setAttribute("viewBox", "0 0 " + result.image.width + " " + result.image.height);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  for (const bottle of result.bottles) {
    const group = document.createElementNS(svg.namespaceURI, "g");
    group.dataset.bottle = bottle.uuid;
    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "button");
    group.setAttribute("aria-label", "Выбрать бутылку " + bottle.index);
    group.onclick = () => selectBottle(bottle.uuid);
    group.onkeydown = event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectBottle(bottle.uuid); } };
    for (const points of bottle.polygons) {
      const polygon = document.createElementNS(svg.namespaceURI, "polygon");
      polygon.setAttribute("points", points.map(point => point.join(",")).join(" "));
      group.append(polygon);
    }
    const number = document.createElementNS(svg.namespaceURI, "text");
    number.textContent = bottle.index;
    number.setAttribute("x", bottle.bbox[0]);
    number.setAttribute("y", Math.max(30, bottle.bbox[1] + 30));
    group.append(number);
    svg.append(group);
  }
}

function renderBottles() {
  const container = $("bottles");
  container.replaceChildren();
  $("summary").textContent = "Бутылок: " + result.bottles.length + " · Отброшено: " + result.ignored + " · " +
    Object.entries(result.timings_s).map(([step, time]) => step + ": " + time + " с").join(" · ");
  if (!result.bottles.length) container.append(element("p", "Читаемых бутылок не найдено. Попробуйте более близкое и чёткое фото.", "empty"));
  for (const bottle of result.bottles) {
    const node = element("article", undefined, "bottle");
    node.dataset.bottle = bottle.uuid;
    const heading = element("div", undefined, "bottle-heading");
    const labels = {matched: "В каталоге", rejected: "Не определено", candidates: "Есть кандидаты", normalized: "Только нормализация"};
    heading.append(button("Бутылка " + bottle.index, () => selectBottle(bottle.uuid), "quiet"),
      element("span", labels[bottle.status] || bottle.status, "badge" + (bottle.status === "matched" ? "" : " unknown")));
    node.append(heading);
    if (bottle.match) {
      const info = element("div", undefined, "wine-info");
      const content = element("div");
      content.append(element("h3", bottle.match.catalog["Название вина"] || bottle.match.slug),
        element("p", "Сходство: " + (bottle.match.score * 100).toFixed(1) + "% · " + bottle.match.source, "muted"),
        button("Обсудить с сомелье", () => startChat(bottle.match.catalog, bottle.candidates.map(c => c.catalog))));
      info.append(catalogImage(bottle.match), content);
      node.append(info);
      const details = element("details");
      details.append(element("summary", "Карточка и наблюдения модели"), fields(bottle.match.catalog));
      if (Object.keys(bottle.match.checklist || {}).length) details.append(element("pre", JSON.stringify(bottle.match.checklist, null, 2)));
      node.append(details);
    }
    if (bottle.rejection) node.append(element("p", bottle.rejection.message, "muted"), element("p", bottle.rejection.description, "muted"));
    if (bottle.unknown_wine) node.append(featureCard(bottle.unknown_wine));
    if (bottle.unknown_wine_error) node.append(element("p", "whatis: " + bottle.unknown_wine_error.status_code + " · " + bottle.unknown_wine_error.detail, "error"));
    for (const candidate of bottle.candidates) {
      const item = element("div", undefined, "candidate"), info = element("div");
      info.append(element("strong", candidate.catalog["Название вина"] || candidate.slug),
        element("div", (candidate.score * 100).toFixed(1) + "% сходства", "muted"),
        button("Обсудить этот вариант", () => startChat(candidate.catalog, bottle.candidates.filter(c => c.slug !== candidate.slug).slice(0, 5).map(c => c.catalog))));
      const details = element("details");
      details.append(element("summary", "Карточка кандидата"), fields(candidate.catalog));
      info.append(details);
      item.append(catalogImage(candidate), info);
      node.append(item);
    }
    container.append(node);
  }
  drawMasks();
  if (result.bottles.length) selectBottle(result.bottles[0].uuid);
}

function renderMessages() {
  $("messages").replaceChildren();
  if (!messages.length) $("messages").append(element("div", "История пока пуста.", "empty"));
  for (const message of messages) {
    const node = element("div", undefined, "message " + message.role);
    node.append(element("div", message.role === "user" ? "Вы" : "Сомелье", "author"), element("div", message.content, "content"));
    if (message.suggestions?.length) {
      const suggestions = element("div", undefined, "suggestions");
      for (const text of message.suggestions) suggestions.append(button(text, () => sendMessage(text)));
      node.append(suggestions);
    }
    $("messages").append(node);
  }
  $("messages").scrollTop = $("messages").scrollHeight;
}

function validSessionId() {
  const id = $("session-id").value.trim();
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id)) throw new Error("Введите корректный UUIDv7 диалога.");
  return id;
}

function startChat(wine, candidates = []) {
  return run("Сомелье знакомится с вином…", async () => {
    $("wine-card").value = JSON.stringify(wine, null, 2);
    $("alternatives").value = JSON.stringify(candidates.slice(0, 5), null, 2);
    const id = uuid7();
    // Keep the ID before POST: after a timeout the server may already have saved this session.
    rememberSession(id);
    activeSession = null;
    messages = [];
    renderMessages();
    $("chat-wine").textContent = wine["Название вина"] || "Выбранное вино";
    const answer = await jsonRequest("/v1/sessions/" + id, {wine, candidates: candidates.slice(0, 5)});
    activeSession = id;
    messages = [answer.message];
    renderMessages();
    $("chat-title").scrollIntoView({behavior: "smooth", block: "nearest"});
  });
}

function sendMessage(content) {
  return run("Сомелье готовит ответ…", async () => {
    if (!activeSession) throw new Error("Откройте или восстановите диалог.");
    const text = content.trim();
    if (!text || text.length > 4000) throw new Error("Вопрос должен содержать от 1 до 4000 символов.");
    const answer = await jsonRequest("/v1/sessions/" + activeSession + "/messages", {content: text});
    messages.push({role: "user", content: text, suggestions: null}, answer.message);
    $("message").value = "";
    renderMessages();
  });
}

$("photo").onchange = () => {
  photo = $("photo").files[0] || null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = photo ? URL.createObjectURL(photo) : null;
  $("photo-preview").removeAttribute("src");
  if (previewUrl) $("photo-preview").src = previewUrl;
  $("preview").hidden = !photo;
  $("filename").textContent = photo ? photo.name + " · " + (photo.size / 1024 / 1024).toFixed(2) + " МиБ" : "Фото ещё не выбрано";
  result = null;
  $("masks").replaceChildren();
  $("features").replaceChildren();
  $("bottles").replaceChildren(element("div", "Фото выбрано. Можно начать распознавание.", "empty"));
  $("summary").textContent = "";
  $("raw-result").hidden = true;
  controls();
};
$("photo-preview").onerror = () => { $("filename").textContent += " · Браузер не показывает этот формат; файл всё ещё можно отправить в API."; };
$("show-masks").onchange = () => { $("masks").hidden = !$("show-masks").checked; };
$("recognize").onclick = () => run("Ищем бутылки и проверяем каталог…", async () => {
  result = await request("/recognize", {method: "POST", body: uploadBody()});
  renderBottles();
  $("raw-json").textContent = JSON.stringify(result, null, 2);
  $("raw-result").hidden = false;
});
$("predict").onclick = () => run("Определяем категорию и винодельню…", async () => {
  if (photo.size > MAX_WHATIS_BYTES) throw new Error("Для whatis выберите фото до 10 МиБ.");
  const features = await request("/v1/predict", {method: "POST", body: uploadBody()});
  $("features").replaceChildren(featureCard(features));
});
$("health").onclick = () => run("Проверяем основной API…", async () => {
  const health = await request("/health");
  $("health-json").textContent = JSON.stringify(health, null, 2);
  $("health-result").hidden = false;
  $("health-result").open = true;
});
$("manual-chat").onclick = () => {
  try {
    const wine = JSON.parse($("wine-card").value), candidates = JSON.parse($("alternatives").value);
    if (!wine || typeof wine !== "object" || Array.isArray(wine) || !Object.keys(wine).length) throw new Error("Нужна непустая JSON-карточка вина.");
    if (!Array.isArray(candidates) || candidates.length > 5) throw new Error("Альтернативы должны быть массивом из 0–5 карточек.");
    startChat(wine, candidates);
  } catch (error) { run("Проверяем карточку…", () => { throw error; }); }
};
$("message-form").onsubmit = event => { event.preventDefault(); sendMessage($("message").value); };
$("message").onkeydown = event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendMessage($("message").value); } };
$("session-id").oninput = controls;
$("restore").onclick = () => run("Загружаем историю диалога…", async () => {
  const id = validSessionId();
  const session = await request("/v1/sessions/" + id);
  activeSession = id;
  rememberSession(id);
  messages = session.messages;
  $("chat-wine").textContent = "Восстановленный диалог";
  renderMessages();
});
$("delete").onclick = () => run("Удаляем диалог…", async () => {
  const id = validSessionId();
  await request("/v1/sessions/" + id, {method: "DELETE"});
  if (activeSession === id) {
    activeSession = null;
    messages = [];
    $("chat-wine").textContent = "Диалог удалён";
    renderMessages();
  }
  rememberSession(activeSession);
});
try { $("session-id").value = localStorage.getItem(SESSION_STORAGE_KEY) || ""; } catch { /* Optional storage. */ }
controls();
