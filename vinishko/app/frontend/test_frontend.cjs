// Run: node vinishko/app/test_frontend.cjs. No browser or provider required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const crypto = require("node:crypto").webcrypto;

const nodes = new Map();
class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {};
    this.value = ""; this.textContent = ""; this.disabled = false; this.namespaceURI = "http://www.w3.org/2000/svg";
    this.classList = {toggle() {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  removeAttribute(key) { delete this.attributes[key]; }
  scrollIntoView() {}
}
const html = fs.readFileSync(__dirname + "/static/index.html", "utf8");
for (const match of html.matchAll(/<([a-z-]+)[^>]*\bid="([^"]+)"/g)) nodes.set(match[2], new Element(match[1]));
const document = {
  getElementById: id => { assert(nodes.has(id), "Unknown UI element: " + id); return nodes.get(id); },
  createElement: tag => new Element(tag),
  createElementNS: (ns, tag) => new Element(tag),
  querySelectorAll: () => [],
  querySelector: () => null,
};
const storage = new Map(), calls = [];
let reply = null, status = 200;
const context = vm.createContext({
  document, crypto, FormData, Blob, URL, CSS: {escape: value => value},
  localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
  fetch: async (path, options = {}) => { calls.push({path, options}); return {ok: status < 400, status, text: async () => status === 204 ? "" : JSON.stringify(reply)}; },
});
vm.runInContext(fs.readFileSync(__dirname + "/static/app.js", "utf8"), context);
const evaluate = code => vm.runInContext(code, context);
const assistant = {role: "assistant", content: "<script>alert(1)</script>", suggestions: ["Как подавать?"]};

(async () => {
  const ids = new Set();
  for (let i = 0; i < 100; i++) {
    const id = evaluate("uuid7()");
    assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    assert(Math.abs(parseInt(id.replaceAll("-", "").slice(0, 12), 16) - Date.now()) < 2000);
    ids.add(id);
  }
  assert.equal(ids.size, 100);
  assert(nodes.get("send").disabled);

  status = 201; reply = {message: assistant};
  await evaluate('startChat({"Название вина":"Кокур"}, [{"Название вина":"Альтернатива"}])');
  const id = nodes.get("session-id").value;
  assert.equal(calls.at(-1).path, "/v1/sessions/" + id);
  assert.equal(JSON.parse(calls.at(-1).options.body).candidates.length, 1);
  assert.equal(evaluate("activeSession"), id);
  assert.equal(storage.get("vinishko.session"), id);
  assert.equal(nodes.get("messages").children[0].children[1].textContent, assistant.content);
  assert.equal(nodes.get("send").disabled, false);

  await evaluate('sendMessage("  Как подавать?  ")');
  assert.equal(JSON.parse(calls.at(-1).options.body).content, "Как подавать?");
  assert.equal(evaluate("messages.length"), 3);

  status = 200; reply = {session_id: id, messages: [assistant]};
  await nodes.get("restore").onclick();
  assert.equal(calls.at(-1).path, "/v1/sessions/" + id);
  assert.equal(evaluate("messages.length"), 1);
  nodes.get("session-id").value = "bad-id";
  const before = calls.length;
  await nodes.get("restore").onclick();
  assert.equal(calls.length, before);
  assert.match(nodes.get("notice").textContent, /UUIDv7/);

  nodes.get("session-id").value = id;
  status = 204;
  await nodes.get("delete").onclick();
  assert.equal(calls.at(-1).options.method, "DELETE");
  assert.equal(evaluate("activeSession"), null);
  assert.equal(nodes.get("send").disabled, true);
  assert.equal(storage.size, 0);

  status = 503; reply = {detail: "Service offline"};
  await evaluate('startChat({"Название вина":"Кокур"})');
  assert.equal(evaluate("activeSession"), null);
  assert(storage.get("vinishko.session")); // Recover this ID after an ambiguous failure.
  assert.match(nodes.get("notice").textContent, /503: Service offline/);
  assert.equal(evaluate("busy"), false);

  status = 200;
  evaluate('photo = new Blob(["test-photo"], {type:"image/png"})');
  reply = {category: "Красное", brand: "Фанагория"};
  await nodes.get("predict").onclick();
  assert.equal(calls.at(-1).path, "/v1/predict");
  assert(calls.at(-1).options.body.get("image"));
  assert.equal(nodes.get("features").children.length, 1);

  const match = {slug: "kokur", score: .92, source: "filter_v1", catalog: {"Название вина": "Кокур"}, checklist: {}, image_url: "/catalog/images/kokur.jpg"};
  const base = {uuid: id, index: 1, bbox: [0, 0, 20, 40], polygons: [[[0, 0], [20, 0], [20, 40]]], candidates: []};
  reply = {image: {width: 100, height: 200}, ignored: 1, timings_s: {search: .1}, bottles: [
    {...base, status: "matched", match},
    {...base, uuid: "unknown", index: 2, status: "rejected", rejection: {message: "Нет в каталоге", description: "Не найдено"}, unknown_wine: {category: "Красное", brand: "Фанагория"}},
    {...base, uuid: "candidates", index: 3, status: "candidates", candidates: [match]},
  ]};
  await nodes.get("recognize").onclick();
  assert.equal(calls.at(-1).path, "/recognize");
  assert.equal(nodes.get("bottles").children.length, 3);
  assert.equal(nodes.get("masks").children.length, 3);
  assert.equal(nodes.get("masks").attributes.viewBox, "0 0 100 200");
  assert.equal(nodes.get("raw-result").hidden, false);
  nodes.get("show-masks").checked = false;
  nodes.get("show-masks").onchange();
  assert.equal(nodes.get("masks").hidden, true);
  console.log("Frontend flow checks passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
