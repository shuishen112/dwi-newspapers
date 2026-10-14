// Browser port of www_experiments/search/engine.py over the static export of
// www_experiments/scripts/export_site.py. Keep the two in step: tokenization,
// fold(), the adaptive threshold, in-order proximity matching and scoring.
"use strict";

const DATA = "data/";
const PAGE_SIZE = 20;
const REVIEW_KEY = "dwi-newspaper-reviews-v1";
const OCR_KEY = "dwi-newspaper-ocr-v1";
const HANDLE = "https://hdl.handle.net/109.3.1/uuid:";

const state = {
  meta: null,
  vocab: new Map(),     // length -> {terms, arrays, df: Map}
  postShards: new Map(), // shard -> Map(term -> raw postings)
  textShards: new Map(), // shard -> Map(pid -> [page, text, geo])
  hits: [],
  shown: 0,
  expansions: [],
  excluded: new Set(),
  lastQuery: null,
};

// ------------------------------------------------------------------ text
const TOKEN_RE = /[\p{L}\p{N}]+/gu;
const fold = (s) => s.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
function tokenize(text) {
  const out = [];
  for (const m of text.matchAll(TOKEN_RE)) out.push([m.index, m.index + m[0].length, fold(m[0])]);
  return out;
}
function parseQuery(q) {
  const out = [];
  for (const chunk of q.split(/\s+/).filter(Boolean)) {
    const exact = chunk.startsWith("=");
    for (const t of tokenize(chunk.replace(/^=+/, ""))) out.push([t[2], exact]);
  }
  return out;
}
function fnv1a(s) {
  let h = 0x811c9dc5;
  for (const ch of s) {
    h ^= ch.codePointAt(0);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

// ------------------------------------------------------------------ thresholds
function binomCdf(k, n, p) {
  if (k < 0) return 0;
  if (k >= n) return 1;
  let term = Math.pow(1 - p, n), s = term;
  for (let i = 1; i <= k; i++) {
    term *= ((n - i + 1) / i) * (p / (1 - p));
    s += term;
  }
  return s;
}
function adaptiveK(ell, eps, eta) {
  for (let k = 0; k <= ell + 8; k++) if (binomCdf(k, ell, eps) >= 1 - eta) return k;
  return ell + 8;
}
function levenshtein(a, b, max) {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const v = Math.min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      cur.push(v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev = cur;
  }
  return prev[b.length];
}

// ------------------------------------------------------------------ data
async function fetchText(path) {
  const r = await fetch(DATA + path);
  if (!r.ok) throw new Error(`could not load ${path} (${r.status})`);
  return r.text();
}
async function loadVocab(n) {
  if (n < 1 || n > state.meta.max_len) return { terms: [], arrays: [], df: new Map() };
  if (!state.vocab.has(n)) {
    const p = fetchText(`vocab/${n}.txt`).then((txt) => {
      const terms = [], df = new Map();
      for (const line of txt.split("\n")) {
        if (!line) continue;
        const [t, d] = line.split("\t");
        terms.push(t);
        df.set(t, +d);
      }
      return { terms, arrays: null, df };
    }).catch(() => ({ terms: [], arrays: [], df: new Map() }));
    state.vocab.set(n, p);
  }
  return state.vocab.get(n);
}
async function dfOf(form) {
  const v = await loadVocab([...form].length);
  return v.df.get(form) || 0;
}
async function postingsOf(form) {
  const h = fnv1a(form) % state.meta.post_shards;
  if (!state.postShards.has(h)) {
    state.postShards.set(h, fetchText(`post/${h}.txt`).then((txt) => {
      const m = new Map();
      for (const line of txt.split("\n")) {
        const tab = line.indexOf("\t");
        if (tab > 0) m.set(line.slice(0, tab), line.slice(tab + 1));
      }
      return m;
    }));
  }
  const raw = (await state.postShards.get(h)).get(form);
  if (!raw) return [];
  const out = [];
  let pid = 0;
  for (const part of raw.split(";")) {
    const [d, ps] = part.split(":");
    pid += parseInt(d, 36);
    out.push([pid, ps.split(",").map((x) => parseInt(x, 36))]);
  }
  return out;
}
async function passage(pid) {
  const n = Math.floor(pid / state.meta.text_shard);
  if (!state.textShards.has(n)) {
    state.textShards.set(n, fetch(`${DATA}text/${n}.json`).then((r) => r.json()).then((items) => {
      const m = new Map();
      for (const [p, page, text, geo] of items) m.set(p, { page, text, geo });
      return m;
    }));
  }
  return (await state.textShards.get(n)).get(pid);
}
function issueOf(pid) {
  const is = state.meta.issues;
  let lo = 0, hi = is.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (is[mid][0] <= pid) lo = mid; else hi = mid - 1;
  }
  const [first, , date, paper, stratum, pages, pageStarts] = is[lo];
  let page = 1;
  for (const [offset, pg] of pageStarts) if (offset <= pid - first) page = pg;
  return { date, year: +date.slice(0, 4), paper: state.meta.papers[paper], paperIdx: paper, stratum, pages, page };
}

// ------------------------------------------------------------------ search
async function expand(word, forceExact, opts) {
  const ell = [...word].length;
  const exact = forceExact || opts.exact || ell < opts.minFuzzy || opts.exactTerms.has(word);
  const k = exact ? 0 : adaptiveK(ell, opts.epsilon, opts.eta);
  const found = new Map();
  if (k === 0) {
    found.set(word, 0);
  } else {
    const w = [...word];
    for (let n = Math.max(ell - k, 1); n <= ell + k; n++) {
      const v = await loadVocab(n);
      if (!v.arrays) v.arrays = v.terms.map((t) => [...t]);
      for (let i = 0; i < v.terms.length; i++) {
        const d = levenshtein(w, v.arrays[i], k);
        if (d <= k) found.set(v.terms[i], d);
      }
    }
  }
  const variants = [];
  for (const [form, d] of found) {
    const df = await dfOf(form);
    if (df > 0) variants.push({ form, d, df });
  }
  variants.sort((a, b) => a.d - b.d || b.df - a.df || (a.form < b.form ? -1 : 1));
  return { word, k, exact, variants };
}

function passesFilters(info, f) {
  if (f.stratum && info.stratum !== f.stratum) return false;
  if (f.from && info.year < f.from) return false;
  if (f.to && info.year > f.to) return false;
  if (f.paper !== "" && f.paper != null && info.paperIdx !== +f.paper) return false;
  return true;
}

async function searchOne(query, opts, filters) {
  const words = parseQuery(query);
  if (!words.length) return { hits: [], expansions: [] };
  const expansions = [];
  for (const [w, ex] of words) expansions.push(await expand(w, ex, opts));
  const active = expansions.map((e) => e.variants.filter((v) => !state.excluded.has(v.form)));
  if (active.some((a) => !a.length)) return { hits: [], expansions };

  // per word: pid -> [[pos, form, d], ...]
  const occ = [];
  for (const vs of active) {
    const m = new Map();
    for (const v of vs) {
      for (const [pid, ps] of await postingsOf(v.form)) {
        if (!m.has(pid)) m.set(pid, []);
        for (const p of ps) m.get(pid).push([p, v.form, v.d]);
      }
    }
    occ.push(m);
  }
  const lengths = expansions.reduce((s, e) => s + [...e.word].length, 0);
  const N = state.meta.n_passages;
  const dfs = new Map(active.flat().map((v) => [v.form, v.df]));
  const idf = (f) => Math.log((N + 1) / ((dfs.get(f) || 0) + 0.5));
  const smallest = occ.reduce((a, b) => (a.size <= b.size ? a : b));

  const hits = [];
  for (const pid of smallest.keys()) {
    if (!occ.every((m) => m.has(pid))) continue;
    const info = issueOf(pid);
    if (!passesFilters(info, filters)) continue;
    const lists = occ.map((m) => m.get(pid).slice().sort((a, b) => a[0] - b[0]));
    const matches = [];
    for (const first of lists[0]) {
      const chain = [first];
      for (const l of lists.slice(1)) {
        const prev = chain[chain.length - 1][0];
        const nxt = l.find((o) => o[0] > prev && o[0] <= prev + opts.near + 1);
        if (!nxt) break;
        chain.push(nxt);
      }
      if (chain.length !== lists.length) continue;
      const dist = chain.reduce((s, o) => s + o[2], 0);
      const score = (1 - dist / Math.max(lengths, 1)) * chain.reduce((s, o) => s + idf(o[1]), 0);
      matches.push({ start: chain[0][0], end: chain[chain.length - 1][0], forms: chain.map((o) => o[1]), dist, score, query });
    }
    if (!matches.length) continue;
    const best = Math.max(...matches.map((m) => m.score));
    matches.sort((a, b) => a.dist - b.dist || a.start - b.start);
    hits.push({ pid, info, matches, score: best * (1 + 0.1 * (Math.min(matches.length, 10) - 1)) });
  }
  return { hits, expansions };
}

async function searchMany(queries, opts, filters) {
  const merged = new Map();
  const expansions = [];
  for (const q of queries) {
    const r = await searchOne(q, opts, filters);
    expansions.push(...r.expansions);
    for (const h of r.hits) {
      const old = merged.get(h.pid);
      if (!old) { merged.set(h.pid, h); continue; }
      const spans = new Set(old.matches.map((m) => m.start + ":" + m.end));
      old.matches.push(...h.matches.filter((m) => !spans.has(m.start + ":" + m.end)));
      old.score = Math.max(old.score, h.score);
    }
  }
  const hits = [...merged.values()];
  for (const h of hits) h.matches.sort((a, b) => a.dist - b.dist || a.start - b.start);
  return { hits, expansions };
}

function cmpDate(a, b) {
  if (a.info.date !== b.info.date) return a.info.date < b.info.date ? -1 : 1;
  if (a.info.page !== b.info.page) return a.info.page - b.info.page;
  return a.pid - b.pid;
}

function applySort(hits) {
  const mode = ($("sort") && $("sort").value) || "oldest";
  if (mode === "newest") hits.sort((a, b) => -cmpDate(a, b) || b.score - a.score);
  else if (mode === "score") hits.sort((a, b) => b.score - a.score || cmpDate(a, b));
  else hits.sort((a, b) => cmpDate(a, b) || b.score - a.score);
  return hits;
}

function currentSort() {
  return ($("sort") && $("sort").value) || "oldest";
}

function sortLabel() {
  return currentSort() === "newest" ? "newest first" : currentSort() === "score" ? "best match first" : "oldest first";
}

function updateSortHint() {
  const el = $("summary-text") && $("summary-text").querySelector(".hint");
  if (el) el.textContent = " · " + sortLabel();
}

// ------------------------------------------------------------------ geometry
function charSpanBox(lines, start, end) {
  const boxes = [];
  for (const [ls, le, x0, y0, x1, y1] of lines) {
    if (le <= start || ls >= end || le <= ls) continue;
    const w = (x1 - x0) / (le - ls);
    boxes.push([x0 + (Math.max(start, ls) - ls) * w, y0, x0 + (Math.min(end, le) - ls) * w, y1]);
  }
  if (!boxes.length) return null;
  return [Math.min(...boxes.map((b) => b[0])), Math.min(...boxes.map((b) => b[1])),
          Math.max(...boxes.map((b) => b[2])), Math.max(...boxes.map((b) => b[3]))];
}
function whereOnPage(cx, cy) {
  const col = cx < 0.36 ? "left column" : cx > 0.64 ? "right column" : "middle column";
  const row = cy < 0.36 ? "top" : cy > 0.64 ? "bottom" : "middle";
  return `${col}, ${row} of the page`;
}

// ------------------------------------------------------------------ rendering
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const prettyDate = (d) => `${+d.slice(8, 10)} ${MONTHS[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const periodLabel = (s) => (s === "S1" ? "before emancipation" : s === "S2" ? "after emancipation" : "outside the study periods");

function status(msg, isError = false) {
  $("status").textContent = msg;
  $("status").className = isError ? "error" : "";
}

function pageLink(info, page) {
  const uuid = info.pages[page - 1];
  return uuid ? HANDLE + uuid : null;
}

function highlighted(text, spans, from, to) {
  let out = "", pos = from;
  for (const [a, b] of spans) {
    if (b <= from || a >= to || a < pos) continue;
    out += esc(text.slice(pos, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>";
    pos = b;
  }
  return out + esc(text.slice(pos, to));
}

async function renderHit(h, rank) {
  const p = await passage(h.pid);
  const toks = tokenize(p.text);
  const spans = h.matches
    .filter((m) => toks[m.start] && toks[m.end])
    .map((m) => [toks[m.start][0], toks[m.end][1]])
    .sort((a, b) => a[0] - b[0]);
  const first = h.matches[0] && toks[h.matches[0].start] ? toks[h.matches[0].start][0] : 0;
  const a = Math.max(0, first - 220), b = Math.min(p.text.length, first + 380);
  const snippet = (a > 0 ? "… " : "") + highlighted(p.text, spans, a, b) + (b < p.text.length ? " …" : "");
  h.snippet = p.text.slice(a, b);
  h.page = p.page;

  const g = p.geo;
  let hint, clipHtml = "";
  if (g.clip) {
    const [x0, y0, x1, y1] = g.box;
    hint = whereOnPage((x0 + x1) / 2 / g.page[0], (y0 + y1) / 2 / g.page[1]);
    const [cx0, cy0, cx1, cy1] = g.clip;
    const boxes = spans.map(([s, e]) => charSpanBox(g.lines, s, e)).filter(Boolean).map(([bx0, by0, bx1, by1]) => {
      const l = ((bx0 - cx0) / (cx1 - cx0)) * 100, t = ((by0 - cy0) / (cy1 - cy0)) * 100;
      const w = ((bx1 - bx0) / (cx1 - cx0)) * 100, hh = ((by1 - by0) / (cy1 - cy0)) * 100;
      return `<div class="box" style="left:calc(${l}% - 4px);top:calc(${t}% - 4px);width:calc(${w}% + 8px);height:calc(${hh}% + 8px)"></div>`;
    }).join("");
    clipHtml = `<div class="clip"><img loading="lazy" src="${DATA}clips/${h.pid}.jpg" alt="Picture of the printed notice">${boxes}</div>`;
  } else {
    hint = whereOnPage(g.hint[0], g.hint[1]);
  }
  const link = pageLink(h.info, p.page);
  const rev = reviews()[h.pid] || {};
  const ocrFix = ocrCorrection(h.pid);
  const div = document.createElement("article");
  div.className = "hit" + (clipHtml ? "" : " noclip");
  div.dataset.pid = h.pid;
  div.innerHTML = `
    <div>
      <h3>${rank}. ${prettyDate(h.info.date)} — <i>${esc(h.info.paper.replace(/ \(\d{4}-\d{4}\)$/, ""))}</i></h3>
      <div class="where">Page ${p.page}, ${hint} · ${periodLabel(h.info.stratum)}
        · found: ${[...new Set(h.matches.map((m) => m.forms.join(" ")))].slice(0, 4).map(esc).join(", ")}</div>
      <div class="text">${snippet}</div>
      <div class="links">
        ${link ? `<a href="${link}" target="_blank" rel="noopener">Open the page at the Royal Library ↗</a>`
               : `<span class="hint">No Royal Library link for this issue; search Mediestream for ${h.info.date}.</span>`}
        <button type="button" class="toggle-full">Show full text</button>
        <button type="button" class="edit-ocr">${ocrFix ? "Edit OCR correction" : "Correct OCR"}</button>
      </div>
      <div class="full" hidden>${highlighted(p.text, spans, 0, p.text.length)}</div>
      <div class="ocr-saved"${ocrFix ? "" : " hidden"}><b>Your correction:</b> <span class="ocr-saved-text"></span></div>
      <div class="ocr-editor" hidden>
        <p class="hint">Look at the picture (or the Royal Library page) and type what the print actually says.
          The computer's reading is kept for comparison. This does not change the search index.</p>
        <textarea rows="7" spellcheck="true"></textarea>
        <div class="ocr-actions">
          <button type="button" class="save-ocr primary">Save correction</button>
          <button type="button" class="revert-ocr">Revert to computer OCR</button>
        </div>
      </div>
      <div class="review">Does this describe a woman?
        <button type="button" data-label="yes" aria-pressed="${rev.label === "yes"}">Yes</button>
        <button type="button" data-label="no" aria-pressed="${rev.label === "no"}">No</button>
        <button type="button" data-label="unsure" aria-pressed="${rev.label === "unsure"}">Unsure</button>
        <input type="text" placeholder="Note (optional)" value="${esc(rev.note || "")}">
      </div>
    </div>
    ${clipHtml}`;
  div.querySelector(".toggle-full").onclick = (e) => {
    const f = div.querySelector(".full");
    f.hidden = !f.hidden;
    e.target.textContent = f.hidden ? "Show full text" : "Hide full text";
  };
  for (const btn of div.querySelectorAll(".review button")) {
    btn.onclick = () => {
      const label = btn.dataset.label;
      const cur = reviews()[h.pid];
      const next = cur && cur.label === label ? null : label;
      saveReview(h, next, div.querySelector(".review input").value);
      for (const x of div.querySelectorAll(".review button")) x.setAttribute("aria-pressed", x.dataset.label === next);
    };
  }
  div.querySelector(".review input").onchange = (e) => {
    const cur = reviews()[h.pid];
    saveReview(h, cur ? cur.label : null, e.target.value);
  };
  const editor = div.querySelector(".ocr-editor");
  const area = editor.querySelector("textarea");
  const savedBox = div.querySelector(".ocr-saved");
  const savedText = div.querySelector(".ocr-saved-text");
  const editBtn = div.querySelector(".edit-ocr");
  area.value = ocrFix ? ocrFix.corrected : p.text;
  if (ocrFix) savedText.textContent = ocrFix.corrected;
  const paintOcr = (fix) => {
    editBtn.textContent = fix ? "Edit OCR correction" : "Correct OCR";
    savedBox.hidden = !fix;
    savedText.textContent = fix ? fix.corrected : "";
    area.value = fix ? fix.corrected : p.text;
  };
  editBtn.onclick = () => {
    editor.hidden = !editor.hidden;
    if (!editor.hidden) area.focus();
  };
  editor.querySelector(".save-ocr").onclick = () => {
    const fix = saveOcrCorrection(h, p.text, area.value);
    paintOcr(fix);
    editor.hidden = true;
    status(fix ? "OCR correction saved in this browser." : "Reverted to the computer's OCR.");
  };
  editor.querySelector(".revert-ocr").onclick = () => {
    area.value = p.text;
    paintOcr(saveOcrCorrection(h, p.text, p.text));
    editor.hidden = true;
    status("Reverted to the computer's OCR.");
  };
  return div;
}

async function showMore() {
  const batch = state.hits.slice(state.shown, state.shown + PAGE_SIZE);
  $("more").disabled = true;
  const els = await Promise.all(batch.map((h, i) => renderHit(h, state.shown + i + 1)));
  for (const el of els) $("results").appendChild(el);
  state.shown += batch.length;
  $("more").disabled = false;
  $("more").hidden = state.shown >= state.hits.length;
}

function renderSummary(hits, expansions) {
  const issues = new Set(hits.map((h) => h.info.date + h.info.paperIdx));
  $("summary-text").innerHTML = hits.length
    ? `<b>${hits.length.toLocaleString()}</b> passages in <b>${issues.size.toLocaleString()}</b> newspaper issues
       <span class="hint"> · ${sortLabel()}</span>`
    : "Nothing found. Try another spelling, or remove a filter.";
  const byYear = new Map();
  for (const h of hits) byYear.set(h.info.year, (byYear.get(h.info.year) || 0) + 1);
  const years = [...byYear.keys()].sort();
  const max = Math.max(1, ...byYear.values());
  $("years").innerHTML = years.length > 1 ? years.map((y) =>
    `<div class="bar ${y > 1848 ? "s2" : ""}" style="height:${(byYear.get(y) / max) * 100}%" title="${y}: ${byYear.get(y)} passages"><span>${y}</span></div>`).join("") : "";

  const seen = new Set();
  let html = "";
  for (const e of expansions) {
    if (seen.has(e.word)) continue;
    seen.add(e.word);
    html += `<div class="word">${esc(e.word)}${e.exact ? " (exact spelling)" : ""}</div>`;
    html += e.variants.length ? e.variants.map((v) =>
      `<label><input type="checkbox" data-form="${esc(v.form)}" ${state.excluded.has(v.form) ? "" : "checked"}> ${esc(v.form)}
        <span class="hint">${v.df.toLocaleString()}×</span></label>`).join("") : `<span class="hint">not in the newspapers</span>`;
  }
  $("variants").innerHTML = html;
  $("summary").hidden = false;
}

// ------------------------------------------------------------------ run
function currentFilters() {
  return {
    stratum: $("period").value,
    from: +$("from").value || null,
    to: +$("to").value || null,
    paper: $("paper").value,
  };
}
function currentOpts() {
  const s = state.meta.search;
  return {
    exact: $("exact").checked,
    epsilon: s.epsilon, eta: s.eta, minFuzzy: s.min_fuzzy_len, near: s.near,
    exactTerms: new Set(state.lastQuery && state.lastQuery.lexicon ? state.meta.exact_only : []),
  };
}

async function run(query, { lexicon = false, keepExcluded = false } = {}) {
  if (!keepExcluded) state.excluded.clear();
  state.lastQuery = { query, lexicon };
  const queries = lexicon ? state.meta.lexicon : [query];
  if (!lexicon && !parseQuery(query).length) { status("Type a word to search for."); return; }
  $("results").innerHTML = "";
  $("more").hidden = true;
  $("summary").hidden = true;
  status("Searching…");
  const t0 = performance.now();
  try {
    const { hits, expansions } = await searchMany(queries, currentOpts(), currentFilters());
    state.hits = applySort(hits);
    state.expansions = expansions;
    state.shown = 0;
    renderSummary(hits, expansions);
    status(`Search took ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
    writeHash();
    await showMore();
  } catch (err) {
    console.error(err);
    status("Something went wrong while searching: " + err.message, true);
  }
}

function writeHash() {
  const f = currentFilters();
  const p = new URLSearchParams();
  if (state.lastQuery.lexicon) p.set("lexicon", "1"); else p.set("q", state.lastQuery.query);
  if (f.stratum) p.set("period", f.stratum);
  if (f.from) p.set("from", f.from);
  if (f.to) p.set("to", f.to);
  if (f.paper !== "") p.set("paper", f.paper);
  if ($("exact").checked) p.set("exact", "1");
  if (currentSort() !== "oldest") p.set("sort", currentSort());
  history.replaceState(null, "", "#" + p.toString());
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  $("q").value = p.get("q") || "";
  $("period").value = p.get("period") || "";
  $("from").value = p.get("from") || "";
  $("to").value = p.get("to") || "";
  $("paper").value = p.get("paper") || "";
  $("exact").checked = p.get("exact") === "1";
  $("sort").value = ["oldest", "newest", "score"].includes(p.get("sort")) ? p.get("sort") : "oldest";
  if (p.get("lexicon") === "1") return run("", { lexicon: true });
  if (p.get("q")) return run(p.get("q"));
}

// ------------------------------------------------------------------ reviews & downloads
function reviews() {
  try { return JSON.parse(localStorage.getItem(REVIEW_KEY)) || {}; } catch { return {}; }
}
function ocrCorrections() {
  try { return JSON.parse(localStorage.getItem(OCR_KEY)) || {}; } catch { return {}; }
}
function ocrCorrection(pid) {
  return ocrCorrections()[pid] || null;
}
function saveOcrCorrection(h, original, corrected) {
  const all = ocrCorrections();
  const next = (corrected || "").replace(/\r\n/g, "\n");
  const orig = (original || "").replace(/\r\n/g, "\n");
  if (!next.trim() || next === orig) {
    delete all[h.pid];
    localStorage.setItem(OCR_KEY, JSON.stringify(all));
    updateReviewCount();
    return null;
  }
  all[h.pid] = {
    original: orig,
    corrected: next,
    time: new Date().toISOString(),
    query: state.lastQuery && state.lastQuery.lexicon ? "(all words for women)" : (state.lastQuery && state.lastQuery.query) || "",
    date: h.info.date,
    newspaper: h.info.paper,
    page: h.page,
    link: pageLink(h.info, h.page) || "",
  };
  localStorage.setItem(OCR_KEY, JSON.stringify(all));
  updateReviewCount();
  return all[h.pid];
}
function saveReview(h, label, note) {
  const all = reviews();
  if (!label && !note) delete all[h.pid];
  else all[h.pid] = {
    label, note, time: new Date().toISOString(),
    query: state.lastQuery.lexicon ? "(all words for women)" : state.lastQuery.query,
    date: h.info.date, newspaper: h.info.paper, page: h.page,
    link: pageLink(h.info, h.page) || "", text: h.snippet || "",
  };
  localStorage.setItem(REVIEW_KEY, JSON.stringify(all));
  updateReviewCount();
}
function updateReviewCount() {
  const all = Object.values(reviews());
  const n = (l) => all.filter((r) => r.label === l).length;
  const nOcr = Object.keys(ocrCorrections()).length;
  const checks = all.length
    ? `${all.length} checked: ${n("yes")} yes, ${n("no")} no, ${n("unsure")} unsure`
    : "No passages checked yet";
  $("review-count").textContent = nOcr ? `${checks}; ${nOcr} OCR correction${nOcr === 1 ? "" : "s"}.` : `${checks}.`;
}
function csv(rows) {
  const cell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return "\ufeff" + rows.map((r) => r.map(cell).join(",")).join("\r\n");
}
function downloadBlob(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function download(name, text) {
  downloadBlob(name, new Blob([text], { type: "text/csv;charset=utf-8" }));
}

const CRC32_TAB = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c;
  }
  return t;
})();
function crc32(u8) {
  let c = 0xffffffff;
  for (let i = 0; i < u8.length; i++) c = CRC32_TAB[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function concatBytes(parts) {
  const n = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
function u16le(n) { return new Uint8Array([n & 255, (n >>> 8) & 255]); }
function u32le(n) { return new Uint8Array([n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255]); }
function zipStore(files) {
  const enc = new TextEncoder();
  const locals = [], centrals = [];
  let offset = 0;
  for (const f of files) {
    const name = enc.encode(f.name);
    const data = f.data;
    const crc = crc32(data);
    const local = concatBytes([
      u32le(0x04034b50), u16le(20), u16le(0), u16le(0), u16le(0), u16le(0),
      u32le(crc), u32le(data.length), u32le(data.length), u16le(name.length), u16le(0),
      name, data,
    ]);
    const central = concatBytes([
      u32le(0x02014b50), u16le(20), u16le(20), u16le(0), u16le(0), u16le(0), u16le(0),
      u32le(crc), u32le(data.length), u32le(data.length),
      u16le(name.length), u16le(0), u16le(0), u16le(0), u16le(0), u32le(0), u32le(offset),
      name,
    ]);
    locals.push(local);
    centrals.push(central);
    offset += local.length;
  }
  const central = concatBytes(centrals);
  const end = concatBytes([
    u32le(0x06054b50), u16le(0), u16le(0), u16le(files.length), u16le(files.length),
    u32le(central.length), u32le(offset), u16le(0),
  ]);
  return concatBytes([...locals, central, end]);
}

function matchedSurface(text, matches) {
  const toks = tokenize(text);
  const seen = new Set();
  const out = [];
  for (const m of matches) {
    if (!toks[m.start] || !toks[m.end]) continue;
    const s = text.slice(toks[m.start][0], toks[m.end][1]);
    if (!seen.has(s)) { seen.add(s); out.push(s); }
  }
  return out.join(" | ");
}

async function poolMap(items, limit, fn) {
  const out = new Array(items.length);
  let i = 0;
  async function worker() {
    while (i < items.length) {
      const j = i++;
      out[j] = await fn(items[j], j);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

function clipUrl(pid) {
  return new URL(`${DATA}clips/${pid}.jpg`, location.href).href;
}

async function exportSearchList() {
  if (!state.hits.length) { status("Search first, then download."); return; }
  const btn = $("download-results");
  btn.disabled = true;
  status("Preparing the spreadsheet (including OCR text)…");
  try {
    const rows = [["rank", "date", "newspaper", "page", "ocr text", "corrected ocr", "matched ocr",
                   "words found", "clip file", "clip url", "passage id", "period", "royal library link"]];
    const clipPids = [];
    for (let i = 0; i < state.hits.length; i++) {
      const h = state.hits[i];
      const p = await passage(h.pid);
      const ocr = p && p.text ? p.text : "";
      const fix = ocrCorrection(h.pid);
      const hasClip = !!(p && p.geo && (p.geo.clip || p.geo.lines));
      if (hasClip) clipPids.push(h.pid);
      rows.push([
        i + 1, h.info.date, h.info.paper, p.page, ocr, fix ? fix.corrected : "",
        matchedSurface(ocr, h.matches),
        [...new Set(h.matches.map((m) => m.forms.join(" ")))].join("; "),
        hasClip ? `clips/${h.pid}.jpg` : "",
        hasClip ? clipUrl(h.pid) : "",
        h.pid, periodLabel(h.info.stratum),
        pageLink(h.info, p.page) || "",
      ]);
    }
    const csvText = csv(rows);
    let got = 0, missed = 0;
    if (clipPids.length) {
      status(`Fetching ${clipPids.length} pictures…`);
      const files = [{ name: "results.csv", data: new TextEncoder().encode(csvText) }];
      await poolMap(clipPids, 8, async (pid) => {
        try {
          const r = await fetch(`${DATA}clips/${pid}.jpg`);
          if (r.ok) {
            files.push({ name: `clips/${pid}.jpg`, data: new Uint8Array(await r.arrayBuffer()) });
            got++;
          } else missed++;
        } catch { missed++; }
        if ((got + missed) % 20 === 0 || got + missed === clipPids.length) {
          status(`Fetching pictures… ${got + missed}/${clipPids.length}`);
        }
      });
      if (got) {
        downloadBlob("search-results.zip", new Blob([zipStore(files)], { type: "application/zip" }));
        status(`Saved ${state.hits.length} passages (OCR in results.csv) and ${got} pictures.`
               + (missed ? ` ${missed} pictures were missing on the server.` : "")
               + " Unzip and open results.csv; ocr text is the fifth column.");
        return;
      }
    }
    download("search-results.csv", csvText);
    status(`Saved ${state.hits.length} passages with OCR text.`
           + (clipPids.length ? " Pictures were not on the server, so only the spreadsheet was saved. The clip url column still points to where they would be." : ""));
  } catch (err) {
    console.error(err);
    status("Could not prepare the download: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
}

function init() {
  $("search-form").onsubmit = (e) => { e.preventDefault(); run($("q").value); };
  for (const chip of document.querySelectorAll(".chip[data-q]")) chip.onclick = () => { $("q").value = chip.dataset.q; run(chip.dataset.q); };
  $("lexicon-btn").onclick = () => { $("q").value = ""; run("", { lexicon: true }); };
  $("sort").onchange = async () => {
    writeHash();
    updateSortHint();
    if (!state.hits.length) return;
    applySort(state.hits);
    state.shown = 0;
    $("results").innerHTML = "";
    $("more").hidden = true;
    await showMore();
  };
  $("more").onclick = showMore;
  $("rerun").onclick = () => {
    state.excluded = new Set([...document.querySelectorAll("#variants input:not(:checked)")].map((x) => x.dataset.form));
    run(state.lastQuery.query, { lexicon: state.lastQuery.lexicon, keepExcluded: true });
  };
  $("copy-link").onclick = async () => {
    try { await navigator.clipboard.writeText(location.href); status("Link copied. Paste it into an email to share this search."); }
    catch { status("Copy this link from the address bar: " + location.href); }
  };
  $("download-results").onclick = () => exportSearchList();
  $("reviewer").value = localStorage.getItem(REVIEW_KEY + "-name") || "";
  $("reviewer").onchange = (e) => localStorage.setItem(REVIEW_KEY + "-name", e.target.value);
  $("download-reviews").onclick = () => {
    const who = $("reviewer").value;
    const labels = reviews();
    const fixes = ocrCorrections();
    const pids = [...new Set([...Object.keys(labels), ...Object.keys(fixes)])];
    const rows = [["reviewer", "passage id", "describes a woman", "note", "date", "newspaper", "page", "search",
                   "ocr text", "corrected ocr", "royal library link", "checked at"]];
    for (const pid of pids) {
      const r = labels[pid] || {};
      const f = fixes[pid] || {};
      rows.push([who, pid, r.label || "", r.note || "", r.date || f.date || "", r.newspaper || f.newspaper || "",
                 r.page || f.page || "", r.query || f.query || "", f.original || r.text || "",
                 f.corrected || "", r.link || f.link || "", r.time || f.time || ""]);
    }
    download(`checks-${(who || "reviewer").replace(/\W+/g, "_")}-${new Date().toISOString().slice(0, 10)}.csv`, csv(rows));
  };
  $("clear-reviews").onclick = () => {
    if (confirm("Delete all your checks and OCR corrections from this browser? Download them first if you need them.")) {
      localStorage.removeItem(REVIEW_KEY);
      localStorage.removeItem(OCR_KEY);
      updateReviewCount();
      for (const b of document.querySelectorAll(".review button")) b.setAttribute("aria-pressed", "false");
      for (const box of document.querySelectorAll(".ocr-saved")) box.hidden = true;
      for (const btn of document.querySelectorAll(".edit-ocr")) btn.textContent = "Correct OCR";
    }
  };
  updateReviewCount();
}

(async function main() {
  init();
  try {
    state.meta = await (await fetch(DATA + "meta.json")).json();
  } catch (err) {
    status("Could not load the newspaper index. If you opened this file directly from your computer, " +
           "use the website address instead.", true);
    return;
  }
  const m = state.meta;
  $("corpus-size").textContent = `${m.n_issues.toLocaleString()} issues (${m.issues[0][2].slice(0, 4)}–${m.issues.reduce((a, i) => (i[2] > a ? i[2] : a), "").slice(0, 4)})`;
  $("built").textContent = `Index built ${m.built}.`;
  m.papers.forEach((p, i) => $("paper").insertAdjacentHTML("beforeend", `<option value="${i}">${esc(p)}</option>`));
  window.addEventListener("hashchange", readHash);
  await readHash();
})();
