/* ============================================================
   SAM Section — main thread UI / interaction controller

   The worker pre-maps every object once (grid sweep over the
   cached embedding) and streams compact masks here. Hovering is
   then a pure hit-test against those masks — instant, no model
   calls per cursor move.
   ============================================================ */

const MAX_SIDE = 1024; // SAM operates on a 1024px long side.

const SAMPLES = [
  { name: "Dog", src: "assets/img/samples/dog.jpg" },
  { name: "Truck", src: "assets/img/samples/truck.jpg" },
  { name: "Groceries", src: "assets/img/samples/groceries.jpg" },
];

// Grid density for pre-mapping (a denser grid finds more/smaller objects).
const GRID = { webgpu: 24, wasm: 14 };
const MAX_OBJECTS = 220;

/* ---- DOM --------------------------------------------------- */
const $ = (id) => document.getElementById(id);
const statusPill = $("statusPill"), statusText = $("statusText");
const devicePill = $("devicePill"), deviceText = $("deviceText");
const dropzone = $("dropzone"), fileInput = $("fileInput");
const browseBtn = $("browseBtn"), pasteHint = $("pasteHint"), sampleRow = $("sampleRow");
const newImageBtn = $("newImageBtn"), helpBtn = $("helpBtn"), help = $("help");
const stage = $("stage"), canvasWrap = $("canvasWrap");
const imageCanvas = $("imageCanvas"), overlayCanvas = $("overlayCanvas");
const hud = $("hud"), hudPick = $("hudPick"), hudScale = $("hudScale"), hudScore = $("hudScore"), hudMeter = $("hudMeter");
const loader = $("loader"), loaderTitle = $("loaderTitle"), loaderMsg = $("loaderMsg"), loaderBar = $("loaderBar");
const dock = $("dock"), scaleLabel = $("scaleLabel");
const sizeUp = $("sizeUp"), sizeDown = $("sizeDown");
const freezeBtn = $("freezeBtn"), removeBtn = $("removeBtn");
const undoBtn = $("undoBtn"), resetBtn = $("resetBtn"), downloadBtn = $("downloadBtn");
const toastEl = $("toast");
const webgpuNotice = $("webgpuNotice"), noticeClose = $("noticeClose");

const imageCtx = imageCanvas.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlayCanvas.getContext("2d");

const PICK = { fill: [124, 92, 255, 84], edge: [186, 160, 255, 255] };
const FROZEN = { fill: [255, 92, 200, 86], edge: [255, 178, 224, 255] };

/* ---- State ------------------------------------------------- */
const state = {
  ready: false,
  device: "—",
  hasImage: false,
  mapping: false,
  pendingGenerate: false,
  jobId: 0,
  W: 0, H: 0,
  originalImageData: null,
  overlayImage: null,
  objects: [],            // all pre-mapped objects: {id,bbox,w,h,area,score,bits}
  candidates: null,       // objects under the cursor, sorted largest→smallest
  index: 0,               // scrub index into candidates
  frozen: false,
  frozenObj: null,
  undo: [],
};
const MAX_UNDO = 15;

/* ---- Worker ------------------------------------------------ */
const params = new URLSearchParams(location.search);
const worker = new Worker("assets/js/sam-worker.js", { type: "module" });
worker.addEventListener("message", onWorkerMessage);
worker.addEventListener("error", () => setStatus("error", "Worker failed to start"));
worker.postMessage({ type: "load", backend: params.get("backend") || undefined });

/* ---- Small helpers ----------------------------------------- */
let toastTimer;
function toast(msg, isErr = false) {
  toastEl.textContent = msg;
  toastEl.classList.toggle("err", isErr);
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2600);
}
function setStatus(kind, text) {
  statusPill.classList.remove("is-loading", "is-ready", "is-error");
  statusPill.classList.add(`is-${kind}`);
  statusText.textContent = text;
}
function showLoader(title, msg, pct) {
  loaderTitle.textContent = title;
  loaderMsg.textContent = msg || "";
  loader.classList.add("show");
  loaderBar.style.width = (pct ?? 0) + "%";
  loaderBar.parentElement.style.visibility = pct == null ? "hidden" : "visible";
}
function hideLoader() { loader.classList.remove("show"); }

/* ---- Worker messages --------------------------------------- */
function onWorkerMessage(e) {
  const m = e.data;
  switch (m.type) {
    case "progress":
      if (state.mapping) setStatus("loading", m.text);
      else { showLoader("Preparing SAM", m.text, m.pct); setStatus("loading", "Loading model…"); }
      break;
    case "ready":
      state.ready = true;
      state.device = m.device;
      deviceText.textContent = m.device === "webgpu" ? "WebGPU" : "WASM · CPU";
      devicePill.title = m.device === "webgpu"
        ? "Running on your GPU via WebGPU"
        : "WebGPU unavailable — running on CPU (slower mapping)";
      setStatus("ready", "Ready");
      if (m.device !== "webgpu" && localStorage.getItem("sam-webgpu-dismissed") !== "1") {
        webgpuNotice.hidden = false;
      }
      if (state.pendingGenerate) doGenerate();
      else hideLoader();
      break;
    case "encoded":
      if (m.id !== state.jobId) return;
      showLoader("Mapping objects", "Finding every object — this runs once per image…", 0);
      break;
    case "mapping_start":
      if (m.id !== state.jobId) return;
      state.mapping = true;
      hideLoader();
      hud.classList.add("show");
      dock.classList.add("show");
      canvasWrap.classList.remove("cursor-busy");
      canvasWrap.classList.add("cursor-pick");
      setStatus("loading", "Mapping objects · 0%");
      updateHUD();
      break;
    case "object":
      if (m.id !== state.jobId) return;
      state.objects.push(m.obj);
      break;
    case "objects_done":
      if (m.id !== state.jobId) return;
      state.mapping = false;
      setStatus("ready", `Pick anything · ${m.count} objects`);
      refreshButtons();
      break;
    case "error":
      console.error("[SAM]", m.message);
      if (m.fatal) { setStatus("error", "Model failed to load"); showLoader("Couldn't start SAM", m.message + " — try reloading.", null); }
      else { toast("Error: " + m.message, true); state.mapping = false; setStatus("error", "Mapping failed"); }
      break;
  }
}

/* ---- Image intake ------------------------------------------ */
browseBtn.addEventListener("click", () => fileInput.click());
newImageBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) loadFromFile(f);
  fileInput.value = "";
});
pasteHint.addEventListener("click", () => toast("Press ⌘/Ctrl + V to paste an image"));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadFromFile(f);
});
window.addEventListener("paste", (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const it of items) if (it.type.startsWith("image/")) { loadFromFile(it.getAsFile()); break; }
});

SAMPLES.forEach((s) => {
  const b = document.createElement("button");
  b.className = "sample";
  b.title = s.name;
  b.innerHTML = `<img src="${s.src}" alt="${s.name}" />`;
  b.addEventListener("click", () => loadFromURL(s.src));
  sampleRow.appendChild(b);
});

function loadFromFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { onImageReady(img); URL.revokeObjectURL(url); };
  img.onerror = () => { toast("Could not read that image.", true); URL.revokeObjectURL(url); };
  img.src = url;
}
function loadFromURL(src) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => onImageReady(img);
  img.onerror = () => toast("Could not load image (blocked or offline).", true);
  img.src = src;
}

/* ---- Prepare an image -------------------------------------- */
function onImageReady(img) {
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const W = Math.max(1, Math.round(img.naturalWidth * scale));
  const H = Math.max(1, Math.round(img.naturalHeight * scale));

  state.W = W; state.H = H;
  imageCanvas.width = W; imageCanvas.height = H;
  overlayCanvas.width = W; overlayCanvas.height = H;
  imageCtx.clearRect(0, 0, W, H);
  imageCtx.drawImage(img, 0, 0, W, H);

  state.originalImageData = imageCtx.getImageData(0, 0, W, H);
  state.overlayImage = overlayCtx.createImageData(W, H);

  // reset interaction
  state.hasImage = true;
  state.mapping = false;
  state.objects = [];
  state.candidates = null;
  state.index = 0;
  state.frozen = false;
  state.frozenObj = null;
  state.undo = [];
  clearOverlay();
  refreshButtons();

  dropzone.style.display = "none";
  canvasWrap.classList.remove("hidden");
  newImageBtn.hidden = false;
  fitImage();

  if (state.ready) doGenerate();
  else { state.pendingGenerate = true; showLoader("Preparing SAM", "Waiting for the model to finish loading…", null); }
}

function doGenerate() {
  state.pendingGenerate = false;
  state.jobId++;
  const id = state.jobId;
  showLoader("Reading the image", "Encoding on " + (state.device === "webgpu" ? "your GPU" : "CPU") + "…", null);
  setStatus("loading", "Encoding…");
  canvasWrap.classList.add("cursor-busy");

  const pointsPerSide = GRID[state.device] || GRID.wasm;
  const snapshot = imageCtx.getImageData(0, 0, state.W, state.H);
  worker.postMessage(
    { type: "generate", id, width: state.W, height: state.H, data: snapshot.data.buffer, pointsPerSide, maxObjects: MAX_OBJECTS },
    [snapshot.data.buffer]
  );
}

/* ---- Display sizing ---------------------------------------- */
function fitImage() {
  if (!state.hasImage) return;
  const pad = 48;
  const availW = stage.clientWidth - pad;
  const availH = stage.clientHeight - 150;
  const ar = state.W / state.H;
  let w = Math.min(availW, 1100);
  let h = w / ar;
  if (h > availH) { h = availH; w = h * ar; }
  canvasWrap.style.width = Math.round(w) + "px";
  canvasWrap.style.height = Math.round(h) + "px";
}
let resizeRAF;
window.addEventListener("resize", () => { cancelAnimationFrame(resizeRAF); resizeRAF = requestAnimationFrame(fitImage); });

/* ---- Hit-testing (instant hover) --------------------------- */
function hitTest(px, py) {
  const out = [];
  for (const o of state.objects) {
    const [x0, y0, x1, y1] = o.bbox;
    if (px < x0 || px >= x1 || py < y0 || py >= y1) continue;
    if (o.bits[(py - y0) * o.w + (px - x0)]) out.push(o);
  }
  out.sort((a, b) => b.area - a.area); // largest → smallest
  return out;
}

canvasWrap.addEventListener("mousemove", (e) => {
  if (state.frozen || !state.objects.length) return;
  const r = imageCanvas.getBoundingClientRect();
  const px = Math.floor(((e.clientX - r.left) / r.width) * state.W);
  const py = Math.floor(((e.clientY - r.top) / r.height) * state.H);
  if (px < 0 || py < 0 || px >= state.W || py >= state.H) { clearHover(); return; }
  const list = hitTest(px, py);
  if (!list.length) { clearHover(); return; }
  state.candidates = list;
  state.index = Math.min(state.index, list.length - 1);
  renderObject(list[state.index], PICK);
  refreshButtons();
  updateHUD();
});
canvasWrap.addEventListener("mouseleave", () => { if (!state.frozen) clearHover(); });

function clearHover() {
  state.candidates = null;
  clearOverlay();
  refreshButtons();
  updateHUD();
}

/* ---- Overlay rendering ------------------------------------- */
function clearOverlay() { overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); }

function renderObject(obj, color) {
  const W = state.W;
  const img = state.overlayImage;
  const data = img.data;
  data.fill(0);
  const [x0, y0] = obj.bbox;
  const bw = obj.w, bh = obj.h, bits = obj.bits;
  const [fr, fg, fb, fa] = color.fill;
  const [er, eg, eb, ea] = color.edge;
  for (let yy = 0; yy < bh; yy++) {
    for (let xx = 0; xx < bw; xx++) {
      if (!bits[yy * bw + xx]) continue;
      const edge =
        xx === 0 || yy === 0 || xx === bw - 1 || yy === bh - 1 ||
        !bits[yy * bw + xx - 1] || !bits[yy * bw + xx + 1] ||
        !bits[(yy - 1) * bw + xx] || !bits[(yy + 1) * bw + xx];
      const o = ((y0 + yy) * W + (x0 + xx)) << 2;
      if (edge) { data[o] = er; data[o + 1] = eg; data[o + 2] = eb; data[o + 3] = ea; }
      else { data[o] = fr; data[o + 1] = fg; data[o + 2] = fb; data[o + 3] = fa; }
    }
  }
  overlayCtx.putImageData(img, 0, 0);
}

/* ---- HUD --------------------------------------------------- */
function selectedObj() {
  if (state.frozen) return state.frozenObj;
  return state.candidates ? state.candidates[state.index] : null;
}
function updateHUD() {
  const n = state.candidates ? state.candidates.length : 0;
  const k = n ? state.index + 1 : 0;
  let tag = "";
  if (n) tag = state.index === 0 ? " · largest" : state.index === n - 1 ? " · smallest" : "";
  scaleLabel.textContent = n ? `Size ${k}/${n}` : "Size —";
  hudScale.textContent = n ? `${k}/${n}${tag}` : "—";

  const obj = selectedObj();
  const score = obj ? obj.score : 0;
  hudScore.textContent = obj ? score.toFixed(2) : "—";
  hudMeter.style.width = obj ? Math.round(score * 100) + "%" : "0%";

  const sw = hud.querySelector(".swatch");
  if (sw) sw.style.background = state.frozen ? "var(--magenta)" : "var(--accent)";
  hudPick.textContent = state.frozen
    ? "Frozen — press Delete to remove"
    : n ? "Click or press F to freeze"
    : state.mapping ? "Mapping… hover to pick what's ready"
    : "Move the cursor to pick";
}

/* ---- Freeze / unfreeze ------------------------------------- */
canvasWrap.addEventListener("click", () => {
  if (!state.hasImage) return;
  if (state.frozen) unfreeze();
  else if (state.candidates) freeze();
});
function freeze() {
  const obj = state.candidates && state.candidates[state.index];
  if (!obj) return;
  state.frozen = true;
  state.frozenObj = { ...obj, bits: obj.bits.slice() };
  renderObject(state.frozenObj, FROZEN);
  refreshButtons();
  updateHUD();
}
function unfreeze() {
  state.frozen = false;
  state.frozenObj = null;
  clearOverlay();
  refreshButtons();
  updateHUD();
}

/* ---- Scale scrubbing --------------------------------------- */
function nudgeSize(dir) {
  if (!state.candidates || state.frozen) return;
  const n = state.candidates.length;
  const next = Math.min(n - 1, Math.max(0, state.index + dir));
  if (next === state.index) return;
  state.index = next;
  renderObject(state.candidates[state.index], PICK);
  refreshButtons();
  updateHUD();
}
sizeUp.addEventListener("click", () => nudgeSize(-1));
sizeDown.addEventListener("click", () => nudgeSize(1));
canvasWrap.addEventListener("wheel", (e) => {
  if (!state.candidates || state.frozen) return;
  e.preventDefault();
  nudgeSize(e.deltaY > 0 ? 1 : -1);
}, { passive: false });

/* ---- Removal / history ------------------------------------- */
freezeBtn.addEventListener("click", () => { if (state.frozen) unfreeze(); else freeze(); });
removeBtn.addEventListener("click", removeSelection);

function removeSelection() {
  const obj = selectedObj();
  if (!obj) { toast("Hover an object first."); return; }
  pushUndo();
  const W = state.W;
  const frame = imageCtx.getImageData(0, 0, W, state.H);
  const d = frame.data;
  const [x0, y0] = obj.bbox;
  const bw = obj.w, bh = obj.h, bits = obj.bits;
  for (let yy = 0; yy < bh; yy++) {
    for (let xx = 0; xx < bw; xx++) {
      if (bits[yy * bw + xx]) d[(((y0 + yy) * W + (x0 + xx)) << 2) + 3] = 0;
    }
  }
  imageCtx.putImageData(frame, 0, 0);
  unfreeze();
  clearHover();
  toast("Object removed");
}
function pushUndo() {
  state.undo.push(imageCtx.getImageData(0, 0, state.W, state.H));
  if (state.undo.length > MAX_UNDO) state.undo.shift();
  refreshButtons();
}
undoBtn.addEventListener("click", undo);
function undo() {
  const prev = state.undo.pop();
  if (!prev) return;
  imageCtx.putImageData(prev, 0, 0);
  unfreeze();
  clearHover();
}
resetBtn.addEventListener("click", () => {
  if (!state.originalImageData) return;
  imageCtx.putImageData(state.originalImageData, 0, 0);
  state.undo = [];
  unfreeze();
  clearHover();
  toast("Restored original");
});
downloadBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "sam-cutout.png";
  a.href = imageCanvas.toDataURL("image/png");
  a.click();
});

/* ---- Button states ----------------------------------------- */
function refreshButtons() {
  const hasSel = !!(state.candidates && state.candidates.length) || !!state.frozenObj;
  const canScrub = !!state.candidates && state.candidates.length > 1 && !state.frozen;
  freezeBtn.disabled = !hasSel;
  removeBtn.disabled = !hasSel;
  sizeUp.disabled = !canScrub || state.index === 0;
  sizeDown.disabled = !canScrub || (state.candidates && state.index === state.candidates.length - 1);
  undoBtn.disabled = state.undo.length === 0;
  resetBtn.disabled = state.undo.length === 0;
  downloadBtn.disabled = !state.hasImage;
  freezeBtn.classList.toggle("is-active", state.frozen);
}

/* ---- Keyboard ---------------------------------------------- */
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const k = e.key;
  if ((e.metaKey || e.ctrlKey) && (k === "z" || k === "Z")) { e.preventDefault(); undo(); return; }
  if (!state.hasImage) return;
  switch (k) {
    case "f": case "F":
      e.preventDefault(); state.frozen ? unfreeze() : freeze(); break;
    case "Delete": case "Backspace":
      e.preventDefault(); removeSelection(); break;
    case "Escape":
      if (state.frozen) unfreeze(); break;
    case "ArrowUp":
      e.preventDefault(); nudgeSize(-1); break;
    case "ArrowDown":
      e.preventDefault(); nudgeSize(1); break;
    case "?":
      help.classList.toggle("show"); break;
  }
});

/* ---- WebGPU fallback notice -------------------------------- */
noticeClose.addEventListener("click", () => {
  webgpuNotice.hidden = true;
  localStorage.setItem("sam-webgpu-dismissed", "1");
});

/* ---- Help -------------------------------------------------- */
helpBtn.addEventListener("click", () => help.classList.toggle("show"));
document.addEventListener("click", (e) => {
  if (help.classList.contains("show") && !help.contains(e.target) && e.target !== helpBtn && !helpBtn.contains(e.target))
    help.classList.remove("show");
});
