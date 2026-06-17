/* ============================================================
   SAM inference worker  (pre-mapping edition)

   Strategy: run the heavy encoder ONCE, then sweep a grid of
   point prompts through the lightweight decoder to pre-map every
   object in the image. Masks are deduplicated, cropped to their
   bounding box, and streamed to the UI as they are found, so the
   main thread can offer instant (lookup-only) hover picking.

   Messages IN:
     { type: 'load', backend? }
     { type: 'generate', id, width, height, data(RGBA buffer),
       pointsPerSide, maxObjects }
   Messages OUT:
     { type: 'progress', pct, text }
     { type: 'ready', device }
     { type: 'encoded', id }
     { type: 'mapping_start', id, total }
     { type: 'object', id, dims:[H,W], obj:{id,bbox,w,h,area,score,bits} }
     { type: 'objects_done', id, count }
     { type: 'error', message, fatal }
   ============================================================ */

import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels = false;

const MODEL_ID = "Xenova/slimsam-77-uniform";

let model = null;
let processor = null;
let device = "wasm";

let imageInputs = null;
let imageEmbeddings = null;

/* ---- Model loading ---------------------------------------- */
async function load(preferred) {
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  const tryOrder = preferred === "wasm" ? ["wasm"] : ["webgpu", "wasm"];
  let lastErr = null;
  for (const dev of tryOrder) {
    try {
      const opts =
        dev === "webgpu"
          ? { dtype: "fp16", device: "webgpu" }
          : { dtype: "q8", device: "wasm" };
      model = await SamModel.from_pretrained(MODEL_ID, {
        ...opts,
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            const pct = Math.round((p.loaded / p.total) * 100);
            self.postMessage({ type: "progress", pct, text: `Downloading model · ${pct}%` });
          } else if (p.status === "ready") {
            self.postMessage({ type: "progress", pct: 100, text: "Warming up…" });
          }
        },
      });
      device = dev;
      return;
    } catch (e) {
      lastErr = e;
      model = null;
    }
  }
  throw lastErr || new Error("Could not initialise the model.");
}

/* ---- Decode a single point, return raw masks --------------- */
async function runPoint(nx, ny) {
  const reshaped = imageInputs.reshaped_input_sizes[0]; // [h, w]
  const points = [nx * reshaped[1], ny * reshaped[0]];
  const input_points = new Tensor("float32", points, [1, 1, 1, 2]);
  const input_labels = new Tensor("int64", [1n], [1, 1, 1]);

  const outputs = await model({ ...imageEmbeddings, input_points, input_labels });
  const processed = await processor.post_process_masks(
    outputs.pred_masks,
    imageInputs.original_sizes,
    imageInputs.reshaped_input_sizes
  );
  const t = processed[0];
  const d = t.dims;
  const W = d[d.length - 1];
  const H = d[d.length - 2];
  const num = d.length >= 3 ? d[d.length - 3] : 1;
  return { flat: t.data, W, H, num, scores: Array.from(outputs.iou_scores.data) };
}

/* ---- Pre-map every object ---------------------------------- */
async function generate(msg) {
  const image = new RawImage(new Uint8ClampedArray(msg.data), msg.width, msg.height, 4).rgb();
  imageInputs = await processor(image);
  imageEmbeddings = await model.get_image_embeddings(imageInputs);
  self.postMessage({ type: "encoded", id: msg.id });

  const P = msg.pointsPerSide;
  const total = P * P;
  const minAF = 0.0012; // ignore specks
  const maxAF = 0.92;   // ignore whole-image background blobs

  let W = 0, H = 0, plane = 0, coverage = null;
  let count = 0, done = 0, lastPct = -1, nextId = 0;

  self.postMessage({ type: "mapping_start", id: msg.id, total });

  const progress = () => {
    const pct = Math.round((done / total) * 100);
    if (pct !== lastPct) {
      lastPct = pct;
      self.postMessage({ type: "progress", pct, text: `Mapping objects · ${pct}%` });
    }
  };

  for (let gy = 0; gy < P; gy++) {
    for (let gx = 0; gx < P; gx++) {
      done++;
      const nx = (gx + 0.5) / P;
      const ny = (gy + 0.5) / P;

      // Skip points already inside a discovered object.
      if (coverage) {
        const px = Math.min(W - 1, (nx * W) | 0);
        const py = Math.min(H - 1, (ny * H) | 0);
        if (coverage[py * W + px]) { progress(); continue; }
      }

      const res = await runPoint(nx, ny);
      if (!H) { H = res.H; W = res.W; plane = H * W; coverage = new Uint8Array(plane); }

      const minA = minAF * plane, maxA = maxAF * plane;

      // Pick the highest-scoring mask that passes the area filter.
      const ranked = [...res.scores.keys()].sort((a, b) => res.scores[b] - res.scores[a]);
      let chosen = -1, chosenArea = 0;
      for (const i of ranked) {
        let a = 0;
        const base = i * plane;
        for (let p = 0; p < plane; p++) if (res.flat[base + p] > 0) a++;
        if (a >= minA && a <= maxA) { chosen = i; chosenArea = a; break; }
      }
      if (chosen < 0) { progress(); continue; }

      // Drop near-duplicates: if mostly inside already-mapped area, skip.
      const base = chosen * plane;
      let covered = 0;
      for (let p = 0; p < plane; p++) if (res.flat[base + p] > 0 && coverage[p]) covered++;
      if (covered / chosenArea > 0.5) { progress(); continue; }

      // Accept: compute bbox, mark coverage, crop a compact bitmask.
      let x0 = W, y0 = H, x1 = -1, y1 = -1;
      for (let y = 0; y < H; y++) {
        const row = y * W;
        for (let x = 0; x < W; x++) {
          if (res.flat[base + row + x] > 0) {
            coverage[row + x] = 1;
            if (x < x0) x0 = x; if (x > x1) x1 = x;
            if (y < y0) y0 = y; if (y > y1) y1 = y;
          }
        }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      const bits = new Uint8Array(bw * bh);
      for (let y = 0; y < bh; y++) {
        const srow = (y + y0) * W + x0;
        const drow = y * bw;
        for (let x = 0; x < bw; x++) bits[drow + x] = res.flat[srow + x] > 0 ? 1 : 0;
      }

      count++;
      self.postMessage(
        {
          type: "object",
          id: msg.id,
          dims: [H, W],
          obj: { id: nextId++, bbox: [x0, y0, x1 + 1, y1 + 1], w: bw, h: bh, area: chosenArea, score: res.scores[chosen], bits },
        },
        [bits.buffer]
      );

      if (count >= msg.maxObjects) { self.postMessage({ type: "objects_done", id: msg.id, count }); return; }
      progress();
    }
  }
  self.postMessage({ type: "objects_done", id: msg.id, count });
}

/* ---- Dispatch --------------------------------------------- */
self.addEventListener("message", async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await load(msg.backend);
      self.postMessage({ type: "ready", device });
    } else if (msg.type === "generate") {
      await generate(msg);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      message: (err && err.message) || String(err),
      fatal: msg.type === "load",
    });
  }
});
