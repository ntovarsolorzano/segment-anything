# SAM Studio — on-device Segment Anything (web)

A self-contained, **front-end-only** website that brings Meta's
[Segment Anything Model (SAM)](https://segment-anything.com/) into the browser.

- **`index.html`** — high-end marketing landing page that links into the tool.
- **`sam.html`** — the **SAM Section**: upload an image, glide the cursor to pick
  any object (largest → smallest), freeze it, and remove it.

Everything — the heavy image **encoder** *and* the lightweight **decoder** — runs
locally via [transformers.js](https://github.com/huggingface/transformers.js) on
**WebGPU**, with an automatic **WASM (CPU)** fallback. No backend, no uploads.

## How it works

SAM is split into two very unequal halves, and we exploit that to make picking
feel instant:

1. **Encoder (runs once per image).** A distilled ViT
   ([`Xenova/slimsam-77-uniform`](https://huggingface.co/Xenova/slimsam-77-uniform),
   ~40 MB) turns the image into an embedding. This is the only "slow" step.
2. **Pre-mapping (runs once per image).** Instead of decoding on every cursor
   move, the worker sweeps a grid of point prompts through the cached embedding
   to discover **every object up front**. It skips points already inside a found
   object (coverage) and drops near-duplicates, then crops each mask to its
   bounding box and **streams it to the UI** as it's found — so you can start
   picking before the sweep finishes.
3. **Picking (instant).** Hovering is now a pure hit-test against the pre-mapped
   masks — no model calls. Objects under the cursor are stacked largest→smallest
   for the size scrub.

Inference lives in a **Web Worker** (`assets/js/sam-worker.js`) so the UI thread
stays responsive. The main thread (`assets/js/sam-app.js`) handles the canvas,
the hit-test hover loop, freezing, removal, undo and export.

> Pre-mapping is heavier up front than per-hover decoding, but it happens once
> and is dramatically faster to *use*. Grid density adapts to the backend
> (denser on WebGPU, lighter on CPU); see `GRID` in `sam-app.js`.

## Controls

| Action | Mouse | Keyboard |
| --- | --- | --- |
| Pick the object under the cursor | move | — |
| Grow / shrink the selection (biggest → smallest) | scroll wheel | `↑` / `↓` |
| Freeze the current selection | click | `F` |
| Remove the frozen object | Remove button | `Delete` / `Backspace` |
| Unfreeze / clear | — | `Esc` |
| Undo last removal | Undo button | `⌘/Ctrl + Z` |
| Restore original | Reset button | — |
| Export transparent PNG | Save button | — |

## Running it

It's a static site — serve the `webapp/` folder over HTTP(S) and open `index.html`.
WebGPU requires a **secure context** (`https://` or `http://localhost`).

```bash
cd webapp
python3 -m http.server 8080
# open http://localhost:8080/
```

> **Tip — best performance:** the WASM fallback is fastest with multithreading,
> which needs cross-origin isolation. If you control the server, send:
>
> ```
> Cross-Origin-Opener-Policy: same-origin
> Cross-Origin-Embedder-Policy: require-corp
> ```
>
> WebGPU itself does not require these headers. You can also force the CPU path
> for debugging with `sam.html?backend=wasm`.

## Requirements & notes

- A WebGPU-capable browser (recent Chrome/Edge; Safari/Firefox behind flags) gives
  the best experience. Without WebGPU the app falls back to CPU automatically.
- First load downloads the model (~40 MB) from the Hugging Face CDN and caches it;
  subsequent loads are instant and work offline.
- Images are processed at a 1024px long side (SAM's working resolution) and never
  leave your device.
- "Remove" erases the selected pixels to transparency (a clean cut-out), which is
  why exports are PNGs. It is not content-aware fill/inpainting.

## Credits

Segment Anything © Meta AI Research (FAIR). In-browser inference via
transformers.js / ONNX Runtime Web. SlimSAM distilled checkpoint by the SlimSAM
authors, ONNX weights published by Xenova on the Hugging Face Hub.
