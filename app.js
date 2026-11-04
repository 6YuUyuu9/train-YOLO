// Dog Breed Segmentation: YOLO26n-seg (ONNX) รันใน browser ด้วย onnxruntime-web
// ไม่มี server: ภาพไม่ถูกส่งออกจากเครื่องผู้ใช้

const CLASSES = ["beagle", "doberman", "golden_retriever"];
const COLORS = [[255, 159, 28], [58, 134, 255], [46, 196, 102]];
const SIZE = 640;          // imgsz ตอน export
const IOU = 0.5;           // NMS IoU (ต่ำกว่า default 0.7 เพื่อตัดกล่องซ้อนของตัวเดียวกัน)
const MASK_DIM = 160;      // ขนาด prototype mask (SIZE / 4)

const $ = (id) => document.getElementById(id);
let session = null;
let current = null;        // { bitmap, name }

ort.env.wasm.wasmPaths = new URL("./", location.href).href;
ort.env.wasm.numThreads = 1;

async function loadModel() {
  setStatus("กำลังโหลดโมเดล...", "busy");
  try {
    // model.onnx เก็บเป็น base64 (บางโฮสต์ไม่เสิร์ฟไฟล์ .onnx) ถ้ามี model.onnx จริงจะใช้ไฟล์นั้นก่อน
    let buf;
    const raw = await fetch("model.onnx").catch(() => null);
    if (raw && raw.ok && !(raw.headers.get("content-type") || "").includes("text/html")) {
      buf = await raw.arrayBuffer();
    } else {
      const b64 = await (await fetch("model_onnx.b64.txt")).text();
      const bin = atob(b64.trim());
      const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      buf = u8.buffer;
    }
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    setStatus(`โมเดลพร้อม (${(buf.byteLength / 1e6).toFixed(1)} MB) อัปโหลดภาพได้เลย`, "ok");
    if (current) run();
  } catch (e) {
    setStatus("โหลดโมเดลไม่สำเร็จ: " + e.message, "err");
  }
}

function setStatus(text, kind) {
  const s = $("status");
  s.textContent = text;
  s.className = "status " + kind;
}

// ---------- preprocess: letterbox แบบเดียวกับ Ultralytics ----------
function letterbox(bitmap) {
  const W = bitmap.width, H = bitmap.height;
  const r = Math.min(SIZE / W, SIZE / H);
  const nw = Math.round(W * r), nh = Math.round(H * r);
  const left = Math.round((SIZE - nw) / 2 - 0.1), top = Math.round((SIZE - nh) / 2 - 0.1);
  const c = new OffscreenCanvas(SIZE, SIZE);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(bitmap, left, top, nw, nh);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = px[i * 4] / 255;
    data[n + i] = px[i * 4 + 1] / 255;
    data[2 * n + i] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]), r, left, top, nw, nh };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

// ---------- postprocess: output0 [1, 4+nc+32, 8400], output1 [1, 32, 160, 160] ----------
function decode(out0, conf) {
  const [, ch, N] = out0.dims, d = out0.data, nc = CLASSES.length;
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) {
      const s = d[(4 + c) * N + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < conf) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    const coef = new Float32Array(32);
    for (let k = 0; k < 32; k++) coef[k] = d[(4 + nc + k) * N + i];
    cand.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score: best, cls, coef });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cand) {
    if (keep.every((k) => k.cls !== c.cls || iou(k.box, c.box) < IOU)) keep.push(c);
    if (keep.length >= 300) break;
  }
  return keep;
}

// mask 160x160 (ค่า prob เป็น alpha) แล้วตัดนอก box ทิ้ง
function protoMask(det, proto) {
  const P = proto.data, n = MASK_DIM * MASK_DIM;
  const img = new ImageData(MASK_DIM, MASK_DIM);
  const s = MASK_DIM / SIZE;
  const [x1, y1, x2, y2] = det.box.map((v) => v * s);
  for (let y = 0; y < MASK_DIM; y++) {
    for (let x = 0; x < MASK_DIM; x++) {
      if (x < x1 || x >= x2 || y < y1 || y >= y2) continue;
      const p = y * MASK_DIM + x;
      let v = 0;
      for (let k = 0; k < 32; k++) v += det.coef[k] * P[k * n + p];
      img.data[p * 4 + 3] = Math.round(255 / (1 + Math.exp(-v)));
    }
  }
  return img;
}

async function run() {
  if (!session || !current) return;
  const conf = parseFloat($("conf").value);
  const { bitmap } = current;
  const W = bitmap.width, H = bitmap.height;
  const t0 = performance.now();

  const lb = letterbox(bitmap);
  const out = await session.run({ [session.inputNames[0]]: lb.tensor });
  const out0 = out[session.outputNames[0]], proto = out[session.outputNames[1]];
  const dets = decode(out0, conf);
  const ms = performance.now() - t0;

  // วาดผล
  const view = $("view"), cut = $("cut");
  view.width = cut.width = W; view.height = cut.height = H;
  const vctx = view.getContext("2d");
  vctx.drawImage(bitmap, 0, 0);
  const base = vctx.getImageData(0, 0, W, H);
  const overlay = new Uint8ClampedArray(base.data);
  const alpha = new Uint8Array(W * H);
  const classPix = CLASSES.map(() => new Uint8Array(W * H));
  const count = CLASSES.map(() => 0);

  const m160 = new OffscreenCanvas(MASK_DIM, MASK_DIM);
  const mfull = new OffscreenCanvas(W, H);
  const mctx = mfull.getContext("2d", { willReadFrequently: true });
  const k = MASK_DIM / SIZE;

  for (const det of dets) {
    m160.getContext("2d").putImageData(protoMask(det, proto), 0, 0);
    mctx.clearRect(0, 0, W, H);
    mctx.imageSmoothingEnabled = true;
    mctx.drawImage(m160, lb.left * k, lb.top * k, lb.nw * k, lb.nh * k, 0, 0, W, H);
    const a = mctx.getImageData(0, 0, W, H).data;
    const col = COLORS[det.cls];
    count[det.cls]++;
    for (let p = 0; p < W * H; p++) {
      if (a[p * 4 + 3] > 127) {
        alpha[p] = 255;
        classPix[det.cls][p] = 1;
        for (let c = 0; c < 3; c++) overlay[p * 4 + c] = overlay[p * 4 + c] * 0.55 + col[c] * 0.45;
      }
    }
  }
  vctx.putImageData(new ImageData(overlay, W, H), 0, 0);

  // กรอบ + ชื่อ
  const lw = Math.max(2, Math.round(Math.max(W, H) / 300));
  vctx.lineWidth = lw;
  vctx.font = `600 ${Math.max(13, Math.round(Math.max(W, H) / 40))}px system-ui, sans-serif`;
  for (const det of dets) {
    const [x1, y1, x2, y2] = det.box.map((v, i) =>
      Math.min(Math.max((v - (i % 2 ? lb.top : lb.left)) / lb.r, 0), i % 2 ? H : W));
    const col = `rgb(${COLORS[det.cls].join(",")})`;
    vctx.strokeStyle = col;
    vctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${CLASSES[det.cls]} ${det.score.toFixed(2)}`;
    const tw = vctx.measureText(label).width + 10, th = parseInt(vctx.font.split(" ")[1]) + 8;
    const ty = y1 - th < 0 ? y1 : y1 - th;
    vctx.fillStyle = col;
    vctx.fillRect(x1 - lw / 2, ty, tw, th);
    vctx.fillStyle = "#fff";
    vctx.fillText(label, x1 + 5, ty + th - 7);
  }

  // ตัดพื้นหลัง
  const cutData = new Uint8ClampedArray(base.data);
  for (let p = 0; p < W * H; p++) cutData[p * 4 + 3] = alpha[p];
  cut.getContext("2d").putImageData(new ImageData(cutData, W, H), 0, 0);

  // ตาราง
  const rows = CLASSES.map((c, i) => {
    let px = 0;
    for (const v of classPix[i]) px += v;
    return { c, i, n: count[i], pct: (100 * px) / (W * H) };
  }).filter((r) => r.n > 0);
  $("table").innerHTML = rows.length
    ? rows.map((r) => `<tr><td><span class="dot" style="background:rgb(${COLORS[r.i]})"></span>${r.c}</td>
        <td>${r.n}</td><td>${r.pct.toFixed(1)}%</td></tr>`).join("")
    : `<tr><td colspan="3" class="muted">ไม่พบสุนัขที่ confidence ≥ ${conf.toFixed(2)}</td></tr>`;
  $("total").textContent = `พบทั้งหมด ${dets.length} ตัว`;
  $("time").textContent = `${current.name} · ${W}×${H} · ${ms.toFixed(0)} ms`;
  $("result").hidden = false;
  window.__lastResult = dets.map((d) => ({ cls: CLASSES[d.cls], score: +d.score.toFixed(3) }));
}

async function useBlob(blob, name) {
  current = { bitmap: await createImageBitmap(blob), name };
  run();
}

$("file").addEventListener("change", (e) => e.target.files[0] && useBlob(e.target.files[0], e.target.files[0].name));
const drop = $("drop");
["dragover", "dragenter"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
  if (f) useBlob(f, f.name);
});
$("conf").addEventListener("input", () => { $("confv").textContent = (+$("conf").value).toFixed(2); });
$("conf").addEventListener("change", run);

document.querySelectorAll("[data-example]").forEach(async (btn) => {
  const name = btn.dataset.example;
  const blob = await (await fetch("examples/" + name)).blob();
  const img = btn.querySelector("img");
  img.src = URL.createObjectURL(blob);
  btn.addEventListener("click", () => useBlob(blob, name));
});

$("save").addEventListener("click", () => {
  $("cut").toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = (current?.name || "dog").replace(/\.\w+$/, "") + "_cutout.png";
    a.click();
  });
});

loadModel();
