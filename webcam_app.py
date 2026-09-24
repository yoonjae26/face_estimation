"""Real-time webcam demo in the browser (works when the code runs on a remote server).

The browser on *your* computer captures the webcam and sends frames to this server,
which runs the models and sends back age / gender / emotion for every face.

    python webcam_app.py            # then open http://localhost:8000 in your browser
    python webcam_app.py --port 8080

With VS Code Remote-SSH the port is forwarded automatically (see the "Ports" tab).
Otherwise forward it yourself:  ssh -L 8000:localhost:8000 <user>@<server>
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from utils import config  # noqa: E402
from utils.predictor import FaceAnalyzer  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Estimation Live</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --text:#e8eaf0; --muted:#8a90a0; --accent:#4f8cff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.4 system-ui,sans-serif; }
  main { max-width:1200px; margin:0 auto; padding:16px; display:flex; gap:16px; flex-wrap:wrap; }
  .stage { flex:1 1 640px; min-width:0; }
  canvas { width:100%; border-radius:12px; background:#000; display:block; }
  aside { flex:0 1 320px; background:var(--panel); border-radius:12px; padding:16px; }
  h1 { font-size:18px; margin:0 0 12px; }
  .big { font-size:34px; font-weight:700; margin:2px 0 10px; }
  .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
  .row { display:flex; align-items:center; gap:8px; margin:5px 0; font-size:13px; }
  .row span:first-child { width:66px; }
  .bar { flex:1; height:10px; background:#262a33; border-radius:5px; overflow:hidden; }
  .bar div { height:100%; background:var(--accent); transition:width .15s; }
  .row span:last-child { width:40px; text-align:right; color:var(--muted); }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 16px; font-size:15px;
           cursor:pointer; width:100%; margin-bottom:12px; }
  #status { color:var(--muted); font-size:13px; margin-top:10px; }
</style></head><body><main>
<div class="stage"><canvas id="view" width="640" height="480"></canvas></div>
<aside>
  <h1>Face Estimation Live</h1>
  <button id="start">Start webcam</button>
  <div class="label">Age</div><div class="big" id="age">–</div>
  <div class="label">Gender</div><div class="big" id="gender">–</div>
  <div class="label">Emotion</div><div class="big" id="emotion">–</div>
  <div id="bars"></div>
  <div id="status">Click "Start webcam" and allow camera access.</div>
</aside></main>
<script>
const EMOTIONS = __EMOTIONS__;
const COLORS = {Angry:"#dc2828",Disgust:"#3c8c28",Fear:"#a03ca0",Happy:"#f0c828",Neutral:"#b4b4b4",Sad:"#2878c8",
                Surprise:"#ff961e"};
const view = document.getElementById("view"), ctx = view.getContext("2d");
const grab = document.createElement("canvas"), gctx = grab.getContext("2d");
const video = document.createElement("video"); video.playsInline = true; video.muted = true;
const bars = document.getElementById("bars"), statusEl = document.getElementById("status");
bars.innerHTML = EMOTIONS.map(e => `<div class="row"><span>${e}</span><div class="bar"><div id="b-${e}"
  style="width:0"></div></div><span id="p-${e}">0%</span></div>`).join("");

let faces = [], smooth = null, fps = 0;

function largest(list) { return list.reduce((a, b) => (a && a.box[2]*a.box[3] > b.box[2]*b.box[3]) ? a : b, null); }

// Exponential smoothing of the main face so the numbers don't flicker frame to frame.
function updatePanel(f) {
  if (!f) { smooth = null; ["age","gender","emotion"].forEach(id => document.getElementById(id).textContent = "–"); return; }
  const a = 0.3, pf = f.gender === "Female" ? f.gender_confidence : 1 - f.gender_confidence;
  if (!smooth) smooth = {age: f.age, pf, probs: {...f.emotion_probs}};
  else {
    smooth.age += a * (f.age - smooth.age);
    smooth.pf += a * (pf - smooth.pf);
    for (const e of EMOTIONS) smooth.probs[e] += a * (f.emotion_probs[e] - smooth.probs[e]);
  }
  const top = EMOTIONS.reduce((x, y) => smooth.probs[x] > smooth.probs[y] ? x : y);
  document.getElementById("age").textContent = Math.round(smooth.age) + " yrs";
  document.getElementById("gender").textContent = smooth.pf >= 0.5
    ? `Female ${Math.round(smooth.pf*100)}%` : `Male ${Math.round((1-smooth.pf)*100)}%`;
  document.getElementById("emotion").textContent = top;
  for (const e of EMOTIONS) {
    const p = Math.round(smooth.probs[e] * 100);
    document.getElementById("b-" + e).style.width = p + "%";
    document.getElementById("b-" + e).style.background = COLORS[e];
    document.getElementById("p-" + e).textContent = p + "%";
  }
}

function draw() {
  const W = view.width, H = view.height;
  ctx.save(); ctx.translate(W, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, W, H); ctx.restore(); // mirror
  const sx = W / grab.width, sy = H / grab.height;
  ctx.font = "bold 16px system-ui"; ctx.lineWidth = 3;
  for (const f of faces) {
    const [x, y, w, h] = f.box, X = W - (x + w) * sx, Y = y * sy, bw = w * sx, bh = h * sy;
    const c = COLORS[f.emotion] || "#4f8cff";
    ctx.strokeStyle = c; ctx.strokeRect(X, Y, bw, bh);
    const lines = [`${f.gender} ${Math.round(f.gender_confidence*100)}%, ${Math.round(f.age)} yrs`,
                   `${f.emotion} ${Math.round(f.emotion_confidence*100)}%`];
    lines.forEach((t, i) => {
      const ty = Math.max(0, Y - 46) + i * 22, tw = ctx.measureText(t).width + 10;
      ctx.fillStyle = c; ctx.fillRect(X, ty, tw, 21); ctx.fillStyle = "#000"; ctx.fillText(t, X + 5, ty + 16);
    });
  }
  ctx.fillStyle = "#4f8"; ctx.font = "14px system-ui"; ctx.fillText(`${fps.toFixed(1)} predictions/s`, 10, 20);
  requestAnimationFrame(draw);
}

async function loop() {
  let last = performance.now();
  while (true) {
    gctx.drawImage(video, 0, 0, grab.width, grab.height);
    const blob = await new Promise(r => grab.toBlob(r, "image/jpeg", 0.85));
    try {
      const res = await fetch("/predict", {method: "POST", body: blob});
      faces = await res.json();
      updatePanel(largest(faces));
      statusEl.textContent = faces.length ? `${faces.length} face(s) detected` : "No face detected – look at the camera";
    } catch (e) { statusEl.textContent = "Server error: " + e; await new Promise(r => setTimeout(r, 1000)); }
    const now = performance.now(); fps = 0.8 * fps + 0.2 * (1000 / (now - last)); last = now;
  }
}

document.getElementById("start").onclick = async () => {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
    await video.play();
  } catch (e) { statusEl.textContent = "Cannot open webcam: " + e.message; return; }
  const w = video.videoWidth || 640, h = video.videoHeight || 480;
  grab.width = 640; grab.height = Math.round(640 * h / w);
  view.width = 960; view.height = Math.round(960 * h / w);
  document.getElementById("start").style.display = "none";
  requestAnimationFrame(draw); loop();
};
</script></body></html>
""".replace("__EMOTIONS__", json.dumps(config.EMOTION_LABELS))


class Handler(BaseHTTPRequestHandler):
    analyzer = None
    lock = threading.Lock()

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, b"not found", "text/plain")
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._send(400, b"[]", "application/json")
        with self.lock:  # one prediction at a time; the models are shared
            results = self.analyzer.analyze(frame)
        self._send(200, json.dumps(results).encode(), "application/json")

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = reachable only through SSH/VS Code forwarding")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for TensorFlow (shared server: keep small)")
    parser.add_argument("--no-tta", action="store_true", help="faster, slightly less accurate")
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    Handler.analyzer = FaceAnalyzer(tta=not args.no_tta)
    dummy = np.zeros((160, 160, 3), np.uint8)
    for n in (1, 2):  # warm-up so the first real frame isn't slow
        Handler.analyzer.predict_crops([dummy] * n, [dummy] * n)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Ready: open http://localhost:{args.port} in your browser (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
