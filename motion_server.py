#!/usr/bin/env python3
"""Text to motion over HTTP: words in, a few seconds of whole-body motion out.

    POST   /v1/motion/generations                 start one (JSON); answers {id, status, ...}
    GET    /v1/motion/generations                 the ones kept
    GET    /v1/motion/generations/{id}            status, progress, error
    GET    /v1/motion/generations/{id}/content    ?format=json (default) | glb | bvh
    DELETE /v1/motion/generations/{id}            cancel it or drop it
    GET    /health                                200 once the model is loaded

hemera starts this like any engine and passes these requests through unchanged. Prompts are never
written anywhere (the log has their length); results live in memory only, and are gone after
DELETE, after --keep-seconds, or when the server stops. Nothing is fetched or sent: every model
file is local and the Hugging Face libraries are kept offline.
"""
import argparse
import json
import os
import queue
import random
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
             "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false", "HSA_TOOLS_DISABLE_REGISTER": "1",
             # AMD's memory-efficient attention: 9 s -> 5.6 s per clip on the Radeon 8060S
             "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}.items():
    os.environ.setdefault(k, v)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exports  # noqa: E402

BASE = "/v1/motion/generations"
FORMATS = {"json": ("application/json", "json"), "glb": ("model/gltf-binary", "glb"),
           "bvh": ("text/plain; charset=utf-8", "bvh")}
MAX_PROMPT_CHARS = 1000
MAX_WAITING = 32


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


class Fail(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Job:
    def __init__(self, req):
        self.id = "motion_" + secrets.token_hex(8)
        self.created_at = int(time.time())
        self.completed_at = 0
        self.status = "queued"
        self.progress = 0
        self.error = None
        self.cancel = False
        self.clip = None
        self.timing = None
        self.truncated = False
        self.prompt = req["prompt"]
        self.duration = req["duration"]
        self.fps = req["fps"]
        self.loop = req["loop"]
        self.seed = req["seed"]


class Server:
    def __init__(self, args):
        self.args = args
        self.name = args.name
        self.model = None
        self.load_error = None
        self.jobs = {}
        self.lock = threading.Lock()
        self.work = queue.Queue()

    # ---- the model, loaded once in the background so /health can answer meanwhile ------------------

    def load(self):
        try:
            from hymotion_engine import HYMotion
            t0 = time.time()
            self.model = HYMotion(self.args.code, self.args.models, device=self.args.device, log=log,
                                  precision=self.args.precision)
            log("ready in %.1f s" % (time.time() - t0))
            threading.Thread(target=self.worker, name="motion-worker", daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.load_error = "%s: %s" % (type(e).__name__, e)
            log("load failed: " + self.load_error)

    def worker(self):
        from hymotion_engine import Cancelled
        while True:
            job = self.work.get()
            if job.cancel:
                continue
            job.status = "in_progress"
            t0 = time.time()

            def progress(i, n):
                job.progress = min(95, int(5 + 90 * i / n))

            try:
                clip, timing, truncated = self.model.make(job.prompt, job.duration, job.fps, job.loop, job.seed,
                                                          self.name, progress, lambda: job.cancel)
                job.clip, job.truncated = clip, truncated
                job.timing = dict(timing, total=int((time.time() - t0) * 1000))
                job.status, job.progress = "completed", 100
                log("%s completed: %d frames in %.1f s (text %d ms, sampling %d ms, decode %d ms)" % (
                    job.id, clip.frames, time.time() - t0, timing["text"], timing["sample"], timing["decode"]))
            except Cancelled:
                job.status = "cancelled"
                log("%s cancelled after %.1f s" % (job.id, time.time() - t0))
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "failed", "%s: %s" % (type(e).__name__, e)
                log("%s failed: %s" % (job.id, job.error))
            finally:
                job.prompt = None  # not kept once the motion is made
                job.completed_at = int(time.time())

    # ---- jobs --------------------------------------------------------------------------------------

    def purge(self):
        cutoff = time.time() - self.args.keep_seconds
        with self.lock:
            done = sorted((j for j in self.jobs.values() if j.completed_at), key=lambda j: -j.completed_at)
            for i, j in enumerate(done):
                if i >= self.args.keep_max or j.completed_at < cutoff:
                    del self.jobs[j.id]

    def describe(self, job):
        d = {"id": job.id, "object": "motion.generation", "model": self.name, "status": job.status,
             "progress": job.progress, "created_at": job.created_at, "duration": job.duration, "fps": job.fps,
             "frames": int(round(job.duration * job.fps)), "loop": job.loop, "seed": job.seed}
        if job.completed_at:
            d["completed_at"] = job.completed_at
        if job.timing:
            d["timing_ms"] = job.timing
        if job.truncated:
            d["prompt_truncated"] = True
        if job.clip is not None:
            d["output"] = {"formats": list(FORMATS), "url": BASE + "/" + job.id + "/content"}
        if job.error:
            d["error"] = {"code": job.status, "message": job.error}
        return d

    @staticmethod
    def parse(body):
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            raise Fail(400, "the body is not JSON")
        if not isinstance(req, dict):
            raise Fail(400, "the body must be a JSON object")
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Fail(400, "'prompt' is missing: say in words how the person moves")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise Fail(400, "'prompt' is longer than %d characters; the model reads about 60 words" % MAX_PROMPT_CHARS)
        duration = req.get("duration", 4)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 1 <= duration <= 10:
            raise Fail(400, "'duration' must be a number of seconds from 1 to 10")
        fps = req.get("fps", 30)
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 120:
            raise Fail(400, "'fps' must be a whole number from 1 to 120")
        loop = req.get("loop", False)
        if not isinstance(loop, bool):
            raise Fail(400, "'loop' must be true or false")
        seed = req.get("seed")
        if seed is None:
            seed = random.randrange(2 ** 31)
        elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
            raise Fail(400, "'seed' must be a whole number from 0 to 4294967295")
        if req.get("constraints") not in (None, [], {}):
            raise Fail(400, "this model takes no constraints: describe the motion in words")
        return {"prompt": prompt.strip(), "duration": float(duration), "fps": fps, "loop": loop, "seed": seed}


def handler_for(srv):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "motion-server"

        def log_message(self, *_a):  # no access log: paths are enough to find jobs, and nothing else is needed
            pass

        def send(self, status, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def fail(self, status, message, extra=None):
            self.send(status, {"error": message}, extra=extra)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 64 * 1024:
                raise Fail(413, "the request is larger than 64 KB")
            return self.rfile.read(n) if n else b""

        def job(self, jid):
            srv.purge()
            with srv.lock:
                job = srv.jobs.get(jid)
            if job is None:
                raise Fail(404, "no motion generation '%s': finished ones are kept for %d minutes, or until "
                                "deleted or the model is stopped" % (jid, srv.args.keep_seconds // 60))
            return job

        def route(self):
            url = urlparse(self.path)
            path = url.path.rstrip("/")
            if path == "/health":
                if srv.model is not None:
                    return self.send(200, {"status": "ok"})
                if srv.load_error:
                    return self.send(500, {"status": "failed", "error": srv.load_error})
                return self.send(503, {"status": "loading"})
            if not path.startswith(BASE):
                raise Fail(404, "not found")
            rest = path[len(BASE):].lstrip("/")
            if self.command == "POST" and not rest:
                req = srv.parse(self.body())
                if srv.model is None:
                    raise Fail(503, "the motion model is still loading; try again in a few seconds")
                srv.purge()
                job = Job(req)
                with srv.lock:
                    waiting = sum(1 for j in srv.jobs.values() if j.status == "queued")
                    if waiting >= MAX_WAITING:
                        raise Fail(429, "%d motions are already waiting; try again later" % waiting)
                    srv.jobs[job.id] = job
                srv.work.put(job)
                log("%s queued: prompt %d characters, %.1f s at %d fps%s" % (
                    job.id, len(req["prompt"]), job.duration, job.fps, ", loop" if job.loop else ""))
                return self.send(200, srv.describe(job))
            if self.command == "GET" and not rest:
                srv.purge()
                with srv.lock:
                    jobs = sorted(srv.jobs.values(), key=lambda j: -j.created_at)
                return self.send(200, {"object": "list", "data": [srv.describe(j) for j in jobs]})
            parts = rest.split("/")
            if self.command == "GET" and len(parts) == 1:
                return self.send(200, srv.describe(self.job(parts[0])))
            if self.command == "GET" and len(parts) == 2 and parts[1] == "content":
                job = self.job(parts[0])
                fmt = (parse_qs(url.query).get("format") or ["json"])[0].lower()
                if fmt not in FORMATS:
                    raise Fail(400, "format '%s' is not offered: json, glb or bvh" % fmt)
                if job.clip is None:
                    raise Fail(409, "motion generation '%s' is %s%s" % (
                        job.id, job.status, ": " + job.error if job.error else " — nothing to fetch yet"))
                ctype, ext = FORMATS[fmt]
                data = getattr(exports, "to_" + fmt)(job.clip)
                return self.send(200, data, ctype, {"Content-Disposition": 'attachment; filename="%s.%s"' % (job.id, ext)})
            if self.command == "DELETE" and len(parts) == 1:
                job = self.job(parts[0])
                in_flight = not job.completed_at
                job.cancel = True
                with srv.lock:
                    srv.jobs.pop(job.id, None)
                job.clip = None
                log("%s deleted%s" % (job.id, " while " + job.status if in_flight else ""))
                return self.send(200, {"id": job.id, "object": "motion.generation.deleted", "deleted": True,
                                       "cancelled": in_flight})
            raise Fail(405 if path.startswith(BASE) else 404, "%s %s is not offered" % (self.command, url.path))

        def handle_any(self):
            try:
                self.route()
            except Fail as f:
                self.fail(f.status, str(f))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                log("error: %s: %s" % (type(e).__name__, e))
                self.fail(500, "%s: %s" % (type(e).__name__, e))

        do_GET = do_POST = do_DELETE = handle_any

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--models", required=True, help="directory holding HY-Motion-1.0, Qwen3-8B, clip-vit-large-patch14")
    ap.add_argument("--code", required=True, help="Tencent's HY-Motion-1.0 code")
    ap.add_argument("--name", default="hy-motion-1.0", help="the model name answers carry")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"], help="the network's precision")
    ap.add_argument("--keep-seconds", type=int, default=3600, help="how long a finished motion is kept")
    ap.add_argument("--keep-max", type=int, default=64, help="most finished motions kept")
    args = ap.parse_args()
    srv = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), handler_for(srv))
    httpd.daemon_threads = True
    log("listening on %s:%d, loading %s" % (args.host, args.port, args.name))
    threading.Thread(target=srv.load, name="motion-load", daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
