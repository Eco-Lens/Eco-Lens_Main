"""
ui/server.py — Run-aware FastAPI server with run isolation.
"""
import sys, os, json, asyncio, time, glob, shutil, re, hashlib, traceback, html
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pipeline_core.context import RunContext
from pipeline_core.config import STEPS, TOTAL_WEIGHT
from pipeline_core.events import PipelineEvent, step_started, step_completed, log_msg

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
server_state = {"runs": {}, "current_run_id": None}
state_lock = asyncio.Lock()
MAX_FILE_SIZE = 50 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
RETENTION_DAYS = 7
SSE_HEARTBEAT_SECONDS = 8

STEP_ICONS = {s["id"]: s.get("icon", "\U0001f4c4") for s in STEPS}
STEP_DESCRIPTIONS = {
    "convert": "Render PDF pages to images",
    "ocr": "Extract text with PaddleOCR",
    "layout": "Classify words via LayoutLMv3",
    "tables": "Extract tables via TATR",
    "blocks": "Group words into document blocks",
    "scope": "Classify ESG scope via ClimateBERT",
    "visualize": "Generate HTML, CSV, overlay images",
}

def new_run_state(run_id):
    return {
        "run_id": run_id, "status": "idle", "current_step_id": None,
        "current_step_index": -1, "current_step": "", "progress_pct": 0,
        "error": None, "started_at": None, "elapsed_seconds": 0,
        "estimated_remaining": None, "total_pages": 0, "current_page": 0,
        "document_name": None, "document_size": 0, "output_html": None,
        "output_csv": None, "events": [], "event_id_last": 0,
        "results_ready": False, "task": None,
    }

async def get_run(run_id):
    async with state_lock:
        return server_state["runs"].get(run_id)

async def get_run_or_404(run_id):
    s = await get_run(run_id)
    if s is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return s

async def upd_run(run_id, **kw):
    async with state_lock:
        s = server_state["runs"].setdefault(run_id, new_run_state(run_id))
        s.update(kw)
        server_state["current_run_id"] = run_id

async def emit(run_id, ev):
    async with state_lock:
        s = server_state["runs"].get(run_id)
        if not s: return
        ev.event_id = s["event_id_last"]
        s["event_id_last"] += 1
        ev.run_id = run_id
        s.setdefault("events", []).append(ev.to_dict())
        if len(s["events"]) > 500:
            s["events"] = s["events"][-500:]

async def current_run_id():
    async with state_lock:
        rid = server_state.get("current_run_id")
        if rid and rid in server_state["runs"]: return rid
        return None

async def run_subprocess(ctx, cmd_list, step_index, run_id):
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["RUN_ROOT"] = str(ctx.run_root)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    log_path = ctx.logs_dir / f"step{step_index}.log"
    os.makedirs(ctx.logs_dir, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *cmd_list, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(ctx.project_root), env=env)
    lines = []
    async for line in proc.stdout:
        text = line.decode("utf-8", errors="replace").rstrip()
        if text: lines.append(text)
    rc = await proc.wait()
    with open(str(log_path), "w", encoding="utf-8") as lf:
        lf.write("\n".join(lines))
    tail = "\n".join(lines[-100:])
    step_id = STEPS[step_index]["id"] if step_index < len(STEPS) else f"step{step_index}"
    error_details = {
        "step_id": step_id, "return_code": rc,
        "command": " ".join(str(c) for c in cmd_list)[:500],
        "log_url": f"runs/{run_id}/logs/step{step_index}.log",
        "stdout_tail": tail[-3000:] if tail else "",
        "timestamp": time.time(),
    }
    return rc, lines, error_details

def build_step_status(s, run_id):
    current_idx = s.get("current_step_index", -1)
    run_status = s.get("status", "idle")
    steps = []
    for i, st in enumerate(STEPS):
        if run_status == "completed":
            step_status = "completed"
        elif run_status == "failed":
            step_status = "failed" if i == current_idx else ("completed" if i < current_idx else "queued")
        elif run_status == "cancelled":
            step_status = "cancelled" if i >= current_idx else "completed"
        elif i < current_idx:
            step_status = "completed"
        elif i == current_idx:
            step_status = "running" if run_status == "running" else "queued"
        else:
            step_status = "queued"
        steps.append({
            "id": st["id"], "name": st["name"], "icon": STEP_ICONS.get(st["id"], "\U0001f4c4"),
            "weight": st["weight"], "description": STEP_DESCRIPTIONS.get(st["id"], ""),
            "status": step_status, "active": i == current_idx and run_status == "running",
            "done": step_status == "completed", "pending": step_status == "queued",
            "log_step_index": i - 1 if i > 0 else -1,
        })
    return steps

def run_output_path(run_id, *parts):
    return os.path.join(BASE, "runs", run_id, "output", *parts)

def rel_output_path(run_id, *parts):
    return f"runs/{run_id}/output/" + "/".join(parts)

def read_json_file(path):
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {os.path.basename(path)}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def h(value):
    return html.escape(str(value if value is not None else ""))

def report_shell(title, subtitle, cards_html, body_html):
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{h(title)}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f4;margin:0;color:#1f2937}}
.header{{background:#1a1a2e;color:white;padding:24px 34px;margin-bottom:24px}}
.header h1{{font-size:26px;margin:0 0 6px;font-weight:650}}.sub{{color:#b8bfcc;font-size:13px}}
.wrap{{padding:0 28px 32px}}.stats{{display:flex;gap:14px;margin-bottom:22px;flex-wrap:wrap}}
.card{{background:white;border-radius:11px;padding:18px 22px;box-shadow:0 1px 5px rgba(0,0,0,.08);min-width:150px}}
.num{{font-size:30px;font-weight:750;color:#1a1a2e;line-height:1}}.lbl{{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.8px;margin-top:6px}}
.page-card{{background:white;border-radius:11px;margin-bottom:16px;box-shadow:0 1px 5px rgba(0,0,0,.08);overflow:hidden}}
.page-h{{padding:14px 18px;background:#f8fafc;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;gap:12px;align-items:center}}
.page-title{{font-weight:650;color:#111827}}.page-meta{{font-size:12px;color:#6b7280}}
.page-b{{padding:16px 18px}}.pills{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}}
.pill{{display:inline-flex;gap:6px;align-items:center;padding:5px 9px;background:#eef2ff;color:#273b8f;border-radius:999px;font-size:12px}}
.img-row{{display:flex;gap:10px;overflow-x:auto;margin-bottom:14px;padding:10px;background:#f8fafc;border-radius:8px}}
.img-row img{{max-height:520px;border-radius:7px;box-shadow:0 1px 5px rgba(0,0,0,.12);border:1px solid #e5e7eb;background:white}}
table{{width:100%;border-collapse:collapse;font-size:12px;background:white}}th{{background:#1a1a2e;color:white;text-align:left;padding:8px 10px;font-size:11px;position:sticky;top:0}}td{{border:1px solid #e5e7eb;padding:8px 10px;vertical-align:top}}tr:hover td{{background:#f8fafc}}
.muted{{color:#6b7280}}.tag{{font-size:11px;padding:2px 6px;border-radius:4px;background:#ecfdf5;color:#047857;font-weight:600}}.bad{{background:#fef2f2;color:#b91c1c}}
</style></head><body><div class='header'><h1>{h(title)}</h1><div class='sub'>{h(subtitle)}</div></div><div class='wrap'><div class='stats'>{cards_html}</div>{body_html}</div></body></html>"""

def stat_card(num, label):
    return f"<div class='card'><div class='num'>{h(num)}</div><div class='lbl'>{h(label)}</div></div>"

def image_row(src, label):
    return f"<div><img src='{h(src)}'><div class='muted' style='font-size:11px;text-align:center;margin-top:4px'>{h(label)}</div></div>"

def html_file_with_base(path, base_href):
    if not os.path.exists(path):
        raise HTTPException(404)
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    base_tag = f"<base href='{h(base_href)}'>"
    if "<base " not in content.lower():
        if "<head>" in content:
            content = content.replace("<head>", "<head>" + base_tag, 1)
        elif "<head" in content.lower():
            content = re.sub(r"(<head[^>]*>)", r"\1" + base_tag, content, count=1, flags=re.I)
        else:
            content = base_tag + content
    return HTMLResponse(content)

async def pipeline_worker(ctx, run_id):
    try:
        await upd_run(run_id, status="running", started_at=time.time())
        await emit(run_id, step_started("pipeline"))
        ctx.ensure_dirs()
        ctx.clean_pages()

        st0 = STEPS[0]
        await upd_run(run_id, current_step=st0["name"], current_step_index=0,
                      current_step_id=st0["id"], progress_pct=0)
        await emit(run_id, log_msg("info", "Rendering PDF pages...", step=st0["id"]))
        input_pdf = ctx.resolve_input_pdf()
        with fitz.open(str(input_pdf)) as doc:
            npages = len(doc)
            await upd_run(run_id, total_pages=npages)
            for i in range(npages):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=200)
                pix.save(str(ctx.pages_dir / f"page_{i+1:03d}.jpg"))
                if (i+1) % 3 == 0 or i == npages-1:
                    await emit(run_id, log_msg("info", f"Page {i+1}/{npages}", step=st0["id"]))
                    await upd_run(run_id, current_page=i+1)
        await emit(run_id, step_completed(st0["id"], duration_seconds=0))

        step_scripts = [
            [sys.executable, "1.OCR_step/run_ocr_simple.py",
             "--image-dir", str(ctx.pages_dir), "--out-json", str(ctx.ocr_json)],
            [sys.executable, "2.LayoutLMV3_step/inference_layoutlmv3.py",
             "--ocr-json", str(ctx.ocr_json), "--image-dir", str(ctx.pages_dir),
             "--out-labels", str(ctx.layout_labels_json),
             "--out-layout", str(ctx.layout_words_json), "--run-id", run_id],
            [sys.executable, "3.Table_Understanding/run.py",
             "--ocr-json", str(ctx.ocr_json), "--labels-json", str(ctx.layout_labels_json),
             "--image-dir", str(ctx.pages_dir), "--out-dir", str(ctx.step3_table_dir), "--run-id", run_id],
            [sys.executable, "4.SemanticMapping/build_layout_blocks.py",
             "--layout-json", str(ctx.layout_words_json), "--image-dir", str(ctx.pages_dir),
             "--out-json", str(ctx.layout_blocks_json), "--run-id", run_id],
            [sys.executable, "4.SemanticMapping/run_scope_inference.py",
             "--layout-json", str(ctx.layout_blocks_json), "--tables-json", str(ctx.all_tables_json),
             "--image-dir", str(ctx.pages_dir), "--out-dir", str(ctx.step4_semantic_dir), "--run-id", run_id],
            [sys.executable, "4.SemanticMapping/visualize_results.py",
             "--unified-json", str(ctx.unified_json), "--layout-json", str(ctx.layout_blocks_json),
             "--tables-json", str(ctx.all_tables_json), "--image-dir", str(ctx.pages_dir),
             "--out-dir", str(ctx.step5_viz_dir), "--run-id", run_id],
        ]

        for script_idx, cmd in enumerate(step_scripts):
            st_idx = script_idx + 1
            st = STEPS[st_idx]
            await upd_run(run_id, current_step=st["name"], current_step_index=st_idx,
                          current_step_id=st["id"])
            await emit(run_id, step_started(st["id"]))
            t0 = time.time()
            rc, lines, err_detail = await run_subprocess(ctx, cmd, st_idx, run_id)
            elapsed = time.time() - t0
            if rc != 0:
                emsg = f"{st['name']} failed (rc={rc})"
                # Extract actual error from subprocess output
                err_lines = [l for l in lines[-10:] if l.strip() and not l.startswith("  ") and not l.startswith("C:")]
                if err_lines:
                    emsg = f"{st['name']} failed: {err_lines[-1][:300]}"
                err_detail["summary"] = emsg
                await emit(run_id, log_msg("error", emsg))
                # Emit last output lines for debugging
                for l in lines[-5:]:
                    if l.strip():
                        await emit(run_id, log_msg("info", f"  {l[:300]}"))
                cw_fail = sum(s["weight"] for i, s in enumerate(STEPS) if i < st_idx)
                fail_pct = min(int(cw_fail / TOTAL_WEIGHT * 100), 100)
                await upd_run(run_id, status="failed", error=err_detail,
                              current_step_index=st_idx, current_step_id=st["id"],
                              current_step=st["name"], progress_pct=fail_pct)
                await emit(run_id, step_completed(st["id"], duration_seconds=round(elapsed, 1)))
                return
            await emit(run_id, step_completed(st["id"], duration_seconds=round(elapsed, 1)))
            cw = sum(s["weight"] for i, s in enumerate(STEPS) if i <= st_idx)
            pct = min(int(cw / TOTAL_WEIGHT * 100), 100)
            await upd_run(run_id, progress_pct=pct)

        await upd_run(run_id, progress_pct=100, status="completed", results_ready=True,
                      output_html=f"runs/{run_id}/output/step5_visualization/index.html",
                      output_csv=f"runs/{run_id}/output/step5_visualization/scope_predictions_all.csv")
        await emit(run_id, step_completed("pipeline"))
        await emit(run_id, log_msg("info", "Pipeline complete: " + str(ctx.viz_index_html)))
    except asyncio.CancelledError:
        await emit(run_id, log_msg("info", "Pipeline cancelled"))
        await upd_run(run_id, status="cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        s_entry = server_state["runs"].get(run_id, {})
        err = {
            "step_id": s_entry.get("current_step_id", run_id),
            "return_code": -1, "summary": str(e), "details": tb[:2000],
            "log_url": None, "timestamp": time.time(),
        }
        await emit(run_id, log_msg("error", f"Pipeline exception: {e}"))
        await upd_run(run_id, status="failed", error=err)


app = FastAPI(title="Eco-Lens Pipeline UI", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Not found</h1>")

@app.get("/api/config")
async def get_config():
    return {
        "max_upload_bytes": MAX_FILE_SIZE,
        "steps": [{"id": s["id"], "name": s["name"], "weight": s["weight"],
                   "icon": STEP_ICONS.get(s["id"], "\U0001f4c4"),
                   "description": STEP_DESCRIPTIONS.get(s["id"], "")} for s in STEPS],
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE // 1048576} MB upload limit")
    if not content.startswith(PDF_MAGIC):
        raise HTTPException(400, "File is not a valid PDF")
    run_id = f"run_{int(time.time())}_{hashlib.md5(content[:1024]).hexdigest()[:8]}"
    safe = "".join(c for c in file.filename if c.isalnum() or c in "._- ")[:100]
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    ctx = RunContext(run_id, BASE, os.path.join(BASE, "runs", run_id))
    ctx.ensure_dirs()
    ctx.save_pdf(content, safe)
    async with state_lock:
        s = new_run_state(run_id)
        s.update(status="uploaded", document_name=safe, document_size=len(content))
        server_state["runs"][run_id] = s
        server_state["current_run_id"] = run_id
    return {"status": "ok", "run_id": run_id, "filename": safe,
            "pdf_path": str(ctx.input_pdf), "size_kb": round(len(content) / 1024, 1)}

@app.post("/api/runs/{run_id}/start")
async def start_run(run_id: str):
    s = await get_run_or_404(run_id)
    if not s.get("document_name"):
        raise HTTPException(400, "No PDF uploaded for this run")
    if s["status"] == "running":
        raise HTTPException(409, "Already running")
    ctx = RunContext(run_id, BASE, os.path.join(BASE, "runs", run_id))
    task = asyncio.create_task(pipeline_worker(ctx, run_id))
    async with state_lock:
        s2 = server_state["runs"].get(run_id, {})
        s2["task"] = task
    await upd_run(run_id, status="starting")
    return {"status": "started", "run_id": run_id}

@app.post("/api/runs/{run_id}/retry")
async def retry_run(run_id: str):
    s = await get_run_or_404(run_id)
    if s["status"] == "running":
        raise HTTPException(409, "Already running")
    if s["status"] not in ("failed", "cancelled"):
        raise HTTPException(400, f"Cannot retry run with status '{s['status']}'")
    ctx = RunContext(run_id, BASE, os.path.join(BASE, "runs", run_id))
    async with state_lock:
        s2 = server_state["runs"].get(run_id, {})
        s2["error"] = None; s2["events"] = []; s2["status"] = "starting"
        s2["progress_pct"] = 0; s2["current_step_index"] = -1
        s2["current_step_id"] = None; s2["current_step"] = ""
    task = asyncio.create_task(pipeline_worker(ctx, run_id))
    async with state_lock:
        s3 = server_state["runs"].get(run_id, {})
        s3["task"] = task
    return {"status": "started", "run_id": run_id}

@app.get("/api/runs/{run_id}/status")
async def get_run_status(run_id: str):
    s = await get_run_or_404(run_id)
    err = s.get("error")
    error_out = None
    if err and isinstance(err, dict):
        error_out = {
            "step_id": err.get("step_id"), "return_code": err.get("return_code"),
            "summary": err.get("summary", str(err)), "details": (err.get("details") or "")[:500],
            "log_url": err.get("log_url"),
        }
    elif err and isinstance(err, str):
        error_out = {"summary": err}
    now = time.time()
    started = s.get("started_at")
    running = s.get("status") == "running"
    elapsed = round(now - started) if started and running else s.get("elapsed_seconds", 0)
    pct = s.get("progress_pct", 0)
    remaining = round((elapsed / max(pct, 1)) * (100 - pct)) if pct > 5 and started and running else None
    return {
        "run_id": run_id, "status": s.get("status", "idle"),
        "current_step_id": s.get("current_step_id"),
        "current_step_index": s.get("current_step_index", -1),
        "current_step": s.get("current_step", ""),
        "progress_pct": pct, "error": error_out,
        "estimated_remaining_seconds": remaining,
        "results_ready": s.get("results_ready", False),
        "output_html": s.get("output_html"), "output_csv": s.get("output_csv"),
        "total_pages": s.get("total_pages", 0), "current_page": s.get("current_page", 0),
        "document_name": s.get("document_name"), "document_size": s.get("document_size"),
        "elapsed_seconds": elapsed,
        "steps": build_step_status(s, run_id),
    }

@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, after: int = -1):
    captured = run_id
    async def gen():
        last_eid = after
        while True:
            async with state_lock:
                s = server_state["runs"].get(captured, {})
                if not s.get("run_id") or s.get("run_id") != captured:
                    yield "data: {\"status\":\"closed\"}\n\n"; return
                events = s.get("events", [])
                new = [e for e in events if e.get("event_id", -1) > last_eid]
                if new:
                    last_eid = max(e.get("event_id") for e in new)
                status = s.get("status", "idle")
                done = status in ("completed", "failed", "cancelled")
                steps = build_step_status(s, captured)
                now = time.time()
                started = s.get("started_at")
                elapsed = round(now - started) if started and status == "running" else s.get("elapsed_seconds", 0)
                pct = s.get("progress_pct", 0)
                remaining = round((elapsed / max(pct, 1)) * (100 - pct)) if pct > 5 and started and status == "running" else None
                err = s.get("error")
                error_out = None
                if err and isinstance(err, dict):
                    error_out = {
                        "step_id": err.get("step_id"), "return_code": err.get("return_code"),
                        "summary": err.get("summary", str(err)), "details": (err.get("details") or "")[:500],
                        "log_url": err.get("log_url"),
                    }
                elif err and isinstance(err, str):
                    error_out = {"summary": err}
            if new:
                latest_id = max(ev.get("event_id", 0) for ev in new)
                payload = {
                    "events": new, "status": status, "done": done, "steps": steps,
                    "progress_pct": s.get("progress_pct", 0),
                    "current_step": s.get("current_step", ""),
                    "current_step_index": s.get("current_step_index", -1),
                    "current_step_id": s.get("current_step_id"),
                    "current_page": s.get("current_page", 0),
                    "total_pages": s.get("total_pages", 0),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                    "error": error_out,
                    "output_html": s.get("output_html"),
                    "output_csv": s.get("output_csv"),
                    "results_ready": s.get("results_ready", False),
                    "document_name": s.get("document_name"),
                    "document_size": s.get("document_size"),
                }
                yield f"id: {latest_id}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                yield ": heartbeat\n\n"
            if done:
                break
            await asyncio.sleep(SSE_HEARTBEAT_SECONDS if not new else 0.3)
    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

@app.get("/api/runs/{run_id}/step-logs/{step_index}")
async def get_step_logs(run_id: str, step_index: int):
    log_path = os.path.join(BASE, "runs", run_id, "logs", f"step{step_index}.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        return {"logs": [{"ts": "", "msg": l, "step_index": step_index} for l in lines if l.strip()],
                "step_index": step_index}
    return {"logs": [], "step_index": step_index}

@app.get("/api/runs/{run_id}/results/summary")
async def get_results_summary(run_id: str):
    upath = run_output_path(run_id, "step4_semantic_mapping", "all_results_unified.json")
    if not os.path.exists(upath):
        raise HTTPException(404, "No results found for this run")
    with open(upath, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    total_blocks = 0; total_tables = 0
    scope_counts = {}; scope_blocks = {"Scope 1": [], "Scope 2": [], "Scope 3": [], "Other": [], "Mixed": []}
    total_pages = 0
    for page, pd in data.items():
        if page == "_meta": continue
        total_pages += 1
        for b in pd.get("text_blocks", []):
            total_blocks += 1
            s = b.get("scope", "Other")
            scope_counts[s] = scope_counts.get(s, 0) + 1
            if s in scope_blocks:
                scope_blocks[s].append({
                    "page": page, "block_id": b.get("block_id", ""),
                    "type": b.get("type", ""), "text": (b.get("text", "") or "")[:200],
                    "confidence": round(b.get("confidence", 0), 3),
                    "value": b.get("value"), "unit": b.get("unit"),
                })
        for t in pd.get("tables", []):
            total_tables += 1
            s = t.get("scope", "Other")
            scope_counts[s] = scope_counts.get(s, 0) + 1
            if s in scope_blocks:
                scope_blocks[s].append({
                    "page": page, "type": "table", "table_id": t.get("table_id", ""),
                    "text": f"Table {t.get('table_id','')} ({t.get('rows','?')}r x {t.get('cols','?')}c)",
                    "confidence": round(t.get("confidence", 0), 3),
                })
    return {"total_pages": total_pages, "total_blocks": total_blocks, "total_tables": total_tables,
            "scope_counts": scope_counts, "scope_blocks": scope_blocks,
            "output_html": rel_output_path(run_id, "step5_visualization", "index.html"),
            "output_csv": rel_output_path(run_id, "step5_visualization", "scope_predictions_all.csv")}

@app.get("/api/runs/{run_id}/results/report")
async def get_results_report(run_id: str):
    report_path = run_output_path(run_id, "step5_visualization", "index.html")
    if os.path.exists(report_path):
        return html_file_with_base(report_path, f"/runs/{run_id}/output/step5_visualization/")
    # Fallback: render a minimal report from unified JSON so the Report action never opens a raw 404 page.
    summary = await get_results_summary(run_id)
    rows = []
    for scope, blocks in summary.get("scope_blocks", {}).items():
        for b in blocks:
            rows.append(
                f"<tr><td>{scope}</td><td>{b.get('page','')}</td><td>{b.get('type','')}</td>"
                f"<td>{(b.get('text') or '').replace('<','&lt;').replace('>','&gt;')}</td>"
                f"<td>{b.get('confidence','')}</td></tr>"
            )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Eco-Lens Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;background:#f5f7fa;color:#1f2937}}.cards{{display:flex;gap:12px;margin:18px 0}}.card{{background:white;border-radius:10px;padding:16px 22px;box-shadow:0 1px 4px #0001}}.n{{font-size:28px;font-weight:700}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}}th{{background:#111827;color:white}}</style>
</head><body><h1>Eco-Lens Report</h1><p>Generated fallback report because step5_visualization/index.html was not found.</p>
<div class='cards'><div class='card'><div class='n'>{summary['total_pages']}</div><div>Pages</div></div><div class='card'><div class='n'>{summary['total_blocks']}</div><div>Blocks</div></div><div class='card'><div class='n'>{summary['total_tables']}</div><div>Tables</div></div></div>
<table><thead><tr><th>Scope</th><th>Page</th><th>Type</th><th>Text</th><th>Confidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
    return HTMLResponse(html)

@app.get("/api/runs/{run_id}/results/convert/report")
async def get_convert_report(run_id: str):
    page_dir = os.path.join(BASE, "runs", run_id, "pages")
    imgs = sorted(glob.glob(os.path.join(page_dir, "*.jpg")))
    if not imgs:
        raise HTTPException(404, "No rendered page images found")
    cards = stat_card(len(imgs), "Rendered Pages") + stat_card("200 DPI", "Image Quality")
    body = "".join(
        f"<div class='page-card'><div class='page-h'><div class='page-title'>{h(os.path.basename(img))}</div><div class='page-meta'>PDF page render</div></div><div class='page-b'><div class='img-row'>{image_row('/runs/'+run_id+'/pages/'+os.path.basename(img), 'Rendered page')}</div></div></div>"
        for img in imgs
    )
    return HTMLResponse(report_shell("PDF Rendering Output", "Rendered PDF pages used by downstream OCR/Layout/Table steps", cards, body))

@app.get("/api/runs/{run_id}/results/ocr/report")
async def get_ocr_report(run_id: str):
    data = read_json_file(run_output_path(run_id, "step1_ocr", "ocr_words.json"))
    pages = data if isinstance(data, dict) else {}
    total_words = sum(len(v or []) for v in pages.values())
    cards = stat_card(len(pages), "Pages") + stat_card(total_words, "OCR Words") + stat_card("PaddleOCR", "Engine")
    body_parts = []
    for page, words in sorted(pages.items()):
        base = os.path.splitext(page)[0]
        image_src = f"/runs/{run_id}/pages/{base}.jpg"
        rows = "".join(f"<tr><td>{h(w.get('text'))}</td><td>{h(round(w.get('conf', 0), 3))}</td><td>{h(w.get('bbox'))}</td></tr>" for w in (words or [])[:180])
        body_parts.append(f"<div class='page-card'><div class='page-h'><div class='page-title'>{h(page)}</div><div class='page-meta'>{len(words or [])} words</div></div><div class='page-b'><div class='img-row'>{image_row(image_src, 'Rendered input page')}</div><table><thead><tr><th>Text</th><th>Confidence</th><th>BBox</th></tr></thead><tbody>{rows}</tbody></table></div></div>")
    return HTMLResponse(report_shell("OCR Output", "PaddleOCR extracted text, confidence, and bounding boxes", cards, "".join(body_parts)))

@app.get("/api/runs/{run_id}/results/blocks/report")
async def get_blocks_report(run_id: str):
    data = read_json_file(run_output_path(run_id, "step4_semantic_mapping", "0_layoutlmv3_layout.json"))
    cards = stat_card(len(data), "Pages") + stat_card(sum((p.get('num_blocks') or len(p.get('blocks', []))) for p in data.values()), "Blocks")
    body_parts = []
    for page, pd in sorted(data.items()):
        summary = pd.get("block_summary", {})
        pills = "".join(f"<span class='pill'>{h(k)} <b>{h(v)}</b></span>" for k, v in sorted(summary.items()))
        rows = "".join(f"<tr><td><span class='tag'>{h(b.get('type'))}</span></td><td>{h(b.get('text'))}</td><td>{h(round(b.get('confidence', 0), 3))}</td><td>{h(b.get('bbox'))}</td></tr>" for b in pd.get("blocks", [])[:160])
        body_parts.append(f"<div class='page-card'><div class='page-h'><div class='page-title'>{h(page)}</div><div class='page-meta'>{h(pd.get('num_blocks', len(pd.get('blocks', []))))} blocks</div></div><div class='page-b'><div class='pills'>{pills}</div><table><thead><tr><th>Type</th><th>Text</th><th>Confidence</th><th>BBox</th></tr></thead><tbody>{rows}</tbody></table></div></div>")
    return HTMLResponse(report_shell("Layout Blocks Output", "Grouped LayoutLMv3 words into semantic document blocks", cards, "".join(body_parts)))

@app.get("/api/runs/{run_id}/results/scope/report")
async def get_scope_report(run_id: str):
    summary = await get_results_summary(run_id)
    cards = stat_card(summary.get("total_pages", 0), "Pages") + stat_card(summary.get("total_blocks", 0), "Blocks") + stat_card(summary.get("total_tables", 0), "Tables")
    body_parts = []
    for scope, blocks in summary.get("scope_blocks", {}).items():
        if not blocks: continue
        rows = "".join(f"<tr><td>{h(b.get('page'))}</td><td>{h(b.get('type'))}</td><td>{h(b.get('text'))}</td><td>{h(b.get('confidence'))}</td></tr>" for b in blocks)
        body_parts.append(f"<div class='page-card'><div class='page-h'><div class='page-title'>{h(scope)}</div><div class='page-meta'>{len(blocks)} items</div></div><div class='page-b'><table><thead><tr><th>Page</th><th>Type</th><th>Text</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table></div></div>")
    return HTMLResponse(report_shell("Scope Classification Output", "ClimateBERT ESG scope classification results", cards, "".join(body_parts)))

@app.get("/api/runs/{run_id}/results/layout")
async def get_layout_results(run_id: str):
    data = read_json_file(run_output_path(run_id, "step2_layoutlmv3", "layout_words.json"))
    pages = []
    for page_name, pd in data.get("pages", {}).items():
        counts = {}
        words = pd.get("words", [])
        for w in words:
            label = w.get("layout_label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        sample = [{"text": w.get("text", ""), "label": w.get("layout_label", ""),
                   "confidence": round(w.get("layout_confidence", 0), 3)} for w in words[:80]]
        pages.append({"page": page_name, "width": pd.get("width"), "height": pd.get("height"),
                      "word_count": len(words), "label_counts": counts, "sample_words": sample})
    return {"model": data.get("model", {}), "total_pages": len(pages), "pages": pages,
            "json_url": rel_output_path(run_id, "step2_layoutlmv3", "layout_words.json")}

@app.get("/api/runs/{run_id}/results/layout/report")
async def get_layout_report(run_id: str):
    d = await get_layout_results(run_id)
    total_words = sum(p.get("word_count", 0) for p in d.get("pages", []))
    model = d.get("model", {})
    cards = stat_card(d.get("total_pages", 0), "Pages") + stat_card(total_words, "Words") + stat_card(model.get("name", "LayoutLMv3"), "Model")
    pages_html = []
    for p in d.get("pages", []):
        counts = "".join(f"<span class='pill'>{h(k)} <b>{h(v)}</b></span>" for k, v in sorted((p.get("label_counts") or {}).items(), key=lambda kv: str(kv[0])))
        rows = "".join(
            f"<tr><td>{h(w.get('text'))}</td><td><span class='tag'>{h(w.get('label'))}</span></td><td>{h(w.get('confidence'))}</td></tr>"
            for w in p.get("sample_words", [])
        )
        pages_html.append(f"<div class='page-card'><div class='page-h'><div class='page-title'>{h(p.get('page'))}</div><div class='page-meta'>{h(p.get('word_count'))} words · {h(p.get('width'))}x{h(p.get('height'))}</div></div><div class='page-b'><div class='pills'>{counts}</div><table><thead><tr><th>Word</th><th>Label</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table></div></div>")
    return HTMLResponse(report_shell("LayoutLMv3 Results", "Word-level layout labels and confidence scores", cards, "".join(pages_html)))

@app.get("/api/runs/{run_id}/results/tables")
async def get_table_results(run_id: str):
    data = read_json_file(run_output_path(run_id, "step3_table_understanding", "all_tables.json"))
    pages = data if isinstance(data, list) else []
    total_tables = 0; total_metrics = 0; total_esg = 0
    rows = []
    for page in pages:
        page_name = page.get("page") or page.get("page_name") or page.get("image") or ""
        tables = page.get("tables") or page.get("detected_tables") or page.get("results") or []
        metrics = page.get("extracted_metrics") or []
        if not tables and page.get("table_id"):
            tables = [page]
        total_tables += len(tables)
        total_metrics += len(metrics)
        for t in tables[:20]:
            if t.get("is_esg"): total_esg += 1
            rows.append({"page": page_name, "table_id": t.get("table_id", ""),
                         "rows": t.get("rows", ""), "cols": t.get("cols", ""),
                         "is_esg": t.get("is_esg", False),
                         "caption": (t.get("caption") or t.get("text") or "")[:180]})
    index_url = rel_output_path(run_id, "step3_table_understanding", "index.html")
    return {"total_pages": len(pages), "total_tables": total_tables, "total_metrics": total_metrics,
            "total_esg": total_esg, "tables": rows[:120],
            "index_url": index_url if os.path.exists(run_output_path(run_id, "step3_table_understanding", "index.html")) else None,
            "json_url": rel_output_path(run_id, "step3_table_understanding", "all_tables.json")}

@app.get("/api/runs/{run_id}/results/tables/report")
async def get_table_report(run_id: str):
    index_path = run_output_path(run_id, "step3_table_understanding", "index.html")
    if os.path.exists(index_path):
        return html_file_with_base(index_path, f"/runs/{run_id}/output/step3_table_understanding/")
    d = await get_table_results(run_id)
    cards = stat_card(d.get("total_pages", 0), "Pages") + stat_card(d.get("total_tables", 0), "Tables") + stat_card(d.get("total_metrics", 0), "Metrics") + stat_card(d.get("total_esg", 0), "ESG Tables")
    rows = "".join(
        f"<tr><td>{h(t.get('page'))}</td><td>{h(t.get('table_id'))}</td><td>{h(t.get('rows'))}</td><td>{h(t.get('cols'))}</td><td><span class='tag {'bad' if not t.get('is_esg') else ''}'>{'Yes' if t.get('is_esg') else 'No'}</span></td><td>{h(t.get('caption'))}</td></tr>"
        for t in d.get("tables", [])
    )
    body = f"<div class='page-card'><div class='page-h'><div class='page-title'>Detected Tables</div><div class='page-meta'>{h(len(d.get('tables', [])))} shown</div></div><div class='page-b'><table><thead><tr><th>Page</th><th>Table ID</th><th>Rows</th><th>Cols</th><th>ESG</th><th>Caption</th></tr></thead><tbody>{rows}</tbody></table></div></div>"
    return HTMLResponse(report_shell("Table Understanding Results", "Detected tables, extracted metrics, and ESG signals", cards, body))

@app.get("/api/runs/{run_id}/results/overlays")
async def get_overlays(run_id: str):
    od = os.path.join(BASE, "runs", run_id, "output", "step5_visualization", "overlay")
    if not os.path.exists(od): return {"overlays": []}
    files = sorted(glob.glob(os.path.join(od, "*_scope.jpg")))
    return {"overlays": [{"name": os.path.splitext(os.path.basename(f))[0].replace("_scope", ""),
                          "url": f"runs/{run_id}/output/step5_visualization/overlay/{os.path.basename(f)}"} for f in files]}

# ─── Legacy backward compat endpoints ──────────────────────────

@app.get("/api/status")
async def legacy_status():
    rid = await current_run_id()
    if rid: return await get_run_status(rid)
    return {"status": "idle", "steps": []}

@app.post("/api/run")
async def legacy_start(request: Request):
    body = await request.json()
    pdf_path = body.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(400, "Invalid pdf_path")
    run_id = f"run_legacy_{int(time.time())}"
    ctx = RunContext(run_id, BASE, os.path.join(BASE, "runs", run_id))
    ctx.ensure_dirs()
    with open(pdf_path, "rb") as f:
        ctx.save_pdf(f.read(), os.path.basename(pdf_path))
    async with state_lock:
        s = new_run_state(run_id)
        s.update(status="uploaded", document_name=os.path.basename(pdf_path))
        server_state["runs"][run_id] = s; server_state["current_run_id"] = run_id
    task = asyncio.create_task(pipeline_worker(ctx, run_id))
    async with state_lock:
        s2 = server_state["runs"].get(run_id, {}); s2["task"] = task
    return {"status": "started", "run_id": run_id}

@app.get("/api/stream")
async def legacy_stream():
    rid = await current_run_id()
    captured = rid
    async def gen():
        last_eid = -1
        while True:
            async with state_lock:
                if not captured or captured not in server_state["runs"]:
                    yield "data: {\"status\":\"idle\"}\n\n"
                    await asyncio.sleep(5); continue
                s = server_state["runs"][captured]
                events = s.get("events", []); new = [e for e in events if e.get("event_id", -1) > last_eid]
                if new: last_eid = max(e.get("event_id") for e in new)
                status = s.get("status", "idle")
                step_idx = s.get("current_step_index", -1); step_name = s.get("current_step", "")
                progress = s.get("progress_pct", 0); pages = s.get("total_pages", 0); page = s.get("current_page", 0)
                done = status in ("completed", "failed", "cancelled")
                logs = [{"ts": "", "msg": ev.get("message", ""), "step_index": step_idx} for ev in new if ev.get("message")]
                now_t = time.time(); started = s.get("started_at")
                elapsed = round(now_t - started) if started else 0
                remaining = None
                if progress > 5 and started:
                    remaining = round((elapsed / max(progress, 1)) * (100 - progress))
            data = {"status": status, "logs": logs, "progress_pct": progress,
                    "current_step_index": step_idx, "current_step": step_name,
                    "error": str(s.get("error", "")) if s.get("error") else None,
                    "done": done, "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining, "total_pages": pages,
                    "current_page": page, "document_name": s.get("document_name")}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            if done: break
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/results/summary")
async def legacy_results():
    rid = await current_run_id()
    if not rid: raise HTTPException(404, "No active run")
    return await get_results_summary(rid)

@app.get("/api/results/overlays")
async def legacy_overlays():
    rid = await current_run_id()
    if not rid: return {"overlays": []}
    return await get_overlays(rid)

@app.api_route("/api/reset", methods=["GET", "POST"])
async def reset():
    rid = await current_run_id()
    async with state_lock:
        if rid and rid in server_state["runs"]:
            run_state = server_state["runs"][rid]
            if run_state.get("status") == "running":
                task = run_state.get("task")
                if task: task.cancel()
            del server_state["runs"][rid]
        server_state["current_run_id"] = None
    return {"status": "reset"}

# ─── File serving ──────────────────────────────────────────────

@app.get("/runs/{run_id}/pages/{filename}")
async def serve_page_image(run_id: str, filename: str):
    fp = os.path.join(BASE, "runs", run_id, "pages", os.path.basename(filename))
    if os.path.exists(fp): return FileResponse(fp)
    raise HTTPException(404)

@app.get("/runs/{run_id}/output/{rest:path}")
async def serve_output(run_id: str, rest: str):
    fp = os.path.join(BASE, "runs", run_id, "output", rest)
    if os.path.exists(fp): return FileResponse(fp)
    raise HTTPException(404)

@app.get("/output/{rest:path}")
async def serve_legacy_output(rest: str):
    fp = os.path.join(BASE, "test", "output", rest)
    if os.path.exists(fp): return FileResponse(fp)
    rid = await current_run_id()
    if rid: return await serve_output(rid, rest)
    raise HTTPException(404)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
