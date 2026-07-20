"""Test all UI API endpoints and report errors."""
import urllib.request, urllib.error, json, sys

BASE = "http://127.0.0.1:8000"
errors = []
results = []

def test(method, path, expected=None, desc=""):
    url = BASE + path
    try:
        req = urllib.request.Request(url, method=method)
        resp = urllib.request.urlopen(req, timeout=10)
        status = resp.status
        body = resp.read().decode("utf-8")
        if expected and status != expected:
            errors.append(f"{method} {path}: expected {expected}, got {status} ({desc})")
        results.append(f"OK  {method} {path} -> {status}")
        return status, body
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode("utf-8")[:300]
        if expected and status != expected:
            errors.append(f"{method} {path}: expected {expected}, got {status} ({desc})")
        results.append(f"OK  {method} {path} -> {status} (expected {expected})")
        return status, body
    except Exception as e:
        errors.append(f"{method} {path}: {e} ({desc})")
        results.append(f"ERR {method} {path} -> {e}")
        return None, str(e)

print("=" * 60)
print("Eco-Lens UI API Error Scan")
print("=" * 60)
print()

# 1. Main page
test("GET", "/", 200, "SPA frontend")

# 2. API endpoints
test("GET", "/api/status", 200, "Legacy status (idle)")
test("POST", "/api/reset", 200, "Legacy reset")

# 3. Non-existent run tests (should 404 gracefully)
test("GET", "/api/runs/nonexistent/status", 404, "Non-existent run status")
st, body = test("GET", "/api/runs/nonexistent/results/summary", 404, "Non-existent run results")
test("GET", "/api/runs/nonexistent/step-logs/0", 200, "Non-existent step logs")
test("GET", "/api/runs/nonexistent/results/overlays", 200, "Non-existent overlays")

# 4. Legacy endpoints
test("GET", "/api/results/summary", 404, "Legacy results (no run)")
test("GET", "/api/results/overlays", 200, "Legacy overlays")

# 5. SSE stream format check
print()
print("--- SSE legacy stream check ---")
try:
    req = urllib.request.Request(BASE + "/api/stream")
    resp = urllib.request.urlopen(req, timeout=3)
    chunk = resp.read(500).decode("utf-8")
    print(f"  SSE first chunk ({len(chunk)}b): {chunk[:200]}...")
    resp.close()
except Exception as e:
    print(f"  SSE stream: {e} (expected when idle)")

# 6. Status response fields check
print()
print("--- Status response fields ---")
st, body = test("GET", "/api/status")
if st == 200:
    data = json.loads(body)
    required = ["status", "steps"]
    for field in required:
        if field not in data:
            errors.append(f"/api/status missing field: {field}")
    print(f"  status: {data.get('status')}")
    print(f"  steps count: {len(data.get('steps', []))}")
    print(f"  has current_step: {'current_step' in data}")
    print(f"  has elapsed_seconds: {'elapsed_seconds' in data}")
    print(f"  has progress_pct: {'progress_pct' in data}")
    print(f"  has steps: {'steps' in data}")

# 7. Upload validation
print()
print("--- Upload validation ---")
import io
# Test invalid file type
try:
    data = b"not a pdf"
    req = urllib.request.Request(BASE + "/api/upload", data=data, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=----test")
    resp = urllib.request.urlopen(req, timeout=5)
    errors.append("Upload accepted invalid content")
    print("  Invalid upload: ACCEPTED (ERROR)")
except urllib.error.HTTPError as e:
    print(f"  Invalid upload: REJECTED ({e.code}) - OK")

# 8. UI frontend HTML check
print()
print("--- Frontend HTML sanity ---")
st, body = test("GET", "/")
if st == 200:
    checks = [
        ("/api/upload", "upload endpoint"),
        ("/api/reset", "reset endpoint"),
        ("/api/runs/", "run endpoints"),
        ("EventSource", "SSE connection"),
    ]
    for pattern, name in checks:
        if pattern in body:
            print(f"  OK {name}: found '{pattern}'")
        else:
            errors.append(f"Frontend missing {name}: '{pattern}' not found")

# Summary
print()
print("=" * 60)
print(f"RESULTS: {len(results)} endpoints tested, {len(errors)} errors")
print("=" * 60)
for r in results:
    print(f"  {r}")

if errors:
    print()
    print("ERRORS:")
    for e in errors:
        print(f"  [BUG] {e}")
    sys.exit(1)
else:
    print("No errors found.")
