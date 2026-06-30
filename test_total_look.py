"""Simple test for Find Total Look — pass a photo path as argument."""
import base64
import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python test_total_look.py path\\to\\outfit-photo.jpg")
    sys.exit(1)

path = sys.argv[1]
print(f"Reading: {path}")

with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

print("Sending to ML (wait ~10-30 sec)...")
r = requests.post(
    "http://localhost:8000/process-total-look-base64",
    json={"image": f"data:image/jpeg;base64,{b64}"},
    timeout=180,
)

if r.status_code != 200:
    print("FAILED:", r.status_code, r.text[:500])
    sys.exit(1)

data = r.json()
meta = data.get("detectionMeta", {})
items = data.get("items", [])

print()
print("=== RESULT ===")
print(f"Found {meta.get('garmentCount', len(items))} garment(s): {meta.get('slots', [])}")
print()

for item in items:
    print(f"  {item.get('slotId', '?').upper():8}  color={item.get('color')}  category={item.get('categoryGroup')}  confidence={item.get('confidence')}")

if not items:
    print("  (nothing detected — try another photo)")
else:
    print()
    print("OK — Step 1 works. Ready for Step 2 (backend).")
