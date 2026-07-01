"""
Regression test suite for the Find Total Look detection pipeline.
Run before and after any change to verify no regressions.

Usage:
    python test_regression.py

Pass criteria: each test case checks the exact slot, color, and (where specified)
length/style. A PASS means the detected value matches expected.
"""
import sys
import os
from PIL import Image
import importlib

# Suppress print output from main.py during tests
import io
from contextlib import redirect_stdout

ASSETS = r"C:\Users\dylan\.cursor\projects\c-Users-dylan-Desktop-FML-CODE-PR3-Backend\assets"

def asset(uid):
    prefix = "c__Users_dylan_AppData_Roaming_Cursor_User_workspaceStorage_4b19a1b16d0f8f8b9701b1ddc20cfb05_images_image-"
    return os.path.join(ASSETS, f"{prefix}{uid}.png")

# ---------------------------------------------------------------------------
# Test table — (label, image_uid, slot, expected_color, expected_detail)
#
# expected_detail checks:
#   - top       → topStyle  (tshirt / tank / strapless / ...)
#   - skirt     → skirtLength (mini / midi / maxi)
#   - bottom    → bottomLength (shorts / long_pants)
#   - shoes     → shoeStyle (flat_shoe / flip_flop / birkenstock / ...)
#   - None      → skip detail check
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ── Tops ─────────────────────────────────────────────────────────────────
    ("white_crop_top",       "7177a702-aa95-4eb8-88c0-46fb084d8d85", "top",   "white",  "tank"),
    ("studio_white_tshirt",  "79739444-7de0-44e8-9c21-ac9323176c42", "top",   "white",  "tshirt"),
    ("outdoor_beige_tshirt", "78d743a2-1ca8-4fed-84d6-9d8bfa9f3676", "top",   "beige",  "tshirt"),
    ("white_crop_boots",     "053b8000-7a1f-4755-97c7-3a8181cf2b47", "top",   "white",  None),
    ("dark_stripe_tshirt",   "7d505665-19f2-495c-a66f-1f96279a8b4d", "top",   "black",  "tshirt"),

    # ── Skirts ───────────────────────────────────────────────────────────────
    ("black_mini_skirt",     "7177a702-aa95-4eb8-88c0-46fb084d8d85", "skirt", "black",  "mini"),
    ("black_mini_boots",     "053b8000-7a1f-4755-97c7-3a8181cf2b47", "skirt", "black",  "mini"),
    ("grey_maxi_skirt",      "638a84d0-0d6a-4747-a7e5-bcaa7cb8c403", "skirt", "grey",   "maxi"),

    # ── Bottoms ──────────────────────────────────────────────────────────────
    ("navy_jeans",           "79739444-7de0-44e8-9c21-ac9323176c42", "bottom", "navy",  "long_pants"),
    ("dark_jeans",           "7d505665-19f2-495c-a66f-1f96279a8b4d", "bottom", "grey",  "long_pants"),

    # ── Shoes ────────────────────────────────────────────────────────────────
    ("black_boots",          "053b8000-7a1f-4755-97c7-3a8181cf2b47", "shoes", "black",  "heeled_boot"),
    ("beige_flip_flops",     "75b9175d-f1a9-458a-a691-f452ad2bbd13", "shoes", "beige",  "flip_flop"),
]


def run_tests(verbose=True):
    # Import fresh (allows re-running after edits without restart)
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    sys.path.insert(0, os.path.dirname(__file__))
    import main

    results = []
    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print(f"{'LABEL':<30} {'SLOT':<8} {'COLOR':<12} {'DETAIL':<14} STATUS")
    print("=" * 70)

    for label, uid, slot, exp_color, exp_detail in TEST_CASES:
        path = asset(uid)
        if not os.path.exists(path):
            print(f"{'[MISSING]':<30} {label}")
            continue

        img = Image.open(path).convert("RGB")
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = main.process_total_look_logic(img)

        items = result.get("items", [])
        detected = next((i for i in items if i.get("slotId") == slot), None)

        if detected is None:
            status = "FAIL (slot not detected)"
            failed += 1
        else:
            got_color = detected.get("color", "")
            got_detail = (
                detected.get("topStyle")
                or detected.get("skirtLength")
                or detected.get("bottomLength")
                or detected.get("shoeStyle")
                or ""
            )
            color_ok = got_color == exp_color
            detail_ok = (exp_detail is None) or (got_detail == exp_detail)
            ok = color_ok and detail_ok

            if ok:
                status = "PASS"
                passed += 1
            else:
                parts = []
                if not color_ok:
                    parts.append(f"color={got_color}≠{exp_color}")
                if not detail_ok:
                    parts.append(f"detail={got_detail}≠{exp_detail}")
                status = f"FAIL ({', '.join(parts)})"
                failed += 1

        print(f"{label:<30} {slot:<8} {exp_color:<12} {str(exp_detail):<14} {status}")
        results.append((label, slot, status))

    print("=" * 70)
    total = passed + failed
    print(f"RESULT: {passed}/{total} passed  |  {failed} failed\n")
    return passed, failed, results


if __name__ == "__main__":
    passed, failed, _ = run_tests()
    sys.exit(0 if failed == 0 else 1)
