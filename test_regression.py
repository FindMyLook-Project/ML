"""
Regression test suite for the Find Total Look AND single-item detection pipelines.
Run before and after any change to verify no regressions in either pipeline.

Usage:
    python test_regression.py

═══════════════════════════════════════════════════════════════════════
PIPELINE ISOLATION MAP (Step D audit)
═══════════════════════════════════════════════════════════════════════

SINGLE-ITEM (/process-look-base64, /process-look, /process-url)
  └─ process_image_logic()
       ├─ detect_category_clip()
       ├─ detect_fabric_clip()
       ├─ get_fashion_color()          ← pixel rules only (Steps A-B changes)
       ├─ detect_shoe_style_clip()
       ├─ detect_top_style_clip()
       └─ detect_bottom_length_clip()
       ✗ _analyze_garment_crop()           ← NOT called
       ✗ _detect_white_vs_beige_top_clip() ← NOT called (Step C CLIP tiebreaker)
       ✗ _detect_pastel_top_color_clip()   ← NOT called
       ✗ _refine_top_attributes()          ← NOT called
       ✗ process_total_look_logic()        ← NOT called

TOTAL LOOK (/process-total-look-base64)
  └─ process_total_look_logic()
       └─ _analyze_garment_crop()
            ├─ detect_category_clip()
            ├─ detect_fabric_clip()
            ├─ get_fashion_color()              ← shared with single-item
            ├─ _detect_denim_vest_top_clip()    ← Total Look only
            ├─ _detect_pastel_top_color_clip()  ← Total Look only
            ├─ _detect_stripe_top_clip()        ← Total Look only
            ├─ _refine_top_attributes()         ← Total Look only
            ├─ _detect_white_vs_beige_top_clip()← Total Look only (Step C)
            ├─ detect_shoe_style_clip()
            ├─ detect_top_style_clip()
            ├─ detect_bottom_length_clip()
            └─ detect_skirt_length_clip()

SHARED CODE (safe to improve — benefits both pipelines):
  • get_fashion_color()           (Step B: consolidated top pipeline)
  • detect_skirt_length_clip()    (Step A: removed mini-bias)
  • CLIP_SKIRT_LENGTH_PROMPTS     (Step A: color-agnostic prompts)
  • COLOR_TEXT_PROMPTS            (Step C: sharper white/beige centroids)
  • _classify_stripe_dark_pixels  (Step A: tighter navy/black threshold)
═══════════════════════════════════════════════════════════════════════
"""
import sys
import os
from PIL import Image
import importlib
import io
from contextlib import redirect_stdout

ASSETS = r"C:\Users\dylan\.cursor\projects\c-Users-dylan-Desktop-FML-CODE-PR3-Backend\assets"


def asset(uid):
    prefix = "c__Users_dylan_AppData_Roaming_Cursor_User_workspaceStorage_4b19a1b16d0f8f8b9701b1ddc20cfb05_images_image-"
    return os.path.join(ASSETS, f"{prefix}{uid}.png")


# ──────────────────────────────────────────────────────────────────────────────
# TOTAL LOOK test cases
# (label, image_uid, slot, expected_color, expected_detail)
# expected_detail: topStyle | skirtLength | bottomLength | shoeStyle | None
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_LOOK_CASES = [
    # Tops
    ("white_crop_top",       "7177a702-aa95-4eb8-88c0-46fb084d8d85", "top",    "white",  "tank"),
    ("studio_white_tshirt",  "79739444-7de0-44e8-9c21-ac9323176c42", "top",    "white",  "tshirt"),
    ("outdoor_beige_tshirt", "78d743a2-1ca8-4fed-84d6-9d8bfa9f3676", "top",    "beige",  "tshirt"),
    ("white_crop_boots",     "053b8000-7a1f-4755-97c7-3a8181cf2b47", "top",    "white",  None),
    ("dark_stripe_tshirt",   "7d505665-19f2-495c-a66f-1f96279a8b4d", "top",    "black",  "tshirt"),
    # Skirts
    ("black_mini_skirt",     "7177a702-aa95-4eb8-88c0-46fb084d8d85", "skirt",  "black",  "mini"),
    ("black_mini_boots",     "053b8000-7a1f-4755-97c7-3a8181cf2b47", "skirt",  "black",  "mini"),
    ("grey_maxi_skirt",      "638a84d0-0d6a-4747-a7e5-bcaa7cb8c403", "skirt",  "grey",   "maxi"),
    # Bottoms
    ("navy_jeans",           "79739444-7de0-44e8-9c21-ac9323176c42", "bottom", "navy",   "long_pants"),
    ("dark_jeans",           "7d505665-19f2-495c-a66f-1f96279a8b4d", "bottom", "grey",   "long_pants"),
    # Shoes
    ("black_boots",          "053b8000-7a1f-4755-97c7-3a8181cf2b47", "shoes",  "black",  "heeled_boot"),
    ("beige_flip_flops",     "75b9175d-f1a9-458a-a691-f452ad2bbd13", "shoes",  "beige",  "flip_flop"),
]


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-ITEM isolation + smoke tests (process_image_logic)
#
# Two kinds of checks:
#   1. ISOLATION — the Step C CLIP tiebreaker (_detect_white_vs_beige_top_clip)
#      must NEVER be called from process_image_logic. Verified via monkey-patch.
#   2. SMOKE    — process_image_logic must return at least one item with valid
#      structure {color, categoryGroup, embedding, colorVector}.
#
# Note: these are full-body outfit photos, not clean product shots.
# Accurate color from process_image_logic on outfit photos depends on which YOLO
# box gets assigned — that is expected and not the single-item pipeline's use case.
# ──────────────────────────────────────────────────────────────────────────────
SINGLE_ITEM_SMOKE_UIDS = [
    "79739444-7de0-44e8-9c21-ac9323176c42",  # studio white tshirt + navy jeans
    "78d743a2-1ca8-4fed-84d6-9d8bfa9f3676",  # outdoor beige tshirt
    "75b9175d-f1a9-458a-a691-f452ad2bbd13",  # beige flip-flops
]


def _get_main():
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    sys.path.insert(0, os.path.dirname(__file__))
    import main
    return main


def run_total_look_tests(main_mod) -> tuple:
    passed = failed = 0
    results = []

    print("\n" + "=" * 70)
    print("TOTAL LOOK PIPELINE  (process_total_look_logic)")
    print(f"{'LABEL':<30} {'SLOT':<8} {'COLOR':<12} {'DETAIL':<14} STATUS")
    print("=" * 70)

    for label, uid, slot, exp_color, exp_detail in TOTAL_LOOK_CASES:
        path = asset(uid)
        if not os.path.exists(path):
            print(f"{'[MISSING]':<30} {label}")
            continue

        img = Image.open(path).convert("RGB")
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = main_mod.process_total_look_logic(img)

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
            if color_ok and detail_ok:
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


def run_single_item_tests(main_mod) -> tuple:
    """
    Two checks per image:
      1. ISOLATION: the CLIP white/beige tiebreaker must NOT be invoked.
      2. SMOKE: at least one valid item returned with required fields.
    """
    passed = failed = 0
    results = []

    print("=" * 70)
    print("SINGLE-ITEM PIPELINE  (process_image_logic)")
    print(f"{'UID (short)':<14} {'ISOLATION':<22} {'SMOKE':<22} STATUS")
    print("=" * 70)

    REQUIRED_FIELDS = {"color", "categoryGroup", "embedding", "colorVector"}

    for uid in SINGLE_ITEM_SMOKE_UIDS:
        path = asset(uid)
        short = uid[:8]
        if not os.path.exists(path):
            print(f"{short:<14} [MISSING IMAGE]")
            continue

        img = Image.open(path).convert("RGB")

        # ── Isolation check: monkey-patch the tiebreaker ──────────────────
        tiebreaker_called = [False]
        original_fn = main_mod._detect_white_vs_beige_top_clip
        def patched(crop_img, pixel_color, _flag=tiebreaker_called):
            _flag[0] = True
            return original_fn(crop_img, pixel_color)
        main_mod._detect_white_vs_beige_top_clip = patched

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                items = main_mod.process_image_logic(img)
            isolation_ok = not tiebreaker_called[0]
            isolation_status = "PASS (not called)" if isolation_ok else "FAIL (tiebreaker called!)"
        except Exception as e:
            isolation_ok = False
            isolation_status = f"FAIL ({e})"
            items = []
        finally:
            main_mod._detect_white_vs_beige_top_clip = original_fn

        # ── Smoke check: items returned with valid fields ─────────────────
        has_items = len(items) > 0
        fields_ok = all(
            all(f in item for f in REQUIRED_FIELDS)
            for item in items
        ) if has_items else False
        smoke_ok = has_items and fields_ok
        smoke_status = f"PASS ({len(items)} items)" if smoke_ok else \
                       ("FAIL (no items)" if not has_items else "FAIL (missing fields)")

        ok = isolation_ok and smoke_ok
        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"{short:<14} {isolation_status:<22} {smoke_status:<22} {status}")
        results.append((uid, status))

    print("=" * 70)
    total = passed + failed
    print(f"RESULT: {passed}/{total} passed  |  {failed} failed\n")
    return passed, failed, results


def run_tests(verbose=True):
    main_mod = _get_main()

    tl_pass, tl_fail, tl_res = run_total_look_tests(main_mod)
    si_pass, si_fail, si_res = run_single_item_tests(main_mod)

    total_pass = tl_pass + si_pass
    total_fail = tl_fail + si_fail
    print(f"OVERALL: {total_pass}/{total_pass + total_fail} passed  |  {total_fail} failed")
    return total_pass, total_fail, tl_res + si_res


if __name__ == "__main__":
    passed, failed, _ = run_tests()
    sys.exit(0 if failed == 0 else 1)
