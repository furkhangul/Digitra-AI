"""Surface-level verification of the TID hand, reported in three separate parts.

1. geometry   — the exported mesh: closed, manifold, weights normalised
2. contact    — for every two-hand sign, the gap at the contact the sign intends,
                and, separately, the deepest collision anywhere else
3. motion     — the same collision measure sampled across each transition

The three are reported apart on purpose. A solved contact is not evidence that
nothing else touches, and none of it is evidence of linguistic correctness.

Run from apps/web/scripts/tid_sdf:  python verify.py
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import letters as LT, joined as J

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ROOT = r"C:/Users/Furkan/Desktop/Digitra AI"
OUT = os.path.join(ROOT, "output", "models")

PEN = -0.02        # task acceptance threshold for interpenetration
FAR = 0.06         # wider than this counts as an unintended gap
CONTACT_RADIUS = .34
SEPARATED = {"Ç": .09, "Ö": .11}

def anchor(hand, shape, name):
    return hand.M @ J.local_anchor(shape, name) + hand.T

def centroid(hand):
    """Palm centre: extended fingers may cross without swapping hand bodies."""
    return hand.to_world([0, .42, .17])

def safe_min(values):
    return float(values.min()) if len(values) else None

def contact_report(density=90):
    rows = []
    for ch, L in LT.letters():
        R, Lh = LT.pair(L)
        if not (R.visible and Lh.visible):
            rows.append({
                "letter": ch, "hands": 1,
                "intendedContactGap": None,
                "deepestOtherCollision": None,
                "sideOrder": None,
                "status": "single-hand",
            }); continue
        sr = R.surface_points(n_per_seg=density, seed=101)
        sl = Lh.surface_points(n_per_seg=density, seed=102)
        right_x = float(centroid(R)[0]); left_x = float(centroid(Lh)[0])
        worst = float(min(Lh.sdf(sr).min(), R.sdf(sl).min()))
        con = L.get("contact"); intended = None
        other_r, other_l = sr, sl
        if con:
            ar = anchor(R, L["authored"]["right"]["shape"], con["right"])
            al = anchor(Lh, L["authored"]["left"]["shape"], con["left"])
            mask_r = np.linalg.norm(sr - ar, axis=1) < CONTACT_RADIUS
            mask_l = np.linalg.norm(sl - al, axis=1) < CONTACT_RADIUS
            nr, nl = sr[mask_r], sl[mask_l]
            other_r, other_l = sr[~mask_r], sl[~mask_l]
            if len(nr) > 10 and len(nl) > 10:
                intended = float(min(Lh.sdf(nr).min(), R.sdf(nl).min()))
        other_values = []
        if len(other_r): other_values.append(safe_min(Lh.sdf(other_r)))
        if len(other_l): other_values.append(safe_min(R.sdf(other_l)))
        deepest_other = min(v for v in other_values if v is not None)
        target = SEPARATED.get(ch, 0.0)
        contact_error = None if intended is None else intended - target
        contact_ok = intended is not None and abs(contact_error) <= .025
        collision_ok = deepest_other >= PEN
        rows.append({
            "letter": ch, "hands": 2,
            "contactAnchors": con and [con["right"], con["left"]],
            "targetContactGap": target,
            "intendedContactGap": None if intended is None else round(intended, 4),
            "contactError": None if contact_error is None else round(contact_error, 4),
            "worstCollision": round(worst, 4),
            "deepestOtherCollision": round(deepest_other, 4),
            "rightCenterX": round(right_x, 4),
            "leftCenterX": round(left_x, 4),
            "sideOrder": right_x > left_x,
            "samples": int(len(sr) + len(sl)),
            "contactOK": contact_ok,
            "collisionOK": collision_ok,
            "status": "penetrating" if not collision_ok else ("contact-ok" if contact_ok else "contact-misaligned"),
        })
    return rows

def motion_report(path, density=45):
    """Collision across recorded transition frames, not just the held pose."""
    if not os.path.exists(path): return None
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for L in data["letters"]:
        worst, at = 1e9, None
        for f in L["frames"][::3]:
            R = LT.Hand(f["frame"]["right"], "right"); Lh = LT.Hand(f["frame"]["left"], "left")
            if not (R.visible and Lh.visible): continue
            sr = R.surface_points(n_per_seg=density, seed=201)
            sl = Lh.surface_points(n_per_seg=density, seed=202)
            w = float(min(Lh.sdf(sr).min(), R.sdf(sl).min()))
            if w < worst: worst, at = w, f["t"]
        out.append({"letter": L["ch"], "from": L["from"], "worstCollision": round(worst, 4),
                    "atSeconds": at, "frames": len(L["frames"])})
    return out

if __name__ == "__main__":
    report = {
        "note": "Surface distances in palm-length units (1.0 = wrist to middle knuckle). "
                "Negative means the two surfaces overlap.",
        "thresholds": {"penetration": PEN, "gap": FAR},
        "geometry": json.load(open(os.path.join(OUT, "sdf-build-report.json"), encoding="utf-8")),
        "contact": contact_report(),
    }
    # Reuse the standalone transition scan when it is present; it samples every
    # letter's approach, which is far too slow to redo inline.
    cached = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion-check.json")
    if os.path.exists(cached):
        report["motionSampled"] = json.load(open(cached, encoding="utf-8"))
        report["motionSampledNote"] = "every 4th frame of the transition into each letter, full-surface sampling"
    else:
        mo = motion_report(os.path.join(ROOT, "tmp", "tid", "motion-frames.json"))
        if mo: report["motionSampled"] = mo
    pairs = [r for r in report["contact"] if r["hands"] == 2]
    mo = report.get("motionSampled") or []
    report["summary"] = {
        "pairLetters": len(pairs),
        "transitionsScanned": len(mo),
        "transitionsColliding": [m["letter"] for m in mo if m["worstCollision"] < PEN],
        "worstDuringTransition": min((m["worstCollision"] for m in mo), default=None),
        "penetrating": [r["letter"] for r in pairs if not r["collisionOK"]],
        "contactMisaligned": [r["letter"] for r in pairs if not r["contactOK"]],
        "wrongSideOrder": [r["letter"] for r in pairs if not r["sideOrder"]],
        "worstCollision": min(r["worstCollision"] for r in pairs),
        "deepestOtherCollision": min(r["deepestOtherCollision"] for r in pairs),
    }
    with open(os.path.join(OUT, "surface-contact-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
