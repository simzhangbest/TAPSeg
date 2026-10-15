from __future__ import annotations

import argparse
import os
import shutil
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_nifti_in_dir(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [p for p in input_dir.iterdir() if p.is_file() and _is_nifti(p)]
    files.sort(key=lambda p: p.name)
    return files


FDI_QUADRANTS = {
    ("upper", "right"): 1,
    ("upper", "left"): 2,
    ("lower", "left"): 3,
    ("lower", "right"): 4,
}


@dataclass(frozen=True)
class ArchModel:
    "Parabolic dental arch in PCA coordinates: v = a * (u - u0)**2 + c."

    mean_xy: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    u0: float
    a: float
    c: float

    def project_uv(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(xy, dtype=float) - self.mean_xy.reshape(1, 2)
        u = (pts @ self.u_axis.reshape(2, 1)).reshape(-1)
        v = (pts @ self.v_axis.reshape(2, 1)).reshape(-1)
        return u, v

    def project_u_on_curve(self, u: np.ndarray, v: np.ndarray, *, max_iter: int = 20) -> np.ndarray:
        "Project points onto the arch using Newton iterations for squared distance."
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        if u.size == 0:
            return u

        a = float(self.a)
        if abs(a) < 1e-10:
            return u

        u0 = float(self.u0)
        c = float(self.c)

        x = u.copy()
        for _ in range(int(max_iter)):
            du = x - u0
            f = a * du * du + c
            r = f - v
            g = 2.0 * (x - u) + 4.0 * a * du * r
            gp = 2.0 + 4.0 * a * r + 8.0 * a * a * du * du
            step = g / np.maximum(gp, 1e-10)
            x_new = x - step
            if float(np.max(np.abs(step))) < 1e-6:
                x = x_new
                break
            x = x_new
        return x

    def arc_length_from_vertex(self, u: np.ndarray) -> np.ndarray:
        "Return nonnegative arc length from the vertex, with a straight-line limit."
        u = np.asarray(u, dtype=float).reshape(-1)
        du = np.abs(u - float(self.u0))
        a = float(self.a)
        if abs(a) < 1e-10:
            return du
        a = abs(a)
        t = 2.0 * a * du
        return (t * np.sqrt(1.0 + t * t) + np.arcsinh(t)) / (4.0 * a)


def _parse_fdi_from_filename(path: Path) -> int | None:
    m = re.search(r"[Tt](\d{2})", path.stem)
    if not m:
        return None
    v = int(m.group(1))
    if 11 <= v <= 48:
        return v
    return None


def _candidate_thresholds(values: np.ndarray, max_candidates: int = 72) -> list[float]:
    v = np.asarray(values, dtype=float)
    uniq = np.unique(v)
    if uniq.size <= 1:
        return [float(uniq[0]) if uniq.size == 1 else 0.0]

    uniq.sort()
    mids = ((uniq[:-1] + uniq[1:]) / 2.0).tolist()
    if len(mids) <= max_candidates:
        return mids

    qs = np.linspace(0.02, 0.98, max_candidates)
    return [float(np.quantile(mids, q)) for q in qs]


def _compute_centroids_physical(label_img: sitk.Image) -> tuple[list[int], np.ndarray]:
    "Return nonzero labels and their physical centroids as an (n, 3) array."
    pid = int(label_img.GetPixelIDValue())
    if pid in (
        sitk.sitkUInt8,
        sitk.sitkUInt16,
        sitk.sitkUInt32,
        sitk.sitkUInt64,
        sitk.sitkInt8,
        sitk.sitkInt16,
        sitk.sitkInt32,
        sitk.sitkInt64,
    ):
        lab = sitk.Cast(label_img, sitk.sitkUInt16)
    else:
        lab = sitk.Cast(sitk.Round(label_img), sitk.sitkUInt16)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lab)

    labels = [int(l) for l in stats.GetLabels() if int(l) != 0]
    centroids = np.array([stats.GetCentroid(int(l)) for l in labels], dtype=float)
    return labels, centroids


def _compute_label_bounding_boxes_ijk(label_img: sitk.Image, labels: list[int]) -> dict[int, tuple[int, int, int, int, int, int]]:
    """
    Return per-label axis-aligned bounding boxes in index space.

    Output: {label: (x, y, z, size_x, size_y, size_z)}
    """
    if not labels:
        return {}

    pid = int(label_img.GetPixelIDValue())
    if pid in (
        sitk.sitkUInt8,
        sitk.sitkUInt16,
        sitk.sitkUInt32,
        sitk.sitkUInt64,
        sitk.sitkInt8,
        sitk.sitkInt16,
        sitk.sitkInt32,
        sitk.sitkInt64,
    ):
        lab = sitk.Cast(label_img, sitk.sitkUInt16)
    else:
        lab = sitk.Cast(sitk.Round(label_img), sitk.sitkUInt16)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lab)

    out: dict[int, tuple[int, int, int, int, int, int]] = {}
    for l in labels:
        ll = int(l)
        try:
            if not stats.HasLabel(ll):
                continue
        except Exception:
            pass
        bb = stats.GetBoundingBox(ll)
        out[ll] = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4]), int(bb[5]))
    return out


def _compute_label_bbox_extents_mm(label_img: sitk.Image, labels: list[int]) -> dict[int, np.ndarray]:
    """
    Return per-label physical extents (mm) computed from axis-aligned bounding boxes.

    Output: {label: np.array([sx_mm, sy_mm, sz_mm])}
    """
    if not labels:
        return {}

    pid = int(label_img.GetPixelIDValue())
    if pid in (
        sitk.sitkUInt8,
        sitk.sitkUInt16,
        sitk.sitkUInt32,
        sitk.sitkUInt64,
        sitk.sitkInt8,
        sitk.sitkInt16,
        sitk.sitkInt32,
        sitk.sitkInt64,
    ):
        lab = sitk.Cast(label_img, sitk.sitkUInt16)
    else:
        lab = sitk.Cast(sitk.Round(label_img), sitk.sitkUInt16)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lab)

    spacing = np.asarray(label_img.GetSpacing(), dtype=float).reshape(3)
    out: dict[int, np.ndarray] = {}
    for l in labels:
        ll = int(l)
        try:
            if not stats.HasLabel(ll):
                continue
        except Exception:
            pass

        bb = stats.GetBoundingBox(ll)
        size_ijk = np.asarray(bb[3:6], dtype=float).reshape(3)
        out[ll] = (size_ijk * spacing).astype(float, copy=False)
    return out


def _compute_label_physical_sizes_mm3(label_img: sitk.Image, labels: list[int]) -> dict[int, float]:
    """
    Return per-label physical volume (mm^3) from LabelShapeStatistics.
    """
    if not labels:
        return {}

    pid = int(label_img.GetPixelIDValue())
    if pid in (
        sitk.sitkUInt8,
        sitk.sitkUInt16,
        sitk.sitkUInt32,
        sitk.sitkUInt64,
        sitk.sitkInt8,
        sitk.sitkInt16,
        sitk.sitkInt32,
        sitk.sitkInt64,
    ):
        lab = sitk.Cast(label_img, sitk.sitkUInt16)
    else:
        lab = sitk.Cast(sitk.Round(label_img), sitk.sitkUInt16)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lab)

    out: dict[int, float] = {}
    for l in labels:
        ll = int(l)
        try:
            if not stats.HasLabel(ll):
                continue
        except Exception:
            pass
        out[ll] = float(stats.GetPhysicalSize(ll))
    return out


def _direction_is_identity(label_img: sitk.Image, *, tol: float = 1e-4) -> bool:
    d = np.asarray(label_img.GetDirection(), dtype=float).reshape(3, 3)
    return bool(np.allclose(d, np.eye(3, dtype=float), atol=float(tol), rtol=0.0))


def _estimate_x0_by_count_symmetry(x_phys: np.ndarray) -> float:
    """
    Estimate midline x0 (physical) using tooth count symmetry along the x axis.

    Idea: choose an x0 that makes the number of teeth on each side as balanced as possible.
    This is robust for mandible cases with missing/unerupted teeth where shape-based symmetry can fail.
    """
    x = np.asarray(x_phys, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(x[0])

    xs = np.unique(np.sort(x))
    if xs.size == 1:
        return float(xs[0])

    mids = (xs[:-1] + xs[1:]) / 2.0
    candidates = np.concatenate([mids, np.array([float(np.median(x)), float(np.mean(x))], dtype=float)])

    target = float(np.mean(x))
    best = float(np.median(x))
    best_cost: float | None = None
    for c in candidates.tolist():
        c = float(c)
        n_left = int(np.sum(x > c))
        n_right = int(np.sum(x < c))
        imbalance = abs(n_left - n_right)
        reg = 0.01 * abs(c - target) / max(1e-6, float(xs[-1] - xs[0]))
        cost = float(imbalance) + float(reg)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = float(c)
    return float(best)


def _pick_upper_canine_label(
    label_img: sitk.Image,
    *,
    quadrant_labels: list[int],
    centroid_xy: np.ndarray,
) -> int | None:
    """
    Upper jaw: pick the canine label within a quadrant.

    Heuristic (as requested): the canine is the longest tooth.
    Implementation uses physical z-extent (mm) from bounding box.
    """
    if not quadrant_labels:
        return None

    bbox_ext = _compute_label_bbox_extents_mm(label_img, quadrant_labels)
    lenz = np.array([float(bbox_ext.get(int(l), np.zeros((3,), dtype=float))[2]) for l in quadrant_labels], dtype=float)

    return int(quadrant_labels[int(np.argmax(lenz))])


def _canine_anchor_positions_from_ordered(
    ordered_labels: list[int],
    *,
    canine_label: int | None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Given an ordered label list (near midline -> posterior), force canine to position 3.

    Returns:
    - positioned: list of (pos, label)
    - extras: labels that fall before canine beyond positions 1-2
    """
    if not ordered_labels or canine_label is None:
        return [(i + 1, int(l)) for i, l in enumerate(ordered_labels)], []
    if int(canine_label) not in set(int(x) for x in ordered_labels):
        return [(i + 1, int(l)) for i, l in enumerate(ordered_labels)], []

    k_canine = int([i for i, x in enumerate(ordered_labels) if int(x) == int(canine_label)][0])
    before = [int(x) for x in ordered_labels[:k_canine]]
    after = [int(x) for x in ordered_labels[k_canine + 1 :]]

    positioned: list[tuple[int, int]] = []
    extras: list[int] = []
    if len(before) >= 1:
        positioned.append((1, int(before[0])))
    if len(before) >= 2:
        positioned.append((2, int(before[1])))
    if len(before) > 2:
        extras.extend(before[2:])

    positioned.append((3, int(canine_label)))
    for kk, x in enumerate(after, start=4):
        positioned.append((int(kk), int(x)))
    return positioned, extras


def _canine_anchor_reindex_groups(
    groups: list[list[int]],
    *,
    sub_labels: list[int],
    canine_label: int | None,
) -> tuple[list[tuple[int, list[int]]], list[list[int]]]:
    """
    Reindex ordered groups (near-midline -> posterior) using a canine anchor:
    - the group containing the canine is forced to position 3
    - at most two groups before canine can occupy positions 1 and 2
    - groups after canine become positions 4..N

    Returns:
    - positioned: list of (pos, group)
    - extras: groups that should be treated as "extra teeth" (do not merge into standard positions)
    """
    if not groups or canine_label is None:
        return [(i + 1, g) for i, g in enumerate(groups)], []

    canine_gi: int | None = None
    for gi, g in enumerate(groups):
        for ii in g:
            if int(sub_labels[int(ii)]) == int(canine_label):
                canine_gi = int(gi)
                break
        if canine_gi is not None:
            break

    if canine_gi is None:
        return [(i + 1, g) for i, g in enumerate(groups)], []

    before = groups[: int(canine_gi)]
    canine_g = groups[int(canine_gi)]
    after = groups[int(canine_gi) + 1 :]

    positioned: list[tuple[int, list[int]]] = []
    extras: list[list[int]] = []

    if len(before) >= 1:
        positioned.append((1, before[0]))
    if len(before) >= 2:
        positioned.append((2, before[1]))
    if len(before) > 2:
        extras.extend(before[2:])

    positioned.append((3, canine_g))
    for k, g in enumerate(after, start=4):
        positioned.append((int(k), g))

    return positioned, extras


def _estimate_x0_by_dsc_symmetry(
    label_img: sitk.Image,
    *,
    labels: list[int],
    centroid_x_phys: np.ndarray,
    max_iter: int = 3,
    min_pair_dsc: float = 0.08,
) -> float | None:
    """
    Estimate midline x0 (physical) using bilateral mask similarity (Dice).

    Workflow:
    - Start from a robust initial x0 based on centroid x.
    - Iteratively mirror each "left" tooth mask across x0 (index space, requires identity direction),
      match to the best "right" tooth by DSC, and update x0 as a DSC-weighted mean of matched centroid midpoints.

    Returns:
    - x0 in physical coordinates if estimation is reliable, else None.
    """
    if not labels or len(labels) < 4:
        return None
    if not _direction_is_identity(label_img):
        return None

    arr = sitk.GetArrayFromImage(label_img).astype(np.int32, copy=False)
    zdim, ydim, xdim = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))

    origin = np.asarray(label_img.GetOrigin(), dtype=float).reshape(3)
    spacing = np.asarray(label_img.GetSpacing(), dtype=float).reshape(3)
    sx = float(spacing[0])
    if sx <= 0:
        return None

    bbs = _compute_label_bounding_boxes_ijk(label_img, labels)
    if not bbs:
        return None

    coords_by_label: dict[int, np.ndarray] = {}
    size_by_label: dict[int, int] = {}
    for l in labels:
        ll = int(l)
        bb = bbs.get(ll)
        if bb is None:
            continue
        x0, y0, z0, sx_bb, sy_bb, sz_bb = bb
        if sx_bb <= 0 or sy_bb <= 0 or sz_bb <= 0:
            continue
        sub = arr[z0 : z0 + sz_bb, y0 : y0 + sy_bb, x0 : x0 + sx_bb]
        loc = np.argwhere(sub == ll)
        if loc.size == 0:
            continue
        ijk = loc.astype(np.int32, copy=False)
        ijk[:, 0] += int(z0)
        ijk[:, 1] += int(y0)
        ijk[:, 2] += int(x0)
        coords_by_label[ll] = ijk
        size_by_label[ll] = int(ijk.shape[0])

    if len(coords_by_label) < 4:
        return None

    cx = np.asarray(centroid_x_phys, dtype=float).reshape(-1)
    if cx.size != len(labels):
        return None

    cx_idx = (cx - float(origin[0])) / sx
    x0_idx = float(np.median(cx_idx))

    for _ in range(int(max_iter)):
        left = [int(l) for l, cxi in zip(labels, cx_idx) if float(cxi) > float(x0_idx)]
        right = [int(l) for l, cxi in zip(labels, cx_idx) if float(cxi) < float(x0_idx)]
        if len(left) < 2 or len(right) < 2:
            break

        weights = []
        x0_candidates = []

        right_set = set(int(r) for r in right)

        for l in left:
            vox = coords_by_label.get(int(l))
            if vox is None:
                continue
            xm = np.rint(2.0 * float(x0_idx) - vox[:, 2].astype(np.float32)).astype(np.int32, copy=False)
            valid = (xm >= 0) & (xm < xdim)
            if not bool(np.any(valid)):
                continue
            zz = vox[valid, 0]
            yy = vox[valid, 1]
            xx = xm[valid]
            vals = arr[zz, yy, xx]
            uniq, cnt = np.unique(vals, return_counts=True)
            inter = {int(u): int(c) for u, c in zip(uniq.tolist(), cnt.tolist()) if int(u) in right_set}
            if not inter:
                continue

            size_l = float(size_by_label.get(int(l), int(vox.shape[0])))
            best_r = None
            best_dsc = 0.0
            for r, inter_sz in inter.items():
                size_r = float(size_by_label.get(int(r), 1))
                dsc = (2.0 * float(inter_sz)) / max(1e-6, (size_l + size_r))
                if dsc > best_dsc:
                    best_dsc = float(dsc)
                    best_r = int(r)

            if best_r is None or best_dsc < float(min_pair_dsc):
                continue

            i_l = int(labels.index(int(l)))
            i_r = int(labels.index(int(best_r)))
            cand = 0.5 * (float(cx_idx[i_l]) + float(cx_idx[i_r]))
            w = float(best_dsc) * float(np.sqrt(max(1.0, size_l * float(size_by_label.get(int(best_r), 1)))))
            weights.append(w)
            x0_candidates.append(cand)

        if not weights:
            break

        wsum = float(np.sum(weights))
        if wsum <= 1e-6:
            break
        x0_new = float(np.sum(np.asarray(weights) * np.asarray(x0_candidates)) / wsum)
        if abs(x0_new - x0_idx) < 1e-3:
            x0_idx = x0_new
            break
        x0_idx = x0_new

    return float(origin[0] + sx * float(x0_idx))


def _estimate_root_component_count_in_apical_third(
    label_arr_zyx: np.ndarray,
    *,
    bb_ijk: tuple[int, int, int, int, int, int],
    label_value: int,
    k_xyz: np.ndarray,
    const: float,
    q: float = 0.70,
    opening_radius: int = 1,
) -> int:
    """
    Estimate root count by counting connected components in the apical (root-most) part of a tooth mask.

    - bb_ijk is (x, y, z, size_x, size_y, size_z) in index space.
    - k_xyz and const define the signed distance to a reference plane:
        dist = const + kx*x + ky*y + kz*z
      where (x,y,z) are voxel indices.
    - We keep voxels with dist >= quantile(dist, q) (more rootward for maxilla).
    """
    x0, y0, z0, sx, sy, sz = [int(v) for v in bb_ijk]
    if sx <= 0 or sy <= 0 or sz <= 0:
        return 1

    sub = label_arr_zyx[z0 : z0 + sz, y0 : y0 + sy, x0 : x0 + sx]
    m = (sub == int(label_value))
    n = int(np.count_nonzero(m))
    if n < 50:
        return 1

    coords = np.argwhere(m)
    gx = coords[:, 2].astype(np.float32) + float(x0)
    gy = coords[:, 1].astype(np.float32) + float(y0)
    gz = coords[:, 0].astype(np.float32) + float(z0)

    k = np.asarray(k_xyz, dtype=np.float32).reshape(3)
    dist = float(const) + k[0] * gx + k[1] * gy + k[2] * gz
    if dist.size < 10 or float(np.ptp(dist)) < 1e-6:
        return 1

    thr = float(np.quantile(dist, float(q)))
    keep = dist >= thr
    if int(np.count_nonzero(keep)) < 30:
        return 1

    ap = np.zeros(sub.shape, dtype=np.uint8)
    ap[coords[keep, 0], coords[keep, 1], coords[keep, 2]] = 1

    img = sitk.GetImageFromArray(ap)
    if int(opening_radius) > 0:
        try:
            img = sitk.BinaryMorphologicalOpening(img, [int(opening_radius)] * 3)
        except Exception:
            pass

    cc_f = sitk.ConnectedComponentImageFilter()
    try:
        cc_f.SetFullyConnected(True)
    except Exception:
        pass
    cc = cc_f.Execute(img)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    comps = [int(l) for l in stats.GetLabels() if int(l) != 0]
    if not comps:
        return 1

    min_comp = max(30, int(0.05 * float(np.count_nonzero(ap))))
    kept = 0
    for l in comps:
        if int(stats.GetNumberOfPixels(int(l))) >= int(min_comp):
            kept += 1
    return int(max(1, kept))

def _kmeans_1d_two_clusters(values: np.ndarray, *, max_iter: int = 50) -> tuple[np.ndarray, float, float]:
    "Return high-cluster membership and the sorted means of two 1D clusters."
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return np.zeros((0,), dtype=bool), 0.0, 0.0

    if v.size == 1 or float(v.max() - v.min()) < 1e-6:
        m = float(v.mean())
        return np.zeros_like(v, dtype=bool), m, m

    m1, m2 = (float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75)))
    for _ in range(int(max_iter)):
        d1 = np.abs(v - m1)
        d2 = np.abs(v - m2)
        c1 = d1 <= d2
        if bool(np.all(c1)) or bool(np.all(~c1)):
            break
        nm1 = float(v[c1].mean())
        nm2 = float(v[~c1].mean())
        if abs(nm1 - m1) + abs(nm2 - m2) < 1e-6:
            m1, m2 = nm1, nm2
            break
        m1, m2 = nm1, nm2

    mean_low, mean_high = (m1, m2) if m1 <= m2 else (m2, m1)
    is_high = v > ((mean_low + mean_high) / 2.0)
    return is_high.astype(bool, copy=False), float(mean_low), float(mean_high)


def _split_upper_lower_by_centroids(centroids_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Split teeth into upper/lower jaws based on centroid geometry (orientation-robust).

    Strategy:
    - Primary split: 1D k-means on physical z (ITK/SimpleITK uses LPS; higher z is more superior).
    - Fallback split: when data is strongly tilted or z-split is unstable, also try PCA-based split
      (smallest-variance axis in 3D + 1D k-means), and pick the split that yields larger separation
      in mean physical z between the two clusters.

    Returns:
      - is_upper: bool array, shape=(n,)
      - mean_lower / mean_upper: mean physical z of the two clusters (for debugging)
    """
    pts = np.asarray(centroids_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("centroids_xyz must have shape (n, 3)")
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool), 0.0, 0.0
    if n == 1:
        return np.zeros((1,), dtype=bool), 0.0, 0.0

    z = pts[:, 2].astype(float, copy=False)

    if n == 32:
        mean = pts.mean(axis=0).astype(float, copy=False)
        p = pts - mean.reshape(1, 3)
        if float(np.linalg.norm(p)) < 1e-6 or n < 3:
            w = z.copy()
        else:
            cov = (p.T @ p) / float(max(1, n))
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            axis = vecs[:, int(order[-1])].astype(float, copy=False)
            if float(axis[2]) < 0.0:
                axis = -axis
            w = (p @ axis.reshape(3, 1)).reshape(-1)

        order = np.argsort(w, kind="mergesort")
        low = order[: n // 2]
        high = order[n // 2 :]
        mean_z_low = float(np.mean(z[low])) if low.size else float("-inf")
        mean_z_high = float(np.mean(z[high])) if high.size else float("inf")

        is_upper = np.zeros((n,), dtype=bool)
        if mean_z_high >= mean_z_low:
            is_upper[high] = True
        else:
            is_upper[low] = True

        mean_upper = float(np.mean(z[is_upper])) if bool(np.any(is_upper)) else 0.0
        mean_lower = float(np.mean(z[~is_upper])) if bool(np.any(~is_upper)) else 0.0
        return is_upper.astype(bool, copy=False), float(mean_lower), float(mean_upper)

    is_upper_z, _z_low, _z_high = _kmeans_1d_two_clusters(z)
    if bool(np.any(is_upper_z)) and bool(np.any(~is_upper_z)):
        sep_z = float(np.mean(z[is_upper_z]) - np.mean(z[~is_upper_z]))
    else:
        sep_z = 0.0

    mean = pts.mean(axis=0).astype(float, copy=False)
    p = pts - mean.reshape(1, 3)
    if float(np.linalg.norm(p)) < 1e-6 or n < 3:
        is_upper = is_upper_z
    else:
        cov = (p.T @ p) / float(max(1, n))
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        axis = vecs[:, int(order[-1])].astype(float, copy=False)
        if float(axis[2]) < 0.0:
            axis = -axis
        w = (p @ axis.reshape(3, 1)).reshape(-1)
        is_high_w, _, _ = _kmeans_1d_two_clusters(w)

        z_high = float(np.mean(z[is_high_w])) if bool(np.any(is_high_w)) else float("-inf")
        z_low = float(np.mean(z[~is_high_w])) if bool(np.any(~is_high_w)) else float("inf")
        is_upper_w = is_high_w if z_high >= z_low else ~is_high_w
        sep_w = float(np.mean(z[is_upper_w]) - np.mean(z[~is_upper_w])) if (bool(np.any(is_upper_w)) and bool(np.any(~is_upper_w))) else 0.0

        is_upper = is_upper_w if sep_w > (sep_z + 1e-6) else is_upper_z

    mean_upper = float(np.mean(z[is_upper])) if bool(np.any(is_upper)) else 0.0
    mean_lower = float(np.mean(z[~is_upper])) if bool(np.any(~is_upper)) else 0.0
    return is_upper.astype(bool, copy=False), float(mean_lower), float(mean_upper)


def _pca_axes_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "Return the XY mean and PCA axes oriented toward positive X and Y."
    pts = np.asarray(xy, dtype=float)
    if pts.size == 0:
        return (
            np.array([0.0, 0.0], dtype=float),
            np.array([1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
        )

    mean_xy = pts.mean(axis=0).astype(float, copy=False)
    p = pts - mean_xy.reshape(1, 2)

    if pts.shape[0] < 3 or float(np.linalg.norm(p)) < 1e-6:
        return mean_xy, np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float)

    cov = (p.T @ p) / float(max(1, p.shape[0]))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]

    u_axis = vecs[:, int(order[0])].astype(float, copy=False)
    v_axis = vecs[:, int(order[1])].astype(float, copy=False)

    if float(np.dot(u_axis, np.array([1.0, 0.0], dtype=float))) < 0.0:
        u_axis = -u_axis
    if float(np.dot(v_axis, np.array([0.0, 1.0], dtype=float))) < 0.0:
        v_axis = -v_axis

    u_axis = u_axis / max(1e-12, float(np.linalg.norm(u_axis)))
    v_axis = v_axis - float(np.dot(v_axis, u_axis)) * u_axis
    v_axis = v_axis / max(1e-12, float(np.linalg.norm(v_axis)))

    return mean_xy, u_axis, v_axis


def _fit_arch_model_parabola(xy: np.ndarray, *, jaw: str = "unknown") -> ArchModel:
    "Fit a symmetric parabolic arch in PCA coordinates and return an ArchModel."
    pts = np.asarray(xy, dtype=float)
    mean_xy, u_axis, v_axis0 = _pca_axes_xy(pts)

    if pts.shape[0] < 3:
        return ArchModel(
            mean_xy=mean_xy,
            u_axis=u_axis,
            v_axis=v_axis0,
            u0=0.0,
            a=0.0,
            c=0.0,
        )

    jaw = str(jaw).lower()

    for attempt in range(2):
        v_axis = v_axis0 if attempt == 0 else -v_axis0
        temp_model = ArchModel(mean_xy=mean_xy, u_axis=u_axis, v_axis=v_axis, u0=0.0, a=0.0, c=0.0)
        u, v = temp_model.project_uv(pts)

        if u.size < 3 or float(np.max(u) - np.min(u)) < 1e-6:
            return ArchModel(
                mean_xy=mean_xy,
                u_axis=u_axis,
                v_axis=v_axis,
                u0=float(np.median(u)) if u.size else 0.0,
                a=0.0,
                c=float(np.median(v)) if v.size else 0.0,
            )

        candidates = _candidate_thresholds(u, max_candidates=72)
        candidates.extend([float(np.median(u)), float(np.mean(u))])
        umin, umax = float(u.min()), float(u.max())
        mid = 0.5 * (umin + umax)
        rng = max(1e-6, umax - umin)

        uniq_cands: list[float] = []
        seen = set()
        for x in candidates:
            xv = float(min(max(float(x), umin), umax))
            key = round(xv, 6)
            if key in seen:
                continue
            seen.add(key)
            uniq_cands.append(xv)

        def _solve(u0: float, uu: np.ndarray, vv: np.ndarray) -> tuple[float, float, float]:
            w = (uu - float(u0)) ** 2
            A = np.column_stack([w, np.ones_like(w)])
            sol, *_ = np.linalg.lstsq(A, vv, rcond=None)
            a = float(sol[0])
            c = float(sol[1])
            pred = A @ sol
            mse = float(np.mean((pred - vv) ** 2))
            reg = 0.02 * ((float(u0) - mid) / rng) ** 2
            return mse + reg, a, c

        best_u0 = float(np.median(u))
        best_a = 0.0
        best_c = float(np.mean(v))
        best_score: float | None = None

        for u0 in uniq_cands:
            score, a, c = _solve(float(u0), u, v)
            if best_score is None or score < best_score:
                best_score = score
                best_u0, best_a, best_c = float(u0), float(a), float(c)

        if best_a >= 0.0 or attempt == 1:
            return ArchModel(mean_xy=mean_xy, u_axis=u_axis, v_axis=v_axis, u0=float(best_u0), a=float(best_a), c=float(best_c))

    return ArchModel(mean_xy=mean_xy, u_axis=u_axis, v_axis=v_axis0, u0=0.0, a=0.0, c=0.0)


def _estimate_u0_by_symmetry(u: np.ndarray, v: np.ndarray, *, alpha_v: float = 0.25) -> float:
    "Estimate the arch midline from mirrored tooth positions, emphasizing anterior teeth."
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    n = int(u.size)
    if n == 0:
        return 0.0
    if n == 1:
        return float(u[0])

    order = np.argsort(v, kind="mergesort")
    w = np.linspace(1.0, 0.2, n, dtype=float)
    w_full = np.empty((n,), dtype=float)
    w_full[order] = w

    u_sorted = np.unique(np.sort(u))
    mids = ((u_sorted[:-1] + u_sorted[1:]) / 2.0).tolist() if u_sorted.size >= 2 else [float(u_sorted[0])]
    qs = [float(np.quantile(u, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
    candidates = mids + qs + [float(np.mean(u)), float(np.median(u)), 0.0]

    umin, umax = float(u.min()), float(u.max())
    mid = 0.5 * (umin + umax)
    rng = max(1e-6, umax - umin)

    uniq: list[float] = []
    seen = set()
    for x in candidates:
        xv = float(min(max(float(x), umin), umax))
        key = round(xv, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(xv)

    alpha_v = float(alpha_v)
    best_u0 = float(np.median(u))
    best_cost: float | None = None

    for u0 in uniq:
        cost = 0.0
        for i in range(n):
            ui = float(u[i])
            vi = float(v[i])
            wi = float(w_full[i])
            um = 2.0 * float(u0) - ui

            sign = ui - float(u0)
            if abs(sign) < 1e-6:
                mask = np.ones((n,), dtype=bool)
                mask[i] = False
            else:
                mask = ((u - float(u0)) * sign) <= 0.0
                mask[i] = False
                if not bool(np.any(mask)):
                    mask = np.ones((n,), dtype=bool)
                    mask[i] = False

            du = (um - u[mask]) ** 2
            dv = (vi - v[mask]) ** 2
            d2 = du + alpha_v * dv
            cost += wi * float(np.min(d2))

        cost += 0.03 * ((float(u0) - mid) / rng) ** 2

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_u0 = float(u0)

    return float(best_u0)


def _fit_parabola_fixed_u0(u: np.ndarray, v: np.ndarray, u0: float) -> tuple[float, float]:
    w = (np.asarray(u, dtype=float) - float(u0)) ** 2
    A = np.column_stack([w, np.ones_like(w)])
    sol, *_ = np.linalg.lstsq(A, np.asarray(v, dtype=float), rcond=None)
    return float(sol[0]), float(sol[1])


def _estimate_u0_by_lr_kmeans(u: np.ndarray) -> float:
    "Estimate u0 as the midpoint of the left and right cluster means."
    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size == 0:
        return 0.0
    if u.size < 2 or float(np.ptp(u)) < 1e-6:
        return float(np.median(u))
    _, mu_low, mu_high = _kmeans_1d_two_clusters(u)
    return float((mu_low + mu_high) / 2.0)


def _estimate_x0_by_kmeans(x: np.ndarray) -> float:
    "Estimate the left/right X threshold from two 1D cluster means."
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    if x.size < 2 or float(np.ptp(x)) < 1e-6:
        return float(np.median(x))
    _, mx_low, mx_high = _kmeans_1d_two_clusters(x)
    return float((mx_low + mx_high) / 2.0)


def _order_within_quadrant_from_curve_vertex(labels: list[int], centroids: np.ndarray, *, model: ArchModel) -> list[int]:
    if len(labels) <= 1:
        return labels[:]

    xy = np.asarray(centroids[:, :2], dtype=float)
    u, v = model.project_uv(xy)
    u_proj = model.project_u_on_curve(u, v)
    dist = model.arc_length_from_vertex(u_proj)
    du = np.abs(u_proj - float(model.u0))
    y = xy[:, 1]

    def _norm(vv: np.ndarray) -> np.ndarray:
        vmin = float(vv.min())
        vmax = float(vv.max())
        if vmax - vmin < 1e-6:
            return np.zeros_like(vv, dtype=float)
        return (vv - vmin) / (vmax - vmin)

    score = dist + 0.002 * _norm(du) + 0.001 * _norm(y)
    order_idx = np.argsort(score, kind="mergesort")
    return [labels[int(i)] for i in order_idx]


def _fit_plane_svd(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    "Return a fitted plane origin and a normal oriented toward positive Z."
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must have shape (n, 3)")

    origin = pts.mean(axis=0) if pts.size else np.array([0.0, 0.0, 0.0], dtype=float)
    if pts.shape[0] < 3:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        return origin.astype(float, copy=False), normal

    p = pts - origin.reshape(1, 3)
    if float(np.linalg.norm(p)) < 1e-6:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        return origin.astype(float, copy=False), normal

    _, _, vh = np.linalg.svd(p, full_matrices=False)
    normal = vh[-1, :].astype(float, copy=False)
    n = float(np.linalg.norm(normal))
    if n < 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        return origin.astype(float, copy=False), normal

    normal = normal / n
    if float(normal[2]) < 0.0:
        normal = -normal
    return origin.astype(float, copy=False), normal.astype(float, copy=False)


def _signed_distance_to_plane(points_xyz: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    o = np.asarray(origin, dtype=float).reshape(1, 3)
    n = np.asarray(normal, dtype=float).reshape(3)
    return (pts - o) @ n


def _cluster_by_arc_distance(
    sorted_indices: np.ndarray,
    arc_dist: np.ndarray,
    xy: np.ndarray | None = None,
    *,
    same_pos_mm: float,
    same_xy_mm: float = 5.0,
) -> list[list[int]]:
    "Group nearby arch positions, also requiring XY proximity when coordinates are available."
    same_pos_mm = float(same_pos_mm)
    if same_pos_mm <= 0:
        raise ValueError("same_pos_mm must be > 0")

    if sorted_indices.size == 0:
        return []

    if xy is None:
        xy = np.zeros((int(arc_dist.size), 2), dtype=float)
        same_xy_mm = float("inf")

    xy = np.asarray(xy, dtype=float)
    same_xy_mm = float(same_xy_mm)
    if same_xy_mm <= 0:
        raise ValueError("same_xy_mm must be > 0")

    groups: list[list[int]] = []
    cur: list[int] = [int(sorted_indices[0])]
    cur_mean = float(arc_dist[int(sorted_indices[0])])
    cur_xy_mean = xy[int(sorted_indices[0]), :].astype(float, copy=True)

    for idx in sorted_indices[1:]:
        ii = int(idx)
        d = float(arc_dist[ii])
        dxy = float(np.linalg.norm(xy[ii, :] - cur_xy_mean))
        if abs(d - cur_mean) <= same_pos_mm and dxy <= same_xy_mm:
            cur.append(ii)
            cur_mean = float(np.mean([float(arc_dist[j]) for j in cur]))
            cur_xy_mean = np.mean(xy[np.array(cur, dtype=int), :], axis=0).astype(float, copy=False)
        else:
            groups.append(cur)
            cur = [ii]
            cur_mean = d
            cur_xy_mean = xy[ii, :].astype(float, copy=True)

    groups.append(cur)
    return groups


def _extra_label(old_label: int, *, extras_policy: str, extras_offset: int) -> int:
    "Map unmatched instances to background, original labels, or extras_offset + old_label."
    p = str(extras_policy)
    if p == "background":
        return 0
    if p == "keep":
        return int(old_label)
    if p == "offset":
        return int(extras_offset) + int(old_label)
    raise ValueError(f"Unknown extras_policy={extras_policy}. Supported: offset / keep / background")


def build_label_to_fdi_mapping(
    label_img: sitk.Image,
    *,
    swap_left_right: bool = False,
    swap_upper_lower: bool = False,
    max_teeth_per_quadrant: int = 8,
    extras_policy: str = "offset",
    extras_offset: int = 1000,
    verbose: bool = False,
) -> dict[int, int]:
    labels, centroids = _compute_centroids_physical(label_img)
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: 0}

    max_old = int(max(labels))
    safe_extras_offset = int(extras_offset)
    if str(extras_policy) == "offset" and safe_extras_offset + max_old > 65535:
        safe_extras_offset = 0
        extras_policy = "keep"

    vol_by_label = _compute_label_physical_sizes_mm3(label_img, labels)
    vols = np.array([float(vol_by_label.get(int(l), 0.0)) for l in labels], dtype=float)
    med = float(np.median(vols[vols > 0])) if bool(np.any(vols > 0)) else float(np.median(vols))
    min_keep = float(max(30.0, 0.01 * max(1.0, med)))
    keep = vols >= min_keep
    if int(np.count_nonzero(keep)) < 2:
        keep[:] = True

    kept_labels = np.array([int(labels[i]) for i in range(len(labels)) if bool(keep[i])], dtype=int)
    kept_centroids = centroids[keep, :].astype(float, copy=False)
    kept_vols = vols[keep].astype(float, copy=False)

    if kept_labels.size < 2:
        return {int(labels[0]): 0}

    jaw_is_upper, w_low, w_high = _split_upper_lower_by_centroids(kept_centroids)
    jaw = np.where(jaw_is_upper, "upper", "lower")
    if swap_upper_lower:
        jaw = np.where(jaw == "upper", "lower", "upper")

    if verbose:
        print(f"  jaw z means: lower≈{w_low:.2f} upper≈{w_high:.2f}")

    mapping: dict[int, int] = {int(l): 0 for l in labels}

    for j in ("upper", "lower"):
        j_mask = (jaw == j)
        if not bool(np.any(j_mask)):
            continue

        jaw_labels = kept_labels[j_mask]
        jaw_centroids = kept_centroids[j_mask, :]
        jaw_xy = jaw_centroids[:, :2].astype(float, copy=False)

        mean_xy, u_axis, v_axis0 = _pca_axes_xy(jaw_xy)

        model: ArchModel | None = None
        for attempt in range(2):
            v_axis = v_axis0 if attempt == 0 else -v_axis0
            tmp = ArchModel(mean_xy=mean_xy, u_axis=u_axis, v_axis=v_axis, u0=0.0, a=0.0, c=0.0)
            u, v = tmp.project_uv(jaw_xy)

            u0 = _estimate_u0_by_lr_kmeans(u)
            if u.size >= 6:
                try:
                    u0_sym = _estimate_u0_by_symmetry(u, v, alpha_v=0.20)
                    if abs(float(u0_sym) - float(u0)) <= 0.35 * float(np.ptp(u)):
                        u0 = 0.5 * (float(u0) + float(u0_sym))
                except Exception:
                    pass

            a, c = _fit_parabola_fixed_u0(u, v, float(u0))
            if float(a) >= 0.0 or attempt == 1:
                model = ArchModel(mean_xy=mean_xy, u_axis=u_axis, v_axis=v_axis, u0=float(u0), a=float(a), c=float(c))
                break

        if model is None:
            continue

        u, v = model.project_uv(jaw_xy)
        u_proj = model.project_u_on_curve(u, v)
        arc_dist = model.arc_length_from_vertex(u_proj)

        is_right0 = u_proj < float(model.u0)
        n_jaw = int(u_proj.size)
        n_r0 = int(np.count_nonzero(is_right0))
        n_l0 = int(n_jaw - n_r0)
        use_balanced = False
        if n_jaw >= 6 and (n_jaw % 2 == 0):
            if n_jaw >= 12:
                use_balanced = True
            elif abs(int(n_r0) - int(n_l0)) <= 2:
                use_balanced = True

        if use_balanced:
            order_u = np.argsort(u_proj, kind="mergesort")
            is_right = np.zeros((n_jaw,), dtype=bool)
            is_right[order_u[: n_jaw // 2]] = True
        else:
            is_right = is_right0

        if swap_left_right:
            is_right = ~is_right

        for side_name, side_mask in (("right", is_right), ("left", ~is_right)):
            if not bool(np.any(side_mask)):
                continue

            side_labels = jaw_labels[side_mask]
            side_xy = jaw_xy[side_mask, :].astype(float, copy=False)
            side_dist = arc_dist[side_mask].astype(float, copy=False)

            sorted_idx = np.argsort(side_dist, kind="mergesort")
            if sorted_idx.size >= 2:
                d_sorted = side_dist[sorted_idx]
                dd = np.diff(d_sorted)
                dd = dd[dd > 1e-3]
                med_dd = float(np.median(dd)) if dd.size else 5.0
                same_pos_mm = float(min(3.5, max(1.0, 0.45 * med_dd)))

                dxy = np.linalg.norm(np.diff(side_xy[sorted_idx, :], axis=0), axis=1)
                dxy = dxy[dxy > 1e-3]
                med_dxy = float(np.median(dxy)) if dxy.size else 5.0
                same_xy_mm = float(min(6.0, max(3.0, 0.60 * med_dxy)))
            else:
                same_pos_mm = 2.5
                same_xy_mm = 4.0

            groups = _cluster_by_arc_distance(
                sorted_idx,
                side_dist,
                side_xy,
                same_pos_mm=float(same_pos_mm),
                same_xy_mm=float(same_xy_mm),
            )

            quad = int(FDI_QUADRANTS[(str(j), str(side_name))])
            for pos_eff, g in enumerate(groups, start=1):
                if int(pos_eff) > int(max_teeth_per_quadrant):
                    for ii in g:
                        old = int(side_labels[int(ii)])
                        mapping[int(old)] = _extra_label(int(old), extras_policy=str(extras_policy), extras_offset=int(safe_extras_offset))
                    continue

                fdi = int(quad) * 10 + int(pos_eff)
                for ii in g:
                    old = int(side_labels[int(ii)])
                    mapping[int(old)] = int(fdi)

    return mapping


def _primary_fdi_from_permanent(permanent_fdi: int) -> int | None:
    quad = int(permanent_fdi) // 10
    pos = int(permanent_fdi) % 10
    if quad not in (1, 2, 3, 4) or pos < 1 or pos > 8:
        return None
    if pos > 5:
        return None
    return (quad + 4) * 10 + pos


def build_label_to_fdi_mapping_with_primary(
    label_img: sitk.Image,
    *,
    swap_left_right: bool = False,
    swap_upper_lower: bool = False,
    max_permanent_per_quadrant: int = 8,
    max_primary_per_quadrant: int = 5,
    same_position_arc_mm: float = 3.5,
    same_position_xy_mm: float = 5.0,
    primary_min_plane_sep_mm: float = 1.5,
    extras_policy: str = "offset",
    extras_offset: int = 1000,
    verbose: bool = False,
) -> dict[int, int]:
    labels, centroids = _compute_centroids_physical(label_img)
    if not labels:
        return {}

    if len(labels) == 1:
        return {int(labels[0]): 0}

    if int(len(labels)) <= 32:
        return build_label_to_fdi_mapping(
            label_img,
            swap_left_right=bool(swap_left_right),
            swap_upper_lower=bool(swap_upper_lower),
            max_teeth_per_quadrant=int(max_permanent_per_quadrant),
            extras_policy=str(extras_policy),
            extras_offset=int(extras_offset),
            verbose=bool(verbose),
        )

    max_old = int(max(labels))
    safe_extras_offset = int(extras_offset)
    if str(extras_policy) == "offset" and safe_extras_offset + max_old > 65535:
        safe_extras_offset = 0
        extras_policy = "keep"

    xs = centroids[:, 0].astype(float, copy=False)
    ys = centroids[:, 1].astype(float, copy=False)
    zs = centroids[:, 2].astype(float, copy=False)

    jaw_is_upper, z_low, z_high = _kmeans_1d_two_clusters(zs)
    jaw = np.where(jaw_is_upper, "upper", "lower")
    if swap_upper_lower:
        jaw = np.where(jaw == "upper", "lower", "upper")

    if verbose:
        print(f"  z cluster means: lower≈{z_low:.2f} upper≈{z_high:.2f}")

    root_score = np.zeros((len(labels),), dtype=float)
    perm_like = np.ones((len(labels),), dtype=bool)
    plane_coeff_by_jaw: dict[str, tuple[np.ndarray, float]] = {}
    bbs_all = _compute_label_bounding_boxes_ijk(label_img, labels)
    img_origin = np.asarray(label_img.GetOrigin(), dtype=float).reshape(3)
    spacing = np.asarray(label_img.GetSpacing(), dtype=float).reshape(3)
    direction = np.asarray(label_img.GetDirection(), dtype=float).reshape(3, 3)
    A = direction @ np.diag(spacing)

    for j in ("upper", "lower"):
        j_idx = np.where(jaw == j)[0]
        if j_idx.size == 0:
            continue

        origin, normal = _fit_plane_svd(centroids[j_idx, :])
        dist = _signed_distance_to_plane(centroids[j_idx, :], origin, normal)
        score = dist if str(j) == "upper" else -dist
        root_score[j_idx] = score
        k = (A.T @ normal.reshape(3)).astype(float, copy=False)
        const = float(np.dot(normal.reshape(3), (img_origin - origin.reshape(3))))
        plane_coeff_by_jaw[str(j)] = (k, const)

        if j_idx.size >= 2 and float(np.ptp(score)) >= 1e-3:
            is_high, _, _ = _kmeans_1d_two_clusters(score)
            perm_like[j_idx] = is_high
        else:
            perm_like[j_idx] = True

        if verbose:
            n = float(np.linalg.norm(normal))
            print(f"  {j} occlusal plane: normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}) |n|={n:.3f}")
            print(f"  {j} root_score range: [{float(score.min()):.2f},{float(score.max()):.2f}]")

    bbox_ext = _compute_label_bbox_extents_mm(label_img, labels)
    length_z_mm = np.array([float(bbox_ext.get(int(l), np.zeros((3,), dtype=float))[2]) for l in labels], dtype=float)
    phys_sizes = _compute_label_physical_sizes_mm3(label_img, labels)
    size_mm3 = np.array([float(phys_sizes.get(int(l), 0.0)) for l in labels], dtype=float)

    mapping: dict[int, int] = {}

    for j in ("upper", "lower"):
        j_idx = np.where(jaw == j)[0]
        if j_idx.size == 0:
            continue

        xy_jaw = np.column_stack([xs[j_idx], ys[j_idx]]).astype(float, copy=False)
        mean_xy, u_axis, _v_axis = _pca_axes_xy(xy_jaw)
        u = ((xy_jaw - mean_xy.reshape(1, 2)) @ u_axis.reshape(2, 1)).reshape(-1)
        order = np.argsort(u, kind="mergesort")
        n = int(order.size)
        n_right = n // 2
        right_loc = order[:n_right]
        left_loc = order[n_right:]
        if swap_left_right:
            right_loc, left_loc = left_loc, right_loc

        if n_right > 0 and n_right < n:
            u_boundary = 0.5 * (float(u[int(order[n_right - 1])]) + float(u[int(order[n_right])]))
        else:
            u_boundary = float(np.median(u))

        if str(j) == "upper":
            sides = [(1, right_loc), (2, left_loc)]
        else:
            sides = [(4, right_loc), (3, left_loc)]

        for perm_quad, side_loc in sides:
            if side_loc.size == 0:
                continue

            pos_dist = np.abs(u[side_loc] - float(u_boundary)).astype(float, copy=False)
            sorted_local = np.argsort(pos_dist, kind="mergesort")

            sub_idx_global = j_idx[side_loc]
            sub_labels = [labels[int(i)] for i in sub_idx_global]
            sub_centroids = centroids[sub_idx_global, :]
            sub_xy = np.asarray(sub_centroids[:, :2], dtype=float)

            groups = _cluster_by_arc_distance(
                sorted_local,
                pos_dist,
                sub_xy,
                same_pos_mm=float(same_position_arc_mm),
                same_xy_mm=float(same_position_xy_mm),
            )

            root_local = root_score[sub_idx_global]
            perm_like_local = perm_like[sub_idx_global]

            def _select_perm_index(group_indices: list[int], pos_eff: int) -> int:
                """
                group_indices are indices into sub_labels/sub_centroids arrays.
                """
                if len(group_indices) == 1:
                    return int(group_indices[0])
                if int(pos_eff) > int(max_primary_per_quadrant):
                    return int(
                        max(
                            (int(ii) for ii in group_indices),
                            key=lambda k: (
                                float(size_mm3[int(sub_idx_global[int(k)])]),
                                float(length_z_mm[int(sub_idx_global[int(k)])]),
                                float(root_local[int(k)]),
                            ),
                        )
                    )
                perm_candidates = [int(ii) for ii in group_indices if bool(perm_like_local[int(ii)])]
                if perm_candidates:
                    return int(
                        max(
                            perm_candidates,
                            key=lambda k: (
                                float(root_local[int(k)]),
                                float(size_mm3[int(sub_idx_global[int(k)])]),
                                float(length_z_mm[int(sub_idx_global[int(k)])]),
                            ),
                        )
                    )
                return int(
                    max(
                        (int(ii) for ii in group_indices),
                        key=lambda k: (
                            float(root_local[int(k)]),
                            float(size_mm3[int(sub_idx_global[int(k)])]),
                            float(length_z_mm[int(sub_idx_global[int(k)])]),
                        ),
                    )
                )
            for pos_eff, g0 in enumerate(groups, start=1):
                if int(pos_eff) > int(max_permanent_per_quadrant):
                    for ii in g0:
                        old_lbl = int(sub_labels[int(ii)])
                        mapping[old_lbl] = _extra_label(old_lbl, extras_policy=str(extras_policy), extras_offset=int(safe_extras_offset))
                    continue

                g_list = [int(ii) for ii in g0]
                perm_i = _select_perm_index(g_list, int(pos_eff))
                perm_lbl = int(sub_labels[int(perm_i)])
                mapping[perm_lbl] = int(int(perm_quad) * 10 + int(pos_eff))

                others = [int(ii) for ii in g_list if int(ii) != int(perm_i)]
                if not others:
                    continue

                if int(pos_eff) <= int(max_primary_per_quadrant):
                    pri_i = int(min(others, key=lambda k: float(root_local[int(k)])))
                    sep = float(root_local[int(perm_i)] - root_local[int(pri_i)])
                    if sep >= float(primary_min_plane_sep_mm):
                        pri_lbl = int(sub_labels[int(pri_i)])
                        mapping[pri_lbl] = int((int(perm_quad) + 4) * 10 + int(pos_eff))
                        for ii in others:
                            if int(ii) != int(pri_i):
                                old_lbl = int(sub_labels[int(ii)])
                                mapping[old_lbl] = _extra_label(old_lbl, extras_policy=str(extras_policy), extras_offset=int(safe_extras_offset))
                    else:
                        for ii in others:
                            old_lbl = int(sub_labels[int(ii)])
                            mapping[old_lbl] = _extra_label(old_lbl, extras_policy=str(extras_policy), extras_offset=int(safe_extras_offset))
                else:
                    for ii in others:
                        old_lbl = int(sub_labels[int(ii)])
                        mapping[old_lbl] = _extra_label(old_lbl, extras_policy=str(extras_policy), extras_offset=int(safe_extras_offset))

    for old in labels:
        mapping.setdefault(int(old), 0)
    return mapping


def apply_mapping_to_image(label_img: sitk.Image, mapping: dict[int, int]) -> sitk.Image:
    if not mapping:
        pid = int(label_img.GetPixelIDValue())
        if pid in (
            sitk.sitkUInt8,
            sitk.sitkUInt16,
            sitk.sitkUInt32,
            sitk.sitkUInt64,
            sitk.sitkInt8,
            sitk.sitkInt16,
            sitk.sitkInt32,
            sitk.sitkInt64,
        ):
            return sitk.Cast(label_img, sitk.sitkUInt16)
        return sitk.Cast(sitk.Round(label_img), sitk.sitkUInt16)

    arr = sitk.GetArrayFromImage(label_img)
    arr_i = np.rint(arr).astype(np.int32, copy=False)
    arr_i[arr_i < 0] = 0
    max_label = int(arr_i.max()) if arr_i.size else 0

    lut = np.zeros((max_label + 1,), dtype=np.uint16)
    for old, new in mapping.items():
        if 0 <= int(old) <= max_label:
            lut[int(old)] = np.uint16(int(new))

    out_arr = lut[arr_i].astype(np.uint16, copy=False)
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(label_img)
    return out_img


def _compute_pad_voxels(spacing_xyz: tuple[float, float, float], pad_mm: float) -> tuple[int, int, int]:
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    rx = max(2, int(round(pad_mm / max(sx, 1e-6))))
    ry = max(2, int(round(pad_mm / max(sy, 1e-6))))
    rz = max(2, int(round(pad_mm / max(sz, 1e-6))))
    return rx, ry, rz


def smooth_multilabel_antialias_roi(
    label_img: sitk.Image,
    *,
    maximum_rms_error: float = 0.05,
    max_iterations: int = 80,
    roi_pad_mm: float = 3.0,
    protect_labels: bool = True,
    verbose: bool = False,
) -> sitk.Image:
    lab = sitk.Cast(sitk.Round(sitk.Cast(label_img, sitk.sitkFloat32)), sitk.sitkUInt16)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lab)

    labels = [int(l) for l in stats.GetLabels() if int(l) != 0]
    if not labels:
        out = sitk.Image(lab.GetSize(), sitk.sitkUInt16)
        out.CopyInformation(lab)
        return out

    labels.sort(key=lambda l: stats.GetNumberOfPixels(int(l)), reverse=True)

    size_xyz = lab.GetSize()
    spacing_xyz = lab.GetSpacing()
    pad_x, pad_y, pad_z = _compute_pad_voxels(spacing_xyz, roi_pad_mm)

    arr_shape_zyx = (size_xyz[2], size_xyz[1], size_xyz[0])
    max_phi = np.full(arr_shape_zyx, -1.0e4, dtype=np.float16)
    out_arr = np.zeros(arr_shape_zyx, dtype=np.uint16)

    for idx, lbl in enumerate(labels, start=1):
        bb = stats.GetBoundingBox(int(lbl))
        x0, y0, z0, sx, sy, sz = [int(v) for v in bb]

        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rz0 = max(0, z0 - pad_z)
        rx1 = min(size_xyz[0], x0 + sx + pad_x)
        ry1 = min(size_xyz[1], y0 + sy + pad_y)
        rz1 = min(size_xyz[2], z0 + sz + pad_z)

        roi_index = (rx0, ry0, rz0)
        roi_size = (rx1 - rx0, ry1 - ry0, rz1 - rz0)
        if roi_size[0] <= 0 or roi_size[1] <= 0 or roi_size[2] <= 0:
            continue

        if verbose:
            nvox = int(stats.GetNumberOfPixels(int(lbl)))
            print(f"  [{idx}/{len(labels)}] label {lbl} voxels={nvox} ROI={roi_size} start={roi_index}", flush=True)

        roi_lab = sitk.RegionOfInterest(lab, roi_size, roi_index)
        roi_mask = sitk.Equal(roi_lab, int(lbl))
        phi = sitk.AntiAliasBinary(
            roi_mask,
            maximumRMSError=float(maximum_rms_error),
            numberOfIterations=int(max_iterations),
        )
        phi = sitk.Cast(phi, sitk.sitkFloat32)
        phi_arr = sitk.GetArrayFromImage(phi)

        z0s, y0s, x0s = roi_index[2], roi_index[1], roi_index[0]
        z1s, y1s, x1s = z0s + roi_size[2], y0s + roi_size[1], x0s + roi_size[0]

        max_slice = max_phi[z0s:z1s, y0s:y1s, x0s:x1s]
        out_slice = out_arr[z0s:z1s, y0s:y1s, x0s:x1s]

        better = phi_arr > max_slice
        if np.any(better):
            max_slice[better] = phi_arr[better].astype(np.float16, copy=False)
            out_slice[better] = np.uint16(int(lbl))

        max_phi[z0s:z1s, y0s:y1s, x0s:x1s] = max_slice
        out_arr[z0s:z1s, y0s:y1s, x0s:x1s] = out_slice

    out_arr[max_phi <= 0] = 0

    if protect_labels:
        in_arr = sitk.GetArrayFromImage(lab).astype(np.uint16, copy=False)
        locked = in_arr != 0
        if np.any(locked):
            out_arr[locked] = in_arr[locked]

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(lab)
    return out


def smooth_multilabel_sdf_gaussian_roi(
    label_img: sitk.Image,
    *,
    sigma_mm: float = 0.8,
    volume_preserve: bool = True,
    preserve_foreground: bool = True,
    protect_labels: bool = True,
    roi_pad_mm: float = 4.0,
    verbose: bool = False,
) -> sitk.Image:
    if sigma_mm <= 0:
        raise ValueError("sigma_mm must be > 0")

    lab = sitk.Cast(sitk.Round(sitk.Cast(label_img, sitk.sitkFloat32)), sitk.sitkUInt16)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lab)

    labels = [int(l) for l in stats.GetLabels() if int(l) != 0]
    if not labels:
        out = sitk.Image(lab.GetSize(), sitk.sitkUInt16)
        out.CopyInformation(lab)
        return out

    labels.sort(key=lambda l: stats.GetNumberOfPixels(int(l)), reverse=True)

    size_xyz = lab.GetSize()
    spacing_xyz = lab.GetSpacing()
    roi_pad_mm = float(max(roi_pad_mm, 3.0 * float(sigma_mm), 2.0 * float(max(spacing_xyz))))
    pad_x, pad_y, pad_z = _compute_pad_voxels(spacing_xyz, roi_pad_mm)

    arr_shape_zyx = (size_xyz[2], size_xyz[1], size_xyz[0])
    max_phi = np.full(arr_shape_zyx, -1.0e4, dtype=np.float16)
    out_arr = np.zeros(arr_shape_zyx, dtype=np.uint16)

    for idx, lbl in enumerate(labels, start=1):
        bb = stats.GetBoundingBox(int(lbl))
        x0, y0, z0, sx, sy, sz = [int(v) for v in bb]

        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rz0 = max(0, z0 - pad_z)
        rx1 = min(size_xyz[0], x0 + sx + pad_x)
        ry1 = min(size_xyz[1], y0 + sy + pad_y)
        rz1 = min(size_xyz[2], z0 + sz + pad_z)

        roi_index = (rx0, ry0, rz0)
        roi_size = (rx1 - rx0, ry1 - ry0, rz1 - rz0)
        if roi_size[0] <= 0 or roi_size[1] <= 0 or roi_size[2] <= 0:
            continue

        if verbose:
            nvox = int(stats.GetNumberOfPixels(int(lbl)))
            print(f"  [{idx}/{len(labels)}] label {lbl} voxels={nvox} ROI={roi_size} start={roi_index}", flush=True)

        roi_lab = sitk.RegionOfInterest(lab, roi_size, roi_index)
        roi_mask = sitk.Equal(roi_lab, int(lbl))

        dist = sitk.SignedMaurerDistanceMap(
            roi_mask,
            insideIsPositive=True,
            squaredDistance=False,
            useImageSpacing=True,
        )
        dist = sitk.Cast(dist, sitk.sitkFloat32)
        dist_s = sitk.SmoothingRecursiveGaussian(dist, float(sigma_mm))
        dist_s = sitk.Cast(dist_s, sitk.sitkFloat32)
        phi_arr = sitk.GetArrayFromImage(dist_s)

        if volume_preserve:
            roi_mask_arr = sitk.GetArrayFromImage(roi_mask).astype(bool, copy=False)
            target_vox = int(roi_mask_arr.sum())
            total_vox = int(phi_arr.size)
            if 0 < target_vox < total_vox:
                k = total_vox - target_vox
                t = float(np.partition(phi_arr.reshape(-1), k)[k])
                phi_arr = phi_arr - t

        z0s, y0s, x0s = roi_index[2], roi_index[1], roi_index[0]
        z1s, y1s, x1s = z0s + roi_size[2], y0s + roi_size[1], x0s + roi_size[0]

        max_slice = max_phi[z0s:z1s, y0s:y1s, x0s:x1s]
        out_slice = out_arr[z0s:z1s, y0s:y1s, x0s:x1s]

        better = phi_arr > max_slice
        if np.any(better):
            max_slice[better] = phi_arr[better].astype(np.float16, copy=False)
            out_slice[better] = np.uint16(int(lbl))

        max_phi[z0s:z1s, y0s:y1s, x0s:x1s] = max_slice
        out_arr[z0s:z1s, y0s:y1s, x0s:x1s] = out_slice

    out_arr[max_phi <= 0] = 0

    if preserve_foreground:
        in_arr = sitk.GetArrayFromImage(lab).astype(np.uint16, copy=False)
        backfill = (in_arr != 0) & (out_arr == 0)
        if np.any(backfill):
            out_arr[backfill] = in_arr[backfill]

    if protect_labels:
        in_arr = sitk.GetArrayFromImage(lab).astype(np.uint16, copy=False)
        locked = in_arr != 0
        if np.any(locked):
            out_arr[locked] = in_arr[locked]

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(lab)
    return out


def relabel_only_image(
    img: sitk.Image,
    *,
    src_path: Path,
    swap_left_right: bool = False,
    swap_upper_lower: bool = False,
    extras_policy: str = "offset",
    extras_offset: int = 1000,
    verbose: bool = False,
) -> sitk.Image:
    mapping = build_label_to_fdi_mapping_with_primary(
        img,
        swap_left_right=bool(swap_left_right),
        swap_upper_lower=bool(swap_upper_lower),
        extras_policy=str(extras_policy),
        extras_offset=int(extras_offset),
        verbose=bool(verbose),
    )

    nonzero_labels = [k for k in mapping.keys() if int(k) != 0]
    if len(nonzero_labels) == 1 and int(mapping[nonzero_labels[0]]) == 0:
        inferred = _parse_fdi_from_filename(src_path)
        if inferred is not None:
            mapping[nonzero_labels[0]] = int(inferred)
        else:
            mapping[nonzero_labels[0]] = int(nonzero_labels[0])

    return apply_mapping_to_image(img, mapping)


def relabel_and_smooth_image(
    img: sitk.Image,
    *,
    src_path: Path,
    swap_left_right: bool = False,
    swap_upper_lower: bool = False,
    smooth_method: str = "sdf-gaussian",
    sigma_mm: float = 0.8,
    volume_preserve: bool = True,
    preserve_foreground: bool = True,
    protect_labels: bool = True,
    maximum_rms_error: float = 0.05,
    max_iterations: int = 80,
    roi_pad_mm: float = 4.0,
    extras_policy: str = "offset",
    extras_offset: int = 1000,
    verbose: bool = False,
) -> sitk.Image:
    relabeled = relabel_only_image(
        img,
        src_path=src_path,
        swap_left_right=bool(swap_left_right),
        swap_upper_lower=bool(swap_upper_lower),
        extras_policy=str(extras_policy),
        extras_offset=int(extras_offset),
        verbose=bool(verbose),
    )

    if str(smooth_method) == "antialias":
        return smooth_multilabel_antialias_roi(
            relabeled,
            maximum_rms_error=float(maximum_rms_error),
            max_iterations=int(max_iterations),
            roi_pad_mm=float(roi_pad_mm),
            protect_labels=bool(protect_labels),
            verbose=bool(verbose),
        )
    if str(smooth_method) == "sdf-gaussian":
        return smooth_multilabel_sdf_gaussian_roi(
            relabeled,
            sigma_mm=float(sigma_mm),
            volume_preserve=bool(volume_preserve),
            preserve_foreground=bool(preserve_foreground),
            protect_labels=bool(protect_labels),
            roi_pad_mm=float(roi_pad_mm),
            verbose=bool(verbose),
        )
    raise ValueError(f"Unknown smooth_method={smooth_method}. Supported: sdf-gaussian / antialias")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = True,
    swap_left_right: bool = False,
    swap_upper_lower: bool = False,
    smooth_method: str = "sdf-gaussian",
    sigma_mm: float = 0.8,
    volume_preserve: bool = True,
    preserve_foreground: bool = True,
    protect_labels: bool = True,
    maximum_rms_error: float = 0.05,
    max_iterations: int = 80,
    roi_pad_mm: float = 4.0,
    extras_policy: str = "offset",
    extras_offset: int = 1000,
    progress_every: int = 20,
    verbose: bool = False,
) -> None:
    files = iter_nifti_in_dir(input_dir)
    if not files:
        print(f"No nii/nii.gz found: {input_dir}", flush=True)
        return

    _safe_mkdir(output_dir)
    print(f"Found {len(files)} files:", flush=True)
    print(f"  Input:  {input_dir}", flush=True)
    print(f"  Output: {output_dir}", flush=True)

    ok = 0
    fail = 0
    skip = 0

    for i, src in enumerate(files, start=1):
        dst = output_dir / src.name
        if dst.exists() and not overwrite:
            skip += 1
            continue

        try:
            img = sitk.ReadImage(str(src))

            arr = sitk.GetArrayViewFromImage(img)
            uniq = np.unique(arr)
            if uniq.size <= 2 and float(uniq.max()) <= 1.0:
                shutil.copy2(src, dst)
                ok += 1
                continue

            out = relabel_and_smooth_image(
                img,
                src_path=src,
                swap_left_right=bool(swap_left_right),
                swap_upper_lower=bool(swap_upper_lower),
                smooth_method=str(smooth_method),
                sigma_mm=float(sigma_mm),
                volume_preserve=bool(volume_preserve),
                preserve_foreground=bool(preserve_foreground),
                protect_labels=bool(protect_labels),
                maximum_rms_error=float(maximum_rms_error),
                max_iterations=int(max_iterations),
                roi_pad_mm=float(roi_pad_mm),
                extras_policy=str(extras_policy),
                extras_offset=int(extras_offset),
                verbose=bool(verbose),
            )
            sitk.WriteImage(out, str(dst), useCompression=True)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] Failed: {src.name}. Error: {e}", flush=True)

        if progress_every and (i % int(progress_every) == 0):
            print(f"Progress [{i}/{len(files)}] ok={ok} skipped={skip} failed={fail}", flush=True)

    print("Done:", flush=True)
    print(f"  Ok:      {ok}", flush=True)
    print(f"  Skipped: {skip}", flush=True)
    print(f"  Failed:  {fail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="relabel(FDI)+smooth: relabel + boundary smoothing (root folder nii/nii.gz)")
    parser.add_argument("--input", type=str, default=r"T:\original_data\zy_CBCT\zy_images_pred", help="Input directory")
    parser.add_argument("--output", type=str, default=r"T:\original_data\zy_CBCT\zy_images_pred_relabel+smooth", help="Output directory")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing outputs (default: enabled)",
    )
    parser.add_argument("--swap-left-right", action="store_true", help="Swap left/right (use when laterality is flipped)")
    parser.add_argument("--swap-upper-lower", action="store_true", help="Swap upper/lower jaws (use when jaws are flipped)")

    parser.add_argument(
        "--smooth-method",
        type=str,
        default="sdf-gaussian",
        choices=["sdf-gaussian", "antialias"],
        help="Smoothing method: sdf-gaussian (moves boundary) or antialias (keeps discrete labels mostly unchanged)",
    )
    parser.add_argument(
        "--sigma-mm",
        type=float,
        default=0.8,
        help="Gaussian sigma (mm) for distance-field smoothing (sdf-gaussian only)",
    )
    parser.add_argument(
        "--volume-preserve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to preserve each label volume (default: enabled)",
    )
    parser.add_argument(
        "--preserve-foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not erase original foreground into background (default: enabled)",
    )
    parser.add_argument(
        "--protect-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prevent different teeth labels from overwriting each other (default: enabled)",
    )
    parser.add_argument(
        "--maximum-rms-error",
        type=float,
        default=0.05,
        help="AntiAliasBinary RMS error threshold (smaller = smoother/slower)",
    )
    parser.add_argument("--max-iterations", type=int, default=80, help="AntiAliasBinary max iterations")
    parser.add_argument("--roi-pad-mm", type=float, default=4.0, help="Per-tooth ROI padding (mm)")

    parser.add_argument(
        "--extras-policy",
        type=str,
        default="offset",
        choices=["offset", "keep", "background"],
        help="Policy when an instance tooth cannot be mapped to standard FDI position: offset/keep/background",
    )
    parser.add_argument("--extras-offset", type=int, default=1000, help="Offset value for extras-policy=offset (new=offset+old)")

    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N files (0 = disable)")
    parser.add_argument("--verbose", action="store_true", help="Print more details (can be very verbose)")

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    process_directory(
        input_dir,
        output_dir,
        overwrite=bool(args.overwrite),
        swap_left_right=bool(args.swap_left_right),
        swap_upper_lower=bool(args.swap_upper_lower),
        smooth_method=str(args.smooth_method),
        sigma_mm=float(args.sigma_mm),
        volume_preserve=bool(args.volume_preserve),
        preserve_foreground=bool(args.preserve_foreground),
        protect_labels=bool(args.protect_labels),
        maximum_rms_error=float(args.maximum_rms_error),
        max_iterations=int(args.max_iterations),
        roi_pad_mm=float(args.roi_pad_mm),
        extras_policy=str(args.extras_policy),
        extras_offset=int(args.extras_offset),
        progress_every=int(args.progress_every),
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(os.cpu_count() or 4))
    raise SystemExit(main())
