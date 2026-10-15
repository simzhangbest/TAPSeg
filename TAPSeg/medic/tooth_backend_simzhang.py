from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

_CUDA_USABLE_CACHE = None
_MODEL_CACHE = None


def _pick_weight_file(dir_path: Path, preferred: Tuple[str, ...]) -> Path:
    for name in preferred:
        p = dir_path / name
        if p.exists():
            return p

    candidates = list(dir_path.glob("iter_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No weight file found in: {dir_path}")

    def _iter_num(p: Path) -> int:
        stem = p.stem  
        try:
            return int(stem.split("_", 1)[1])
        except Exception:
            return -1

    return sorted(candidates, key=_iter_num, reverse=True)[0]


def _resolve_weights(model_dir: str) -> Tuple[str, str, str]:
    "Resolve packaged roi/cnt/sin weights or the legacy sim_model layout."
    root = Path(model_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Weights directory does not exist: {root}")

    roi_dir = root / "roi"
    if not roi_dir.exists():
        raise FileNotFoundError(f"Missing roi directory: {roi_dir}")

    cnt_dir = root / "cnt"
    if not cnt_dir.exists():
        cnt_dir = root / "skl_off"
    if not cnt_dir.exists():
        raise FileNotFoundError(f"Missing cnt/skl_off directory under: {root}")

    ins_dir = root / "sin"
    if not ins_dir.exists():
        for cand in ["inst_v1_bd_test_zy", "inst_v1_bd_test", "inst_noise"]:
            p = root / cand
            if p.exists():
                ins_dir = p
                break
    if not ins_dir.exists():
        raise FileNotFoundError(f"Missing sin/inst_* directory under: {root}")

    roi_weight = _pick_weight_file(roi_dir, ("iter_30002.pth", "iter_30000.pth"))
    cnt_weight = _pick_weight_file(cnt_dir, ("iter_60001.pth", "iter_60000.pth"))
    ins_weight = _pick_weight_file(ins_dir, ("iter_70416.pth", "iter_70000.pth"))

    return str(roi_weight), str(cnt_weight), str(ins_weight)


def _torch_device():
    import torch

    force_cpu = os.environ.get("TAPSEG_FORCE_CPU", "").strip().lower()
    if force_cpu in ("1", "true", "yes", "y", "on"):
        return torch.device("cpu")

    forced = os.environ.get("TAPSEG_DEVICE", "").strip()
    if forced:
        return torch.device(forced)

    if _is_cuda_usable():
        return torch.device("cuda:0")

    return torch.device("cpu")


def _is_cuda_usable() -> bool:
    "Probe CUDA kernel compatibility with the current GPU."
    global _CUDA_USABLE_CACHE
    if _CUDA_USABLE_CACHE is not None:
        return bool(_CUDA_USABLE_CACHE)

    try:
        import torch

        if not torch.cuda.is_available():
            _CUDA_USABLE_CACHE = False
            return False

        try:
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            arch_list = torch.cuda.get_arch_list() or []
            if arch_list and arch not in arch_list:
                pass
        except Exception:
            pass

        x = torch.zeros(1, device="cuda")
        y = x + 1  # noqa: F841
        torch.cuda.synchronize()

        _CUDA_USABLE_CACHE = True
        return True
    except Exception:
        _CUDA_USABLE_CACHE = False
        return False


def _load_model(model_dir: str, device=None):
    import torch

    from .simzhang.networks.vnet_roi import VNet_roi
    from .simzhang.networks.vnet import VNet
    from .simzhang.networks.vnet_ins_seg_bd import VNet_singleTooth

    device = device or _torch_device()
    roi_weight, cnt_weight, ins_weight = _resolve_weights(model_dir)

    global _MODEL_CACHE
    cache_key = (str(Path(model_dir).resolve()), str(device))
    if _MODEL_CACHE is not None and _MODEL_CACHE.get("key") == cache_key:
        return _MODEL_CACHE["device"], _MODEL_CACHE["net_roi"], _MODEL_CACHE["net_cnt"], _MODEL_CACHE["net_ins"]

    def _load_state_dict(net, weight_path: str):
        ckpt = torch.load(weight_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            net.load_state_dict(ckpt["model_state_dict"])
        else:
            net.load_state_dict(ckpt)

    net_roi = VNet_roi(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=False).to(device)
    _load_state_dict(net_roi, roi_weight)
    net_roi.eval()

    net_cnt = VNet(n_channels=1, n_classes=3, normalization="batchnorm", has_dropout=True).to(device)
    _load_state_dict(net_cnt, cnt_weight)
    net_cnt.eval()

    net_ins = VNet_singleTooth(n_channels=2, n_classes=2, normalization="batchnorm", has_dropout=True).to(device)
    _load_state_dict(net_ins, ins_weight)
    net_ins.eval()


    _MODEL_CACHE = {
        "key": cache_key,
        "device": device,
        "net_roi": net_roi,
        "net_cnt": net_cnt,
        "net_ins": net_ins,
    }
    return device, net_roi, net_cnt, net_ins


def _label_rescale(image_label, w_ori, h_ori, z_ori, flag, device):
    import numpy as np
    import torch

    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    if flag == "trilinear":
        teeth_ids = np.unique(image_label)
        image_label_ori = torch.zeros((w_ori, h_ori, z_ori), device=device)
        image_label_t = torch.from_numpy(image_label).to(device)
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label_t == float(teeth_ids[label_id])).float()
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(
                image_label_bn, size=(w_ori, h_ori, z_ori), mode="trilinear"
            )
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_ori[image_label_bn > 0.5] = float(teeth_ids[label_id])
        image_label = image_label_ori.detach().cpu().numpy()

    if flag == "nearest":
        image_label_t = torch.from_numpy(image_label).to(device)
        image_label_t = image_label_t[None, None, :, :, :].float()
        image_label_t = torch.nn.functional.interpolate(
            image_label_t, size=(w_ori, h_ori, z_ori), mode="nearest"
        )
        image_label = image_label_t[0, 0, :, :, :].detach().cpu().numpy()

    return image_label


def _read_data(data_path: str, device):
    import numpy as np
    import SimpleITK as sitk

    src_data_vol = sitk.ReadImage(data_path)
    images = sitk.GetArrayFromImage(src_data_vol)
    spacing = src_data_vol.GetSpacing()
    direction = src_data_vol.GetDirection()
    origin = src_data_vol.GetOrigin()
    w, h, d = images.shape

    images = _label_rescale(images, w * (spacing[0] / 0.4), h * (spacing[0] / 0.4), d * (spacing[0] / 0.4), "nearest", device)

    low_bound = np.percentile(images, 5)
    up_bound = np.percentile(images, 99.9)
    images[images < low_bound] = low_bound
    images[images > up_bound] = up_bound
    images = (images - low_bound) / (up_bound - low_bound)

    return images, spacing, direction, origin, w, h, d


def _img_crop(image_bbox):
    import numpy as np
    from skimage import morphology

    image_bbox = morphology.remove_small_objects(image_bbox.astype(bool), 2500, connectivity=3).astype(int)
    if image_bbox.sum() > 0:
        x_min = np.nonzero(image_bbox)[0].min() - 32
        x_max = np.nonzero(image_bbox)[0].max() + 32

        y_min = np.nonzero(image_bbox)[1].min() - 16
        y_max = np.nonzero(image_bbox)[1].max() + 16

        z_min = np.nonzero(image_bbox)[2].min() - 16
        z_max = np.nonzero(image_bbox)[2].max() + 16

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if z_min < 0:
            z_min = 0
        if x_max > image_bbox.shape[0]:
            x_max = image_bbox.shape[0]
        if y_max > image_bbox.shape[1]:
            y_max = image_bbox.shape[1]
        if z_max > image_bbox.shape[2]:
            z_max = image_bbox.shape[2]
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = -1, image_bbox.shape[0], 0, image_bbox.shape[1], 0, image_bbox.shape[2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def _inference(image, net_roi, net_cnt, net_ins, w, h, d, device):
    import numpy as np
    import torch

    from .simzhang.roi_dect import roi_detection
    from .simzhang.cnt_skl_dect import cnt_detection
    from .simzhang.ins_tooth_seg_v1 import ins_tooth_seg

    w1, h1, d1 = image.shape

    print("---run the roi binary segmentation.", flush=True)
    stride_xy = 224
    stride_z = 224
    patch_size_roi_stage = (256, 256, 256)
    with torch.no_grad():
        label_roi = roi_detection(net_roi, image, stride_xy, stride_z, patch_size_roi_stage, device=device)

    x_min, x_max, y_min, y_max, z_min, z_max = _img_crop(label_roi)
    if x_min == -1:
        whole_label = np.zeros((w, h, d), dtype=np.uint8)
        roi_full = _label_rescale(label_roi, w, h, d, "nearest", device)
        cnt_full = np.zeros((w, h, d), dtype=np.uint8)
        return whole_label, roi_full, cnt_full

    image_crop = image[x_min:x_max, y_min:y_max, z_min:z_max]

    print("---run the 1st stage network.", flush=True)
    stride_xy = 64
    stride_z = 64
    patch_size_1st_stage = (128, 128, 128)
    with torch.no_grad():
        centroids = cnt_detection(net_cnt, image_crop, stride_xy, stride_z, patch_size_1st_stage, device=device)

    print("---run the 2nd stege network.", flush=True)
    patch_size_2nd_stage = (96, 96, 96)
    tooth_label = ins_tooth_seg(net_ins, image_crop, centroids, patch_size_2nd_stage, device=device)

    whole_label_resample = np.zeros((w1, h1, d1), dtype=np.float32)
    whole_label_resample[x_min:x_max, y_min:y_max, z_min:z_max] = tooth_label

    roi_full = _label_rescale(label_roi, w, h, d, "nearest", device)

    cnt_map = np.zeros((w1, h1, d1), dtype=np.uint8)
    if centroids is not None and getattr(centroids, "size", 0) > 0:
        for i in range(centroids.shape[1]):
            cx, cy, cz = centroids[:, i].astype(int)
            cx = cx + x_min
            cy = cy + y_min
            cz = cz + z_min
            if 0 <= cx < w1 and 0 <= cy < h1 and 0 <= cz < d1:
                cnt_map[cx, cy, cz] = 1
    cnt_full = _label_rescale(cnt_map, w, h, d, "nearest", device)

    whole_label = _label_rescale(whole_label_resample, w, h, d, "trilinear", device)
    return whole_label, roi_full, cnt_full


def exec_math_one(sign, input_file=r".\\model\\test.nrrd", output_file=r".\\model\\pred.nrrd", model_dir=r".\\model"):
    import numpy as np
    import SimpleITK as sitk
    import torch

    device = None
    tooth_label_raw = None
    try:
        device, net_roi, net_cnt, net_ins = _load_model(model_dir)
        image, spacing, direction, origin, w, h, d = _read_data(input_file, device)
        tooth_label, roi_label, cnt_label = _inference(image, net_roi, net_cnt, net_ins, w, h, d, device)
        tooth_label_raw = tooth_label
    except RuntimeError as e:
        msg = str(e)
        if device is not None and getattr(device, "type", None) == "cuda" and (
            "no kernel image is available" in msg or "CUDA error" in msg
        ):
            print(
                "[TAPSeg] CUDA kernels are not executable on this GPU. Falling back to CPU. "
                "You can also set TAPSEG_FORCE_CPU=1 before launching Slicer.",
                flush=True,
            )
            device, net_roi, net_cnt, net_ins = _load_model(model_dir, device=torch.device("cpu"))
            image, spacing, direction, origin, w, h, d = _read_data(input_file, device)
            tooth_label, roi_label, cnt_label = _inference(image, net_roi, net_cnt, net_ins, w, h, d, device)
            tooth_label_raw = tooth_label
        else:
            raise

    disable_post = os.environ.get("TAPSEG_DISABLE_POSTPROCESS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    if not disable_post:
        try:
            from .relabel_smooth import relabel_only_image

            lab = sitk.GetImageFromArray(tooth_label.astype(np.uint16, copy=False))
            lab.SetSpacing(spacing)
            lab.SetDirection(direction)
            lab.SetOrigin(origin)

            lab_pp = relabel_only_image(lab, src_path=Path(input_file))
            tooth_label = sitk.GetArrayFromImage(lab_pp).astype(np.uint16, copy=False)
        except Exception as e:
            print(
                f"[TAPSeg] Post-processing (relabel only) failed; using raw segmentation instead. "
                f"({type(e).__name__})",
                flush=True,
            )

    output = sitk.GetImageFromArray(tooth_label.astype(np.uint16, copy=False))
    output.SetSpacing(spacing)
    output.SetDirection(direction)
    output.SetOrigin(origin)
    sitk.WriteImage(output, output_file)

    save_stages = os.environ.get("TAPSEG_SAVE_STAGES", "").strip().lower() in ("1", "true", "yes", "y", "on")
    if save_stages:
        out_path = Path(output_file)
        base_dir = out_path.parent
        case_filename = out_path.name

        stage_dir_map = {
            "roi": base_dir / "0_roi",
            "dec": base_dir / "1_dec",
            "ind": base_dir / "2_indtooth",
        }
        for p in stage_dir_map.values():
            p.mkdir(parents=True, exist_ok=True)

        roi_path = stage_dir_map["roi"] / case_filename
        cnt_path = stage_dir_map["dec"] / case_filename
        ind_path = stage_dir_map["ind"] / case_filename

        roi_img = sitk.GetImageFromArray(roi_label.astype(np.uint8))
        roi_img.SetSpacing(spacing)
        roi_img.SetDirection(direction)
        roi_img.SetOrigin(origin)
        sitk.WriteImage(roi_img, str(roi_path))

        cnt_img = sitk.GetImageFromArray(cnt_label.astype(np.uint8))
        cnt_img.SetSpacing(spacing)
        cnt_img.SetDirection(direction)
        cnt_img.SetOrigin(origin)
        sitk.WriteImage(cnt_img, str(cnt_path))

        if tooth_label_raw is None:
            tooth_label_raw = tooth_label
        ind_img = sitk.GetImageFromArray(tooth_label_raw.astype(np.uint16, copy=False))
        ind_img.SetSpacing(spacing)
        ind_img.SetDirection(direction)
        ind_img.SetOrigin(origin)
        sitk.WriteImage(ind_img, str(ind_path))


def exec_math(sign, input_dir=r".\\model\\img_nrrd", output_dir=r".\\model\\pred", model_dir=r".\\model"):
    import os

    if not os.path.exists(input_dir):
        print("********** no input dir", flush=True)
        return
    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(input_dir):
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)
        try:
            exec_math_one(sign, in_path, out_path, model_dir)
        except Exception as e:
            print(f"********** failed: {name}: {e}", flush=True)
