import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure


DUPLICATE_CONTAINMENT = 0.35


def _default_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_gaussian(meanx, meany, meanz, patch):
    h, w, d = patch.shape
    x, y, z = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")

    delta_squared = 5**2
    distance_squared = (x - meanx) ** 2 + (y - meany) ** 2 + (z - meanz) ** 2
    gaussian = np.exp(-distance_squared / (2 * delta_squared))
    return gaussian


def _is_duplicate(mask, start, voxel_count, accepted):
    for previous_start, previous_mask, previous_count in accepted:
        lo = np.maximum(start, previous_start)
        hi = np.minimum(start + np.asarray(mask.shape), previous_start + np.asarray(previous_mask.shape))
        if np.any(hi <= lo):
            continue
        current_slice = tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(lo, hi, start))
        previous_slice = tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(lo, hi, previous_start))
        overlap = int(np.count_nonzero(mask[current_slice] & previous_mask[previous_slice]))
        if overlap / min(voxel_count, previous_count) >= DUPLICATE_CONTAINMENT:
            return True
    return False


def tooth_seg(net_seg, image, centroids, patch_size, device=None):
    device = _default_device(device)
    w, h, d = image.shape

    if centroids is None or getattr(centroids, "size", 0) == 0:
        return np.zeros_like(image, dtype=int)

    add_pad = False
    w_pad = max(patch_size[0] - w, 0)
    h_pad = max(patch_size[1] - h, 0)
    d_pad = max(patch_size[2] - d, 0)
    add_pad = (w_pad > 0) or (h_pad > 0) or (d_pad > 0)
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )

    crop_size = np.asarray(patch_size)
    image_list, gaussian_list, crop_coord_min_list = [], [], []

    for i in range(centroids.shape[1]):
        meanx = int(centroids[0][i])
        meany = int(centroids[1][i])
        meanz = int(centroids[2][i])

        mean_coord = np.asarray((meanx, meany, meanz))
        crop_coord_min_0 = mean_coord - crop_size / 2
        crop_coord_min = np.clip(crop_coord_min_0, (0, 0, 0), np.asarray(image.shape) - crop_size)
        offset = crop_coord_min - crop_coord_min_0
        crop_coord_min = crop_coord_min.astype(int)

        crop_image = image[
            crop_coord_min[0] : (crop_coord_min[0] + crop_size[0]),
            crop_coord_min[1] : (crop_coord_min[1] + crop_size[1]),
            crop_coord_min[2] : (crop_coord_min[2] + crop_size[2]),
        ]
        crop_gaussian = generate_gaussian(
            crop_size[0] / 2 - offset[0],
            crop_size[1] / 2 - offset[1],
            crop_size[2] / 2 - offset[2],
            crop_image,
        ).astype(np.float32)

        image_list.append(crop_image)
        gaussian_list.append(crop_gaussian)
        crop_coord_min_list.append(crop_coord_min)

    patches_coord_min = np.asarray(crop_coord_min_list)
    image_patches = np.asarray(image_list)
    gaussian_patches = np.asarray(gaussian_list)

    image_patches_t = torch.from_numpy(image_patches[:, None, :, :, :]).float().to(device)
    gaussian_patches_t = torch.from_numpy(gaussian_patches[:, None, :, :, :]).float().to(device)

    seg_logits = []
    with torch.no_grad():
        bs = 10
        for start in range(0, image_patches_t.shape[0], bs):
            end = start + bs
            seg, _seg_bd = net_seg(image_patches_t[start:end], gaussian_patches_t[start:end])
            seg_logits.append(seg)
    seg_patches = torch.cat(seg_logits, 0)

    seg_patches = F.softmax(seg_patches, dim=1)
    seg_patches = torch.argmax(seg_patches, dim=1).detach().cpu().numpy()

    w2, h2, d2 = image.shape
    count = 0
    image_label = np.zeros((w2, h2, d2), dtype=int)
    accepted = []

    for crop_i in range(patches_coord_min.shape[0]):
        labels, num = measure.label(
            seg_patches[crop_i, :, :, :], connectivity=2, background=0, return_num=True
        )
        if num > 1:
            max_num = -1
            true_id = 1
            for lab_id in range(1, num + 1):
                s = int(np.sum(labels == lab_id))
                if s > max_num:
                    max_num = s
                    true_id = lab_id
            seg_bin = labels == true_id
        else:
            seg_bin = seg_patches[crop_i, :, :, :] == 1

        voxel_count = int(np.count_nonzero(seg_bin))
        crop_start = patches_coord_min[crop_i]
        if voxel_count == 0 or _is_duplicate(seg_bin, crop_start, voxel_count, accepted):
            continue

        coord = np.array(np.nonzero(seg_bin))
        coord[0] = coord[0] + patches_coord_min[crop_i, 0]
        coord[1] = coord[1] + patches_coord_min[crop_i, 1]
        coord[2] = coord[2] + patches_coord_min[crop_i, 2]

        count += 1
        image_label[coord[0], coord[1], coord[2]] = count
        accepted.append((crop_start, seg_bin, voxel_count))

    if add_pad:
        image_label = image_label[wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d]
    return image_label


def ins_tooth_seg(ins_net, image, centroids, patch_size, device=None):
    return tooth_seg(ins_net, image, centroids, patch_size, device=device)
