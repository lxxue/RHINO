import numpy as np

damping = 1.0 - 1e-5


def uniform_sampling(data, img_size, num_sample):
    indices = np.random.permutation(np.prod(img_size))[:num_sample]
    output = {key: val.reshape(-1, *val.shape[2:])[indices] for key, val in data.items()}
    return output


def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1).astype(np.float32)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2).astype(np.float32)
    Q = np.stack(
        [dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]], axis=1
    ).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1


def uniform_sampling_continuous(data, img_size, num_sample):
    indices = np.random.rand(num_sample, 2).astype(np.float32) * damping
    indices *= (img_size[0] - 1, img_size[1] - 1)

    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val
    return output


def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where(
        (samples_uniform_row < bbox_min[0])
        | (samples_uniform_row > bbox_max[0])
        | (samples_uniform_col < bbox_min[1])
        | (samples_uniform_col > bbox_max[1])
    )[0]
    return index_outside


def get_index_outside_of_mask(samples_uniform, mask):
    xs = samples_uniform[:, 0]
    ys = samples_uniform[:, 1]
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1
    mask00 = mask[x1, y1].astype(bool)
    mask01 = mask[x1, y2].astype(bool)
    mask10 = mask[x2, y1].astype(bool)
    mask11 = mask[x2, y2].astype(bool)
    mask_all = mask00 | mask01 | mask10 | mask11
    index_outside = np.where(mask_all == 0)[0]
    return index_outside


def get_index_inside_of_mask(samples_uniform, mask):
    xs = samples_uniform[:, 0]
    ys = samples_uniform[:, 1]
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1
    mask00 = mask[x1, y1].astype(bool)
    mask01 = mask[x1, y2].astype(bool)
    mask10 = mask[x2, y1].astype(bool)
    mask11 = mask[x2, y2].astype(bool)
    mask_all = mask00 & mask01 & mask10 & mask11
    index_inside = np.where(mask_all == 1)[0]
    return index_inside


# TODO: improve sampling efficiency
def weighted_sampling(data, img_size, num_sample, ratio=0.9):
    # calculate bounding box
    mask = data["human_mask"]
    where = np.asarray(np.where(mask), dtype=np.float32)
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)

    # sample a certain amount of points inside the bounding box
    num_sample_bbox = int(num_sample * ratio)
    samples_bbox = np.random.rand(num_sample_bbox, 2).astype(np.float32) * damping
    samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min

    num_sample_uniform = num_sample - num_sample_bbox
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    indices = np.concatenate([samples_bbox, samples_uniform], axis=0)
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    if ratio == 1.0:
        index_outside = get_index_outside_of_mask(samples_bbox, mask)
        output["rgb"][index_outside] = 0.0
    else:
        # get indices for uniform samples outside of bbox
        index_outside = (
            get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max) + num_sample_bbox
        )

    return output, index_outside


def bg_only_sampling(data, img_size, num_sample):
    # calculate bounding box
    if "all_mask" in data:
        mask = data["all_mask"]
    else:
        mask = data["human_mask"]

    # make sure all samples are in background
    num_sample_uniform = 3 * num_sample
    # due to numerical issues it seems possible to be rounded down to 1.0
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    index_outside = get_index_outside_of_mask(samples_uniform, mask)
    samples_uniform = samples_uniform[index_outside]
    assert samples_uniform.shape[0] >= num_sample
    samples_uniform = samples_uniform[:num_sample, :]
    index_outside = get_index_outside_of_mask(samples_uniform, mask)
    assert len(index_outside) == num_sample
    indices = samples_uniform
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside


def bg_only_patch_sampling(data, img_size, num_sample):
    # calculate bounding box
    mask = data["human_mask"]

    # due to numerical issues it seems possible to be rounded down to 1.0
    samples = None
    while samples is None or samples.shape[0] == 0:
        bbox_min = np.random.rand(2).astype(np.float32) * damping
        patch_size = int(np.sqrt(num_sample))
        patch_size = int(patch_size) + 1
        samples = np.random.rand(patch_size**2, 2).astype(np.float32) * damping
        samples = samples * patch_size + bbox_min * (
            img_size[0] - 1 - patch_size,
            img_size[1] - 1 - patch_size,
        )

        # top_left = np.random.rand(2).astype(np.float32) * damping
        # print(patch_size)
        # patch_uv = np.mgrid[:patch_size, :patch_size]
        # patch_uv = np.flip(patch_uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        # patch_uv = patch_uv.reshape(-1, 2)
        # shuffle patch_uv along the first axis
        # np.random.shuffle(patch_uv)
        # print(top_left * (img_size[1] - 1 - patch_size, img_size[0] - 1 - patch_size))
        # samples = patch_uv + top_left * (img_size[0] - 1 - patch_size, img_size[1] - 1 - patch_size)
        # print(samples)
        # print(samples.min(axis=0), samples.max(axis=0))

        # print(mask.mean(), mask.max(), mask.min())
        index_outside = get_index_outside_of_mask(samples, mask)
        samples = samples[index_outside]

    # print(samples.shape)
    num_sample_patch = samples.shape[0]
    samples = samples[:num_sample, :]
    # num_sample_patch = 0
    # print(num_sample_patch)
    if num_sample_patch < num_sample:
        num_sample_uniform = (num_sample - num_sample_patch) * 3
        samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
        samples_uniform *= (img_size[0] - 1, img_size[1] - 1)
        index_outside_uniform = get_index_outside_of_mask(samples_uniform, mask)
        samples_uniform = samples_uniform[index_outside_uniform]
        samples_uniform = samples_uniform[: num_sample - num_sample_patch, :]

        # print(samples.shape[0], samples_uniform.shape[0])
        samples = np.concatenate([samples, samples_uniform], axis=0)
        # samples = samples_uniform

    # print(samples.shape[0])
    assert samples.shape[0] >= num_sample
    # assert samples.shape[0] > 0
    samples = samples[:num_sample, :]
    # exit(0)
    # print(samples)
    # exit(0)
    index_outside = get_index_outside_of_mask(samples, mask)
    assert len(index_outside) == num_sample
    # assert len(index_outside) > 0
    indices = samples
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside


def fg_bg_sampling(data, img_size, num_sample):
    # calculate bounding box
    # mask = data["obj_mask"]
    mask = np.logical_not(data["obj_mask"])
    # where = np.asarray(np.where(mask), dtype=np.float32)
    # bbox_min = where.min(axis=1)
    # bbox_max = where.max(axis=1)
    # num_sample_inside = int(num_sample * 0.8)
    # num_sample_outside = num_sample - num_sample_inside

    # sample a certain amount of points inside the bounding box
    # num_sample_bbox = int(num_sample * ratio)
    # samples_bbox = np.random.rand(num_sample_bbox, 2).astype(np.float32) * damping
    # samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min


    num_sample_uniform = 10 * num_sample
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    index_inside = get_index_inside_of_mask(samples_uniform, mask)
    samples_inside = samples_uniform[index_inside]
    if not samples_inside.shape[0] >= num_sample:
        print("Warning: not enough samples inside of mask")
        print(samples_inside.shape[0], num_sample)
    samples_inside = samples_inside[:num_sample, :]
    indices = samples_inside
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output  # , index_outside, index_inside


def obj_bg_sampling(data, img_size, num_sample):
    # calculate bounding box
    # mask = data["obj_mask"]
    mask = np.logical_not(data["human_mask"])
    # where = np.asarray(np.where(mask), dtype=np.float32)
    # bbox_min = where.min(axis=1)
    # bbox_max = where.max(axis=1)
    # num_sample_inside = int(num_sample * 0.8)
    # num_sample_outside = num_sample - num_sample_inside

    # sample a certain amount of points inside the bounding box
    # num_sample_bbox = int(num_sample * ratio)
    # samples_bbox = np.random.rand(num_sample_bbox, 2).astype(np.float32) * damping
    # samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min


    num_sample_uniform = 10 * num_sample
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    index_inside = get_index_inside_of_mask(samples_uniform, mask)
    samples_inside = samples_uniform[index_inside]
    if not samples_inside.shape[0] >= num_sample:
        print("Warning: not enough samples inside of mask")
        print(samples_inside.shape[0], num_sample)
    samples_inside = samples_inside[:num_sample, :]
    # assert len(index_outside) == 0
    indices = samples_inside
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output  # , index_outside, index_inside


def fg_obj_bg_sampling(data, img_size, num_sample, fg_ratio=0.7, obj_ratio=0.2):
    # print(fg_ratio, obj_ratio)
    # human bbox sampling
    human_mask = data["human_mask"]
    human_where = np.asarray(np.where(human_mask), dtype=np.float32)
    human_bbox_min = human_where.min(axis=1)
    human_bbox_max = human_where.max(axis=1)
    num_sample_human = int(num_sample * fg_ratio)
    samples_human = np.random.rand(num_sample_human, 2).astype(np.float32) * damping
    samples_human = samples_human * (human_bbox_max - human_bbox_min) + human_bbox_min

    # object bbox sampling
    obj_mask = data["obj_mask"]
    obj_where = np.asarray(np.where(obj_mask), dtype=np.float32)
    obj_bbox_min = obj_where.min(axis=1)
    obj_bbox_max = obj_where.max(axis=1)
    num_sample_obj = int(num_sample * obj_ratio)
    samples_obj = np.random.rand(num_sample_obj, 2).astype(np.float32) * damping
    samples_obj = samples_obj * (obj_bbox_max - obj_bbox_min) + obj_bbox_min

    num_sample_uniform = num_sample - num_sample_human - num_sample_obj
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    indices = np.concatenate([samples_human, samples_obj, samples_uniform], axis=0)

    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output

def fg_obj_hand_bg_sampling(data, img_size, num_sample, fg_ratio=0.4, obj_ratio=0.3, hand_ratio=0.2):
    # print(fg_ratio, obj_ratio, hand_ratio)
    # human bbox sampling
    human_mask = data["human_mask"]
    human_where = np.asarray(np.where(human_mask), dtype=np.float32)
    human_bbox_min = human_where.min(axis=1)
    human_bbox_max = human_where.max(axis=1)
    num_sample_human = int(num_sample * fg_ratio)
    samples_human = np.random.rand(num_sample_human, 2).astype(np.float32) * damping
    samples_human = samples_human * (human_bbox_max - human_bbox_min) + human_bbox_min

    # object bbox sampling
    obj_mask = data["obj_mask"]
    obj_where = np.asarray(np.where(obj_mask), dtype=np.float32)
    obj_bbox_min = obj_where.min(axis=1)
    obj_bbox_max = obj_where.max(axis=1)
    num_sample_obj = int(num_sample * obj_ratio)
    samples_obj = np.random.rand(num_sample_obj, 2).astype(np.float32) * damping
    samples_obj = samples_obj * (obj_bbox_max - obj_bbox_min) + obj_bbox_min

    # hand bbox sampling
    hand_mask = data["hand_mask"]
    num_sample_hand = int(num_sample * hand_ratio)
    hand_where = np.asarray(np.where(hand_mask), dtype=np.float32)
    if num_sample_hand > 0 and hand_where.shape[1] > 0:
        hand_bbox_min = hand_where.min(axis=1)
        hand_bbox_max = hand_where.max(axis=1)
        samples_hand = np.random.rand(num_sample_hand, 2).astype(np.float32) * damping
        samples_hand = samples_hand * (hand_bbox_max - hand_bbox_min) + hand_bbox_min
    else:
        samples_hand = np.zeros((0, 2), dtype=np.float32)
        num_sample_hand = 0

    num_sample_uniform = num_sample - num_sample_human - num_sample_obj - num_sample_hand
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    indices = np.concatenate([samples_human, samples_obj, samples_hand, samples_uniform], axis=0)

    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output

# TODO: design a more efficient sampling method
# TODO: design a more clean interface to avoid code duplication
def obj_only_sampling(data, img_size, num_sample, ratio=0.8):
    # calculate bounding box
    mask = data["obj_mask"]
    where = np.asarray(np.where(mask), dtype=np.float32)
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)
    # num_sample_inside = int(num_sample * 0.8)
    # num_sample_outside = num_sample - num_sample_inside

    # sample a certain amount of points inside the bounding box
    num_sample_bbox = int(num_sample * ratio)
    samples_bbox = np.random.rand(num_sample_bbox, 2).astype(np.float32) * damping
    samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min

    num_sample_uniform = num_sample - num_sample_bbox
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    indices = np.concatenate([samples_bbox, samples_uniform], axis=0)
    index_inside = get_index_inside_of_mask(indices, mask)
    index_outside = get_index_outside_of_mask(indices, mask)


    # samples_inside = samples_inside_uniform[index_inside]
    # if not samples_inside.shape[0] >= num_sample_inside:
    #     print("Warning: not enough samples inside of mask")
    #     print(samples_inside.shape[0], num_sample_inside)
    # samples_inside = samples_uniform_inside[:num_sample_inside, :]
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside, index_inside


def fg_only_sampling(data, img_size, num_sample):
    # calculate bounding box
    mask = data["human_mask"]
    # where = np.asarray(np.where(mask))
    # bbox_min = where.min(axis=1)
    # bbox_max = where.max(axis=1)

    # make sure all samples are in background
    num_sample_uniform = 40 * num_sample
    # due to numerical issues it seems possible to be rounded down to 1.0
    samples_uniform = np.random.rand(num_sample_uniform, 2).astype(np.float32) * damping
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    index_inside = get_index_inside_of_mask(samples_uniform, mask)
    samples_uniform = samples_uniform[index_inside]
    if not samples_uniform.shape[0] >= num_sample:
        print("Warning: not enough samples inside of mask")
        print(samples_uniform.shape[0], num_sample)
    samples_uniform = samples_uniform[:num_sample, :]
    index_outside = get_index_outside_of_mask(samples_uniform, mask)
    assert len(index_outside) == 0
    indices = samples_uniform
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside
