import numpy as np


def center_dimension(boxes: np.ndarray) -> np.ndarray:
    """ Returns boxes in center and dimension format

    :param boxes: Boxes in (N, 4) dimension. Second axis is in (y_min,
        x_min, y_max, x_max) format.

    :return: Boxes in (N, 4) dimension. Second axis contains (center_y,
        center_x, height, width) format
    """
    dimension = boxes[:, 2:] - boxes[:, :2]
    center = boxes[:, :2] + dimension * 0.5
    return np.concatenate((center, dimension), axis=1)


def corners(boxes: np.ndarray) -> np.ndarray:
    """ Returns boxes in corners format

    :param boxes: Boxes in (N, 4) dimension. Second axis contains (center_y,
        center_x, height, width) format

    :return: Boxes in (N, 4) dimension. Second axis is in (y_min,
        x_min, y_max, x_max) format.
    """
    top_right = boxes[:, :2] - boxes[:, 2:] * 0.5
    bot_left = boxes[:, :2] + boxes[:, 2:] * 0.5
    return np.concatenate((top_right, bot_left), axis=1)


def from_offset_scale(src_bbox: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :math:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        offset (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=offset.dtype)

    src_cd = center_dimension(src_bbox)

    center = offset[:, :2] * src_cd[:, 2:] + src_cd[:, :2]
    dimension = np.exp(offset[:, 2:]) * src_cd[:, 2:]
    dst_cd = np.concatenate((center, dimension), axis=1)
    return corners(dst_cd)


def offset_scale(src_bbox: np.ndarray, dst_bbox: np.ndarray):
    """ Encodes the source and the destination bounding boxes to "loc".
    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.
    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.
    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.
    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    """
    src_cd = center_dimension(src_bbox)
    dst_cd = center_dimension(dst_bbox)

    # To prevent log of zero
    # The smallest representable positive number such that 1.0 + eps != 1.0.
    # Memory expensive operation but cached
    eps = np.finfo(src_cd.dtype).eps
    src_cd = src_cd.clip(min=eps)

    offset = (dst_cd[:, :2] - src_cd[:, :2]) / src_cd[:, 2:]
    dim_ratio = dst_cd[:, 2:] / src_cd[:, 2:]
    scale = np.log(dim_ratio)

    return np.concatenate((offset, scale), axis=1)


def intersection_over_union(boxes_a: np.ndarray,
                            boxes_b: np.ndarray) -> np.ndarray:
    """ Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        boxes_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        boxes_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """

    if boxes_a.shape[1] != 4 or boxes_b.shape[1] != 4:
        raise ValueError("Boxes should have a shape of (N, 4)")

    tl = np.maximum(boxes_a[:, np.newaxis, :2], boxes_b[:, :2])
    br = np.minimum(boxes_a[:, np.newaxis, 2:], boxes_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)
    area_u = area_a[:, np.newaxis] + area_b - area_i

    return area_i / area_u


# def generate_anchor_base(base_size=16,
#                          ratios=(0.5, 1, 2),
#                          anchor_scales=(8, 16, 32)):
#     """Generate anchor base windows by enumerating aspect ratio and scales.
#
#     Generate anchors that are scaled and modified to the given aspect ratios.
#     Area of a scaled anchor is preserved when modifying to the given aspect
#     ratio.
#
#     :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
#     function.
#     The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
#     generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.
#
#     For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
#     the width and the height of the base window will be stretched by :math:`8`.
#     For modifying the anchor to the given aspect ratio,
#     the height is halved and the width is doubled.
#
#     Args:
#         base_size (number): The width and the height of the reference window.
#         ratios (list of floats): This is ratios of width to height of
#             the anchors.
#         anchor_scales (list of numbers): This is areas of anchors.
#             Those areas will be the product of the square of an element in
#             :obj:`anchor_scales` and the original area of the reference
#             window.
#
#     Returns:
#         numpy.ndarray:
#         An array of shape :math:`(R, 4)`.
#         Each element is a set of coordinates of a bounding box.
#         The second axis corresponds to
#         :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.
#
#     """
#     # Using broadcasting populate center points for all
#     # anchor box which is same
#     center_pt = np.array((base_size / 2., base_size / 2.),
#                          dtype=np.float32)
#     center = np.zeros((len(ratios) * len(anchor_scales), 1),
#                       dtype=center_pt.dtype)
#
#     # This operation is common
#     scales = np.array(anchor_scales, dtype=center_pt.dtype) * base_size
#
#     # Add a new axis to broadcast the operation
#     r = np.array(ratios, dtype=center_pt.dtype)[:, np.newaxis]
#
#     dims = scales[np.newaxis] * (np.sqrt(r), np.sqrt(1. / r))  # (2, 3, 3)
#     dims = dims.reshape(2, -1).transpose(1, 0)
#
#     anchors_cd = np.concatenate((center + center_pt, dims), axis=1)
#
#     return corners(anchors_cd)


def non_max_supression(boxes, threshold):
    """ Supress overlapping boxes based on the threshold

    Args:
        boxes (~numpy.ndarray): Boxes of shape :math: `(N, 4)` 
            arranged in descending order based on their 
            objectiveness score
        
        threshold (float): IOU to consdier overlapping boxes.

    Returns:
        ~nump.ndarray:
            An 1d array of shape :math: `(R)`
            Bounding boxes after supression
    """
    keep = []
    order = np.arange(len(boxes))

    while len(order) > 0:
        # Box against which IOU is calculated for the remainder
        curr_box = boxes[np.newaxis, 0]
        rest_boxes = boxes[order[1:]]

        ious = intersection_over_union(curr_box, rest_boxes)[0]
        filtered = np.where(ious <= threshold)[0]

        keep.append(order[0])
        # since the 0th element is excluded and is 0th element vs rest
        order = order[filtered + 1]
    return np.array(keep, dtype=np.int32)
