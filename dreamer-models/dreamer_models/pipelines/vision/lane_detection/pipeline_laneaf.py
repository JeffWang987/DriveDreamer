from collections import defaultdict

import cv2
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from torch import nn

from ...pipeline import BasePipeline


class LaneAFPipeline(BasePipeline):
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        self.device = 'cpu'

    def __call__(self, image):
        input_image, scale = preprocess(image)
        output = self.model(input_image)
        pred_lanes, pred_lane_labels = postprocess(output, (image.height, image.width), scale)
        return pred_lanes, pred_lane_labels


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 32.0)) * 32
    W = int(np.round(W / 32.0)) * 32
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    scale = (input_image.shape[0] / float(H), input_image.shape[1] / float(W))
    return img, scale


def preprocess(image):
    image = np.array(image)[:, :, ::-1]
    image, scale = resize_image(image, resolution=512)
    input_image = np.transpose(image[None], (0, 3, 1, 2))
    input_image = torch.from_numpy(input_image.copy()).type(torch.float32)
    input_image = input_image / 128.0 - 1.0
    return input_image, scale


def postprocess(output, image_shape, scale):
    config = {'do_last_points_fixed': False}
    sample_factor_x = 2
    sample_factor_y = 2
    crop_offset = (0, 0)
    outputs_hm, outputs_vaf, outputs_haf, outputs_lanetype = output
    outputs_hm = outputs_hm.permute([0, 3, 1, 2]).contiguous()
    outputs_vaf = outputs_vaf.permute([0, 3, 1, 2]).contiguous()
    outputs_haf = outputs_haf.permute([0, 3, 1, 2]).contiguous()
    outputs_lanetype = outputs_lanetype.permute([0, 3, 1, 2]).contiguous()
    output_lane_bias = outputs_haf[:, 6:8, :, :]
    output_road_bias = outputs_haf[:, 8:10, :, :]
    lane_vbias = np.transpose(output_lane_bias[0, :1, :, :].detach().cpu().float().numpy(), (1, 2, 0))[:, :, 0]
    road_vbias = np.transpose(output_road_bias[0, :1, :, :].detach().cpu().float().numpy(), (1, 2, 0))[:, :, 0]
    lane_hbias = np.transpose(output_lane_bias[0, 1:2, :, :].detach().cpu().float().numpy(), (1, 2, 0))[:, :, 0]
    road_hbias = np.transpose(output_road_bias[0, 1:2, :, :].detach().cpu().float().numpy(), (1, 2, 0))[:, :, 0]
    hm_sigmoid = outputs_hm[:, :7, :, :]
    hm_sigmoid_diamond = outputs_hm[:, 7:8, :, :]
    output_lanetype = outputs_lanetype[:, 0:4, :, :]
    output_lanecolor = outputs_lanetype[:, 4:6, :, :]
    outputs_haf = outputs_haf[:, 0:6, :, :]
    lane_mask_out = tensor2image(hm_sigmoid[:, :1, :, :].repeat(1, 3, 1, 1).detach())
    lane_vaf_out = np.transpose(outputs_vaf[0, :2, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    lane_haf_out = np.transpose(outputs_haf[0, :1, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    new_th = 10
    lane_lanetype_out = np.transpose(output_lanetype[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    lane_colortype_out = np.transpose(output_lanecolor[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    feat_height, feat_width = lane_lanetype_out.shape[0], lane_lanetype_out.shape[1]
    lane_export_lanes = decodePoses_bias(
        lane_mask_out[:, :, 0],
        lane_vaf_out,
        lane_haf_out,
        lane_hbias,
        lane_vbias,
        fg_thresh=0.5,
        err_thresh=new_th,
        config=config,
    )
    lane_parsed_lanes = adjust_lanes(
        lane_export_lanes,
        sample_factor_x,
        sample_factor_y,
        crop_offset,
        feat_height,
        feat_width,
        lane_lanetype_out,
        lane_colortype_out,
    )
    # road
    road_mask_out = tensor2image(hm_sigmoid[:, 1:2, :, :].repeat(1, 3, 1, 1).detach())
    road_vaf_out = np.transpose(outputs_vaf[0, 2:4, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    road_haf_out = np.transpose(outputs_haf[0, 1:2, :, :].detach().cpu().float().numpy(), (1, 2, 0))  # chw -> hwc
    road_export_lanes = decodePoses_bias(
        road_mask_out[:, :, 0],
        road_vaf_out,
        road_haf_out,
        road_hbias,
        road_vbias,
        fg_thresh=0.5,
        err_thresh=new_th,
        config=config,
    )
    road_parsed_lanes = adjust_lanes(
        road_export_lanes, sample_factor_x, sample_factor_y, crop_offset, feat_height, feat_width, 0, 0
    )
    # stopline
    stop_mask_out = tensor2image(hm_sigmoid[:, 2:3, :, :].repeat(1, 3, 1, 1).detach())
    stopline_vaf_out = np.transpose(outputs_vaf[0, 8:10, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    stopline_haf_out = np.transpose(outputs_haf[0, 2:3, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    export_stoplines = decodePosesHorizon(
        stop_mask_out[:, :, 0], stopline_vaf_out, stopline_haf_out, fg_thresh=0.5, err_thresh=new_th
    )
    parsed_stop_lanes = adjust_stop_lanes(export_stoplines, sample_factor_x, sample_factor_y, crop_offset)
    # double left
    double_left_mask_out = tensor2image(hm_sigmoid[:, 3:4, :, :].repeat(1, 3, 1, 1).detach())
    double_left_vaf_out = np.transpose(outputs_vaf[0, 4:6, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    double_left_haf_out = np.transpose(
        outputs_haf[0, 3:4, :, :].detach().cpu().float().numpy(), (1, 2, 0)
    )  # chw -> hwc
    double_left_export_lanes = decodePoses(
        double_left_mask_out[:, :, 0],
        double_left_vaf_out,
        double_left_haf_out,
        fg_thresh=0.5,
        err_thresh=new_th,
        config=config,
    )
    double_left_parsed_lanes = adjust_lanes(
        double_left_export_lanes,
        sample_factor_x,
        sample_factor_y,
        crop_offset,
        feat_height,
        feat_width,
        lane_lanetype_out,
        lane_colortype_out,
    )
    # double right
    double_right_mask_out = tensor2image(hm_sigmoid[:, 4:5, :, :].repeat(1, 3, 1, 1).detach())
    double_right_vaf_out = np.transpose(outputs_vaf[0, 6:8, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    double_right_haf_out = np.transpose(
        outputs_haf[0, 4:5, :, :].detach().cpu().float().numpy(), (1, 2, 0)
    )  # chw -> hwc
    double_right_export_lanes = decodePoses(
        double_right_mask_out[:, :, 0],
        double_right_vaf_out,
        double_right_haf_out,
        fg_thresh=0.5,
        err_thresh=new_th,
        config=config,
    )
    double_right_parsed_lanes = adjust_lanes(
        double_right_export_lanes,
        sample_factor_x,
        sample_factor_y,
        crop_offset,
        feat_height,
        feat_width,
        lane_lanetype_out,
        lane_colortype_out,
    )
    # diamond
    diamond_mask_out = tensor2image(hm_sigmoid_diamond[:, :, :, :].repeat(1, 3, 1, 1).detach())
    diamond_vaf_out = np.transpose(outputs_vaf[0, 10:12, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    diamond_haf_out = np.transpose(outputs_haf[0, 5:6, :, :].detach().cpu().float().numpy(), (1, 2, 0))  # chw -> hwc
    diamond_export_lanes = decodePoses(
        diamond_mask_out[:, :, 0], diamond_vaf_out, diamond_haf_out, fg_thresh=0.5, err_thresh=new_th, config=config
    )
    diamond_parsed_lanes = adjust_lanes(
        diamond_export_lanes,
        sample_factor_x,
        sample_factor_y,
        crop_offset,
        feat_height,
        feat_width,
        5,
        lane_colortype_out,
    )
    # -1 road, 0 solid, 1 dot, 2 stopline, 4 diamoid两侧的虚线  5 diamoid solid, 6 diamoid dot
    # 修正线型
    lane_parsed_lanes = diamond_type_recheck(lane_parsed_lanes, diamond_parsed_lanes, 20)
    double_lanes = double_left_parsed_lanes + double_right_parsed_lanes
    double_lanes_ids = [1 for _ in range(len(double_left_parsed_lanes))] + [
        2 for _ in range(len(double_right_parsed_lanes))
    ]
    filter_single_lanes, lane_double_ids = lane_match_double(lane_parsed_lanes, double_lanes, double_lanes_ids)
    hm_thr = 0.25
    # fencha
    hm_fencha_pt = torch.clamp(hm_sigmoid[0, 5:6, :, :], min=1e-4, max=1 - 1e-4)
    hm_fencha_pt = _nms(hm_fencha_pt, kernel=11)
    hm_fencha_pt = hm_fencha_pt.detach().cpu().float().numpy()[0]
    fencha_pt = np.where(hm_fencha_pt >= hm_thr)
    fencha_pts = np.stack([fencha_pt[1], fencha_pt[0]], axis=1)
    # hebing
    hm_hebing_pt = torch.clamp(hm_sigmoid[0, 6:7, :, :], min=1e-4, max=1 - 1e-4)
    hm_hebing_pt = _nms(hm_hebing_pt, kernel=11)
    hm_hebing_pt = hm_hebing_pt.detach().cpu().float().numpy()[0]
    hebing_pt = np.where(hm_hebing_pt >= hm_thr)
    hebing_pts = np.stack([hebing_pt[1], hebing_pt[0]], axis=1)
    valid_pts = np.concatenate([fencha_pts, hebing_pts], axis=0)
    valid_pts = adjust_pts(
        valid_pts,
        sample_factor_x,
        sample_factor_y,
        crop_offset,
        feat_height,
        feat_width,
    )
    y_type_process(valid_pts, filter_single_lanes)
    parsed_lanes = filter_single_lanes + road_parsed_lanes + parsed_stop_lanes + diamond_parsed_lanes
    parsed_double_ids = (
        lane_double_ids
        + [-1 for _ in range(len(road_parsed_lanes))]
        + [-1 for _ in range(len(parsed_stop_lanes))]
        + [-1 for _ in range(len(diamond_parsed_lanes))]
    )
    return get_results(parsed_lanes, parsed_double_ids, valid_pts, image_shape, scale)


def check_y_pair(
    interp_lane0,
    interp_lane1,
    intersect_pt,
    lane_dist0,
    lane_dist1,
    longer_lane_thresh_min_thresh=20,
    longer_lane_thresh_max_thresh=30,
    longer_lane_thresh_min_x_thresh=30,
    longer_lane_thresh_max_x_thresh=90,
    start_y=576,
    end_y=1080,
):  # end_y:30fov=1080 120fov=840
    start_ratio = 0.1
    end_ratio = 0.9
    match_cnt = 0
    # intersect_x = intersect_pt[0]
    intersect_y = intersect_pt[1]
    lane0_up_ratio = np.sum(interp_lane0[:, 1] > intersect_y) / len(interp_lane0)
    lane0_down_ratio = np.sum(interp_lane0[:, 1] < intersect_y) / len(interp_lane0)
    lane1_up_ratio = np.sum(interp_lane1[:, 1] > intersect_y) / len(interp_lane1)
    lane1_down_ratio = np.sum(interp_lane1[:, 1] < intersect_y) / len(interp_lane1)
    lane1_match_cnt = 0
    lane2_match_cnt = 0
    if not (start_ratio <= lane0_up_ratio <= end_ratio):
        match_cnt += 1
        lane1_match_cnt += 1
    if not (start_ratio <= lane1_up_ratio <= end_ratio):
        match_cnt += 1
        lane2_match_cnt += 1
    if not (start_ratio <= lane0_down_ratio <= end_ratio):
        match_cnt += 1
        lane1_match_cnt += 1
    if not (start_ratio <= lane1_down_ratio <= end_ratio):
        match_cnt += 1
        lane2_match_cnt += 1
    if match_cnt == 0:
        return False
    else:
        dynamic_thresh = (intersect_y - start_y) / (end_y - start_y) * (
            longer_lane_thresh_max_thresh - longer_lane_thresh_min_thresh
        ) + longer_lane_thresh_min_thresh
        dynamic_x_thresh = (intersect_y - start_y) / (end_y - start_y) * (
            longer_lane_thresh_max_x_thresh - longer_lane_thresh_min_thresh
        ) + longer_lane_thresh_min_x_thresh
        # check the longer lane the distance is miner than a thresh
        if lane1_match_cnt == 0:
            pt_corr_idx = np.argmin(np.abs(interp_lane0[:, 1] - intersect_y))
            corr_dist = np.linalg.norm(interp_lane0[pt_corr_idx] - np.asarray(intersect_pt))
            if corr_dist < dynamic_x_thresh and lane_dist0 < dynamic_thresh:
                return True
            else:
                return False
        else:
            pt_corr_idx = np.argmin(np.abs(interp_lane1[:, 1] - intersect_y))
            corr_dist = np.linalg.norm(interp_lane1[pt_corr_idx] - np.asarray(intersect_pt))
            if corr_dist < dynamic_x_thresh and lane_dist1 < dynamic_thresh:
                return True
            else:
                return False


def tran_lanes(lane):
    trans_lane = list()
    for pt in lane:
        trans_lane.append([pt['x'], pt['y']])
    return trans_lane


def interp_kernel(pt1, pt2):
    def Lerp(a, b, t):
        return a + t * (b - a)

    x_span = abs(pt1[0] - pt2[0])
    y_span = abs(pt1[1] - pt2[1])
    span = max(x_span, y_span)
    span = int(span)
    interp_pts = list()
    if span < 1:
        interp_pts.append(pt1)
    else:
        for span_idx in range(span + 1):
            ratio = span_idx / span
            interp_x = Lerp(pt1[0], pt2[0], ratio)
            interp_y = Lerp(pt1[1], pt2[1], ratio)
            interp_pts.append([interp_x, interp_y])
    return interp_pts


def interp_lane_func(lane):
    lane_size = len(lane)
    all_interp_lane = list()
    for pt_idx in range(lane_size - 1):
        interp_pts = interp_kernel(lane[pt_idx], lane[pt_idx + 1])
        if len(interp_pts) == 1:
            all_interp_lane.append(interp_pts[0])
        elif pt_idx == lane_size - 2:
            all_interp_lane.extend(interp_pts)
        else:
            all_interp_lane.extend(interp_pts[:-1])
    return all_interp_lane


def point_to_string(a, b, a1, b1, a2, b2):
    def base_distance(x1, y1, x2, y2):
        return np.linalg.norm([x1 - x2, y1 - y2])

    c1 = base_distance(a, b, a1, b1)
    c2 = base_distance(a, b, a2, b2)
    c3 = base_distance(a1, b1, a2, b2)
    # 计算半周长
    s = (c1 + c2 + c3) / 2
    # 计算面积
    area = (s * (s - c1) * (s - c2) * (s - c3)) ** 0.5
    # 计算垂线
    h = area / c3 * 2
    return h


def merge_lane_pair(y_point, lane_data1, lane_data2):
    # y_point_x = y_point[0]
    y_point_y = y_point[1]
    y_2_cnt = defaultdict(int)
    y_2_x = dict()
    y_2_islane = dict()
    y_2_lanetype = dict()
    y_2_lanecolor = dict()
    for pt in lane_data1:
        y_2_cnt[int(pt['y'])] += 1
        y_2_x[int(pt['y'])] = int(pt['x'])
        y_2_islane[int(pt['y'])] = pt['islane']
        y_2_lanetype[int(pt['y'])] = pt['lanetype']
        y_2_lanecolor[int(pt['y'])] = pt['lanecolor']
    for pt in lane_data2:
        y_2_cnt[int(pt['y'])] += 1
        y_2_x[int(pt['y'])] = int(pt['x'])
        y_2_islane[int(pt['y'])] = pt['islane']
        y_2_lanetype[int(pt['y'])] = pt['lanetype']
        y_2_lanecolor[int(pt['y'])] = pt['lanecolor']
    up_cnt = 0
    down_cnt = 0
    common_poses = list()
    for y, cnt in y_2_cnt.items():
        if cnt == 2:
            if y > y_point_y:
                down_cnt += 1
            else:
                up_cnt += 1
        elif cnt == 1:
            common_poses.append(
                {
                    'x': y_2_x[y],
                    'y': y,
                    'islane': y_2_islane[y],
                    'lanetype': y_2_lanetype[y],
                    'lanecolor': y_2_lanecolor[y],
                }
            )
    insert_poses = list()
    append_poses = list()
    if up_cnt > down_cnt:
        # Y型线
        for pt in common_poses:
            if pt['y'] > y_point_y:
                append_poses.append(pt)
        down_cnt1 = 0
        down_cnt2 = 0
        for pt in lane_data1:
            if pt['y'] > y_point_y:
                down_cnt1 += 1
        for pt in lane_data2:
            if pt['y'] > y_point_y:
                down_cnt2 += 1
        # 根据长度判断哪个需要扩展
        if down_cnt1 < down_cnt2:
            last_pt = lane_data1[-1]
            for pt in append_poses:
                if pt['y'] > last_pt['y']:
                    lane_data1.append(pt)
        else:
            last_pt = lane_data2[-1]
            for pt in append_poses:
                if pt['y'] > last_pt['y']:
                    lane_data2.append(pt)
    else:
        # 反y型线
        for pt in common_poses:
            if pt['y'] < y_point_y:
                insert_poses.append(pt)
        # 根据高于或者低于y轴线的个数来判断
        up_cnt1 = 0
        up_cnt2 = 0
        for pt in lane_data1:
            if pt['y'] < y_point_y:
                up_cnt1 += 1
        for pt in lane_data2:
            if pt['y'] < y_point_y:
                up_cnt2 += 1

        if up_cnt1 < up_cnt2:
            first_pt = lane_data1[0]
            for pt in insert_poses[::-1]:
                if pt['y'] < first_pt['y']:
                    lane_data1.insert(0, pt)
        else:
            first_pt = lane_data2[0]
            for pt in insert_poses[::-1]:
                if pt['y'] < first_pt['y']:
                    lane_data2.insert(0, pt)


def y_type_process(parsed_det_area, input_lanes_data):
    minist_fit_line_length = 30
    distance_thresh = 30
    first_lane_pt_match_distance_thresh_up = 90
    first_lane_pt_match_distance_thresh_down = 40
    start_y = 576
    end_y = 1080

    def dyn_thresh_with_y(input_y):
        dyn_len_thresh = (input_y - start_y) / (end_y - start_y) * (
            first_lane_pt_match_distance_thresh_up - first_lane_pt_match_distance_thresh_down
        ) + first_lane_pt_match_distance_thresh_down
        return dyn_len_thresh

    lanes_data = [tran_lanes(x) for x in input_lanes_data]
    interp_lane_datas = [interp_lane_func(lane) for lane in lanes_data]
    interp_lane_datas = [np.asarray(x) for x in interp_lane_datas]
    filter_interp_idxs = list()
    filter_interp_lanes = list()
    for idx, interp_lane in enumerate(interp_lane_datas):
        filter_interp_idxs.append(idx)
        filter_interp_lanes.append(interp_lane)
    merge_pt_2_lane_idx_pairs = list()
    merge_lane_idxes = set()
    for pt_idx, pt in enumerate(parsed_det_area):
        pt_array = np.asarray([pt])
        pt_2_lane_dises = list()
        for interp_lane in filter_interp_lanes:
            cdist_map = cdist(pt_array, interp_lane)
            argmin_idx = np.argmin(cdist_map)
            minest_dist = cdist_map[0, argmin_idx]
            corr_y = pt[1]
            if minest_dist < dyn_thresh_with_y(corr_y) and not (
                0.1 <= argmin_idx / cdist_map.shape[1] <= 0.9
            ):  # 动态thresh according y coord
                y_start = min(interp_lane[0, 1], interp_lane[-1, 1])
                y_end = max(interp_lane[0, 1], interp_lane[-1, 1])
                if pt[1] <= y_start or pt[1] >= y_end:  # make sure y is sorted
                    if abs(pt[1] - y_start) < abs(pt[1] - y_end):
                        pt1 = interp_lane[0]
                        pt2 = interp_lane[min(minist_fit_line_length, len(interp_lane)) - 1]
                    else:
                        pt1 = interp_lane[-1]
                        pt2 = interp_lane[-min(minist_fit_line_length, len(interp_lane))]
                    min_dist = point_to_string(pt[0], pt[1], pt1[0], pt1[1], pt2[0], pt2[1])
                    minest_dist = min(minest_dist, min_dist)
            pt_2_lane_dises.append(minest_dist)
        argsorted_idxes = np.argsort(pt_2_lane_dises)
        if (
            len(argsorted_idxes) >= 2
            and pt_2_lane_dises[argsorted_idxes[0]] < distance_thresh
            and pt_2_lane_dises[argsorted_idxes[1]] < distance_thresh
            and argsorted_idxes[0] not in merge_lane_idxes
            and argsorted_idxes[1] not in merge_lane_idxes
            and check_y_pair(
                filter_interp_lanes[argsorted_idxes[0]],
                filter_interp_lanes[argsorted_idxes[1]],
                parsed_det_area[pt_idx],
                pt_2_lane_dises[argsorted_idxes[0]],
                pt_2_lane_dises[argsorted_idxes[1]],
            )
        ):
            merge_lane_idxes.add(argsorted_idxes[0])
            merge_lane_idxes.add(argsorted_idxes[1])
            merge_pt_2_lane_idx_pairs.append(
                (pt_idx, filter_interp_idxs[argsorted_idxes[0]], filter_interp_idxs[argsorted_idxes[1]])
            )
    # merge_lane_pair
    for merge_pt_2_lane_idx_pair in merge_pt_2_lane_idx_pairs:
        pt_idx, lane_idx1, lane_idx2 = merge_pt_2_lane_idx_pair
        merge_lane_pair(parsed_det_area[pt_idx], input_lanes_data[lane_idx1], input_lanes_data[lane_idx2])


def adjust_pts(valid_pts, samp_factor_x, samp_factor_y, crop_offset, feat_height, feat_width):
    valid_pts = np.asarray(valid_pts).astype(float)
    filter_pts = list()
    if len(valid_pts.shape) > 1:
        for coord in valid_pts:
            x, y = int(coord[0]), int(coord[1])
            x = max(0, min(x, feat_width - 1))
            y = max(0, min(y, feat_height - 1))
            x *= float(samp_factor_x)
            y *= float(samp_factor_y)
            y += crop_offset[1]
            filter_pts.append([x, y])
    return filter_pts


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def trans_lane_format(lane):
    trans_lane = dict()
    for coord in lane:
        x, y = int(coord['x']), int(coord['y'])
        trans_lane[y] = x
    return trans_lane


def lane_match_double(single_lanes, double_lanes, double_lanes_ids):
    def lane_dist_ioua(lane_dict1, lane_dict2, x_dist_thres=12, common_ratio_thres=0.7):
        intersect_keys = list(set(lane_dict1.keys()).intersection(set(lane_dict2.keys())))
        intersect_keys.sort(reverse=True)
        common_diffs = list()
        if len(lane_dict2) > len(lane_dict1) * 0.9:
            lanes2_ys = list(lane_dict2.keys())
            lanes2_xs = list(lane_dict2.values())
            y_diff = abs(lanes2_ys[0] - lanes2_ys[-1])
            x_diff = abs(lanes2_xs[0] - lanes2_xs[-1])
            scale = np.sin(np.arctan([y_diff / (x_diff + 1e-7)])[0])
            for k in intersect_keys:
                x1 = lane_dict1[k]
                x2 = lane_dict2[k]
                x_diff = abs(x1 - x2)
                if x_diff * scale <= x_dist_thres:
                    common_diffs.append(x_diff * scale)
        if len(common_diffs) / len(lane_dict1) > common_ratio_thres:
            return np.mean(common_diffs)
        else:
            return np.inf

    x_dist_thres = 16
    single_lanes_dict = list()
    double_lanes_dict = list()
    for single_lane in single_lanes:
        single_lanes_dict.append(trans_lane_format(single_lane))
    for double_lane in double_lanes:
        double_lanes_dict.append(trans_lane_format(double_lane))
    dist_map = np.ones((len(single_lanes_dict), len(double_lanes_dict))) * np.inf
    for i in range(len(single_lanes_dict)):
        for j in range(len(double_lanes_dict)):
            dist_map[i, j] = lane_dist_ioua(single_lanes_dict[i], double_lanes_dict[j], x_dist_thres)
    match_res = greedy_search(dist_map, x_dist_thres)
    map_res = dict()
    for match_pair in match_res:
        map_res[match_pair[0]] = match_pair[1]
    filter_single_lanes = list()
    double_lane_idxs = list()
    for idx in range(len(single_lanes)):
        if idx in map_res:
            double_lane_idxs.append(double_lanes_ids[map_res[idx]])
            filter_single_lanes.append(double_lanes[map_res[idx]])
        else:
            double_lane_idxs.append(-1)
            filter_single_lanes.append(single_lanes[idx])
    return filter_single_lanes, double_lane_idxs


def diamond_type_recheck(single_lanes, diamond_lanes, lane_minest_thresh):
    def reset_lanetype(single_lane):
        single_lane_list = []
        for point in single_lane:
            if point['lanetype'] == 6 or point['lanetype'] == 7:
                point['lanetype'] -= 6
            single_lane_list.append(point)
        return single_lane_list

    def get_x_y(lane, th=628):
        x_list, y_list = [], []
        for point in lane:
            if point['y'] > th:  # 对近端判断，远端的忽略
                x_list.append(point['x'])
                y_list.append(point['y'])
        return np.array(x_list), np.array(y_list), len(y_list)

    def diamond_nearby_distance(single_lane, diamond_lanes):
        dist_list = []
        for diamond_lane in diamond_lanes:
            x1, y1, n = get_x_y(single_lane)
            x2, y2, m = get_x_y(diamond_lane)
            if m == 0 or n == 0:
                return 100  # 不记作菱形线
            ar = np.zeros((n, m))
            for i in range(n):  # 欧氏距离
                ar[i, :] = np.sqrt((x2 - x1[i]) ** 2 + (y2 - y1[i]) ** 2)
            dist_list.append(ar.min())
        return np.array(dist_list).min()

    single_lanes_list = []
    for single_lane in single_lanes:
        if single_lane[0]['lanetype'] == 6 or single_lane[0]['lanetype'] == 7:
            if diamond_lanes == []:
                # 恢复实线虚线
                single_lane = reset_lanetype(single_lane)
            elif diamond_nearby_distance(single_lane, diamond_lanes) > lane_minest_thresh:
                # 与最近的菱形线之间距离大于阈值，则恢复实虚线
                single_lane = reset_lanetype(single_lane)
        single_lanes_list.append(single_lane)
    return single_lanes_list


def greedy_search(dist_map, thresh, is_reverse=False):
    match_res = list()
    if dist_map.shape[0] * dist_map.shape[1] == 0:
        return match_res
    if not is_reverse:
        min_pos = np.unravel_index(np.argmin(dist_map), dist_map.shape)
        min_val = dist_map[min_pos]
        while min_val < thresh:
            dist_map[min_pos[0], :] = np.inf
            dist_map[:, min_pos[1]] = np.inf
            match_res.append(min_pos)
            min_pos = np.unravel_index(np.argmin(dist_map), dist_map.shape)
            min_val = dist_map[min_pos]
    else:
        max_pos = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        max_val = dist_map[max_pos]
        while max_val > thresh:
            dist_map[max_pos[0], :] = 0
            dist_map[:, max_pos[1]] = 0
            match_res.append(max_pos)
            max_pos = np.unravel_index(np.argmax(dist_map), dist_map.shape)
            max_val = dist_map[max_pos]
    return match_res


def decodePoses(BW, VAF, HAF, fg_thresh, err_thresh=5, config=dict()):
    corr_pts = list()
    # start decoding from last row to first
    for row in range(BW.shape[0] - 1, -1, -1):  # 从最后一行
        cols = np.where(BW[row, :] > fg_thresh)[0]  # get fg cols
        clusters = [[]]
        if cols.size > 0:
            prev_col = cols[0]
        # parse horizontally 水平方向解析
        for col in cols:
            if col - prev_col > err_thresh:  # if too far away from last point 如果水平距离太远 则构成一个cluster
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0:  # keep moving to the right > 0 说明向右， 终点还在右边 向右移动
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0:  # found lane center, process VAF ， 找到了中点
                clusters[-1].append(col)
                prev_col = col
            elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0:  # found lane end, spawn new lane 找了了新的车道线 产生新的cluster
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] < 0 and HAF[row, col] < 0:  # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue
        # parse vertically
        # assign existing lanes
        cols = list()
        down_offsets = list()
        up_offsets = list()
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            col = np.mean(cluster)
            down_offset = VAF[row, int(col), 0]
            up_offset = VAF[row, int(col), 1]
            cols.append(col)
            down_offsets.append(down_offset)
            up_offsets.append(up_offset)
        corr_pts.append([row, np.asarray(cols), np.asarray(up_offsets), np.asarray(down_offsets)])
    export_lanes = list()
    lanes = list()
    if len(corr_pts) > 0:
        for idx, col in enumerate(corr_pts[0][1]):
            lane = Lane()
            lane.add_coord([col, corr_pts[0][0]])
            lanes.append(lane)
    for idx in range(len(corr_pts) - 1):
        bottom_items = corr_pts[idx]
        up_items = corr_pts[idx + 1]
        bottom_predicts = bottom_items[1].astype(int) + bottom_items[2]
        up_predicts = up_items[1].astype(int) + up_items[3]
        bottom_predicts = bottom_predicts[:, np.newaxis]  # rows is bottom
        up_orig_items = up_items[1][np.newaxis, :]
        bottom_orig_items = bottom_items[1][:, np.newaxis]
        up_predicts = up_predicts[np.newaxis, :]
        dist_map_a = np.abs(bottom_predicts - up_orig_items)
        dist_map_b = np.abs(bottom_orig_items - up_predicts)
        dist_map = 0.5 * dist_map_a + 0.5 * dist_map_b
        err_thresh_ad = err_thresh * ((up_items[0] / (BW.shape[0] - 1))) + 2
        match_res = greedy_search(dist_map, err_thresh_ad)
        bottom_2_up = dict()
        up_2_bottom = dict()
        for match_pair in match_res:
            bottom_2_up[match_pair[0]] = match_pair[1]
            up_2_bottom[match_pair[1]] = match_pair[0]
        for bottom_idx in range(len(bottom_items[1])):
            if bottom_idx not in bottom_2_up:
                lanes[bottom_idx].mark_miss()
            else:
                corr_up_idx = bottom_2_up[bottom_idx]
                lanes[bottom_idx].add_coord([up_items[1][corr_up_idx], up_items[0]])
        for up_idx in range(len(up_items[1])):
            if up_idx not in up_2_bottom:
                lane = Lane()
                lane.add_coord([up_items[1][up_idx], up_items[0]])
                lanes.append(lane)
        export_lanes.extend([x for x in lanes if not x.is_matched])
        lanes = [x for x in lanes if x.is_matched]
        lanes.sort(key=lambda x: x.coords[-1][0])
    export_lanes.extend(lanes)
    if 'do_last_points_fixed' in config and config['do_last_points_fixed']:
        export_lanes = [last_point_fixed(x) for x in export_lanes]
    output = [lane.coords for lane in export_lanes]
    return output


def adjust_stop_lanes(export_lanes, samp_factor_x, samp_factor_y, crop_offset):
    results = []
    for lane in export_lanes:
        if len(lane) < 10:  # mini size is setted 10
            continue
        lane = np.asarray(lane)
        islane_list = []
        lanetype_list = []
        lanecolor_list = []
        for _ in lane:
            lanetype = 3
            islane_list.append(int(lanetype != 0))
            lanetype_list.append(lanetype - 1)
            lanecolor_list.append(0)  # stopline white 0
        lane[:, 0] *= samp_factor_x
        lane[:, 1] *= samp_factor_y
        lane[:, 0]
        lane[:, 1] += crop_offset[1]
        islane_list = np.asarray(islane_list)
        lanetype_list = np.asarray(lanetype_list)
        argsort_idx = np.argsort(lane[:, 1])
        lane = lane[argsort_idx]
        lanetype_list = lanetype_list[argsort_idx]
        islane_list = islane_list[argsort_idx]
        smooth_lane = list()
        coord_idx = 0
        for _x, _y in zip(lane[:, 0], lane[:, 1]):
            if np.isnan(_x):
                continue
            else:
                smooth_lane.append(
                    {
                        'x': _x,
                        'y': _y,
                        'islane': islane_list[coord_idx],
                        'lanetype': lanetype_list[coord_idx],
                        'lanecolor': lanecolor_list[coord_idx],
                    }
                )
            coord_idx += 1
        results.append(smooth_lane)
    return results


class Lane(object):
    def __init__(self) -> None:
        self.coords = []
        self.is_matched = True
        self.center_x = []
        self.center_y = []
        self.is_valid = True

    def mark_miss(self):
        self.is_matched = False

    def add_coord(self, coord):
        self.is_matched = True
        self.coords.append(coord)

    def set_center(self, center_x, center_y):
        self.center_x.append(center_x)
        self.center_y.append(center_y)

    def sort_coords(self):
        coords = np.asarray(self.coords)
        arg_idxs = np.argsort(-coords[:, 1])
        coords[:, 0] = coords[arg_idxs, 0]
        coords[:, 1] = coords[arg_idxs, 1]
        self.coords = coords


def decodePosesHorizon(BW, VAF, HAF, fg_thresh, err_thresh=5):
    corr_pts = list()
    # start decoding from last row to first
    for col in range(BW.shape[1] - 1, -1, -1):  # 从最后一列
        rows = np.where(BW[:, col] > fg_thresh)[0]  # get fg rows
        clusters = [[]]
        if rows.size > 0:
            prev_row = rows[0]
        # parse horizontally 水平方向解析
        for row in rows:
            if row - prev_row > err_thresh:  # if too far away from last point 如果水平距离太远 则构成一个cluster
                clusters.append([])
                clusters[-1].append(row)
                prev_row = row
                continue
            if HAF[prev_row, col] >= 0 and HAF[row, col] >= 0:  # keep moving to the right > 0 说明向右， 终点还在右边 向右移动
                clusters[-1].append(row)
                prev_row = row
                continue
            elif HAF[prev_row, col] >= 0 and HAF[row, col] < 0:  # found lane center, process VAF ， 找到了中点
                clusters[-1].append(row)
                prev_row = row
            elif HAF[prev_row, col] < 0 and HAF[row, col] >= 0:  # found lane end, spawn new lane 找了了新的车道线 产生新的cluster
                clusters.append([])
                clusters[-1].append(row)
                prev_row = row
                continue
            elif HAF[prev_row, col] < 0 and HAF[row, col] < 0:  # keep moving to the right
                clusters[-1].append(row)
                prev_row = row
                continue
        # parse vertically
        # assign existing lanes
        rows = list()
        left_offsets = list()
        right_offsets = list()
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            row = np.mean(cluster)
            left_offset = VAF[int(row), col, 0]
            right_offset = VAF[int(row), col, 1]
            rows.append(row)
            left_offsets.append(left_offset)
            right_offsets.append(right_offset)
        corr_pts.append([col, np.asarray(rows), np.asarray(left_offsets), np.asarray(right_offsets)])
    export_lanes = list()
    lanes = list()
    if len(corr_pts) > 0:
        for idx, row in enumerate(corr_pts[0][1]):
            lane = Lane()
            lane.add_coord([corr_pts[0][0], row])
            lanes.append(lane)
    for idx in range(len(corr_pts) - 1):
        left_items = corr_pts[idx]
        right_items = corr_pts[idx + 1]
        left_predicts = left_items[1].astype(int) + left_items[2]
        right_predicts = right_items[1].astype(int) + right_items[3]
        left_predicts = left_predicts[:, np.newaxis]  # rows is bottom
        right_orig_items = right_items[1][np.newaxis, :]
        left_orig_items = left_items[1][:, np.newaxis]
        right_predicts = right_predicts[np.newaxis, :]
        dist_map_a = np.abs(left_predicts - right_orig_items)
        dist_map_b = np.abs(left_orig_items - right_predicts)
        dist_map = 0.5 * dist_map_a + 0.5 * dist_map_b
        match_res = greedy_search(dist_map, 5.0)
        left_2_right = dict()
        right_2_left = dict()
        for match_pair in match_res:
            left_2_right[match_pair[0]] = match_pair[1]
            right_2_left[match_pair[1]] = match_pair[0]
        for left_idx in range(len(left_items[1])):
            if left_idx not in left_2_right:
                lanes[left_idx].mark_miss()
            else:
                corr_right_idx = left_2_right[left_idx]
                lanes[left_idx].add_coord([right_items[0], right_items[1][corr_right_idx]])
        for right_idx in range(len(right_items[1])):
            if right_idx not in right_2_left:
                lane = Lane()
                lane.add_coord([right_items[0], right_items[1][right_idx]])
                lanes.append(lane)
        export_lanes.extend([x for x in lanes if not x.is_matched])
        lanes = [x for x in lanes if x.is_matched]
        lanes.sort(key=lambda x: x.coords[-1][1])
    export_lanes.extend(lanes)
    output = [lane.coords for lane in export_lanes]
    return output


def get_y_thresh(y):
    max_y = 900
    min_y = 610
    min_len_thresh = 10
    max_len_thresh = 20
    if y > max_y:
        y_thresh = max_len_thresh
    elif y < min_y:
        y_thresh = min_len_thresh
    else:
        y_thresh = (y - min_y) / (max_y - min_y) * (max_len_thresh - min_len_thresh) + min_len_thresh
    return y_thresh


def check_valid(min_y, lane_length, lane_dis):
    dyn_y_thresh = get_y_thresh(min_y)
    if lane_length > dyn_y_thresh or lane_dis > 200:
        return True
    else:
        return False


def adjust_lanes(
    export_lanes, samp_factor_x, samp_factor_y, crop_offset, feat_height, feat_width, lanetype_mat, lanecolor_mat
):
    results = []
    for lane in export_lanes:
        lane = np.asarray(lane)
        if not check_valid(
            np.min(lane[:, 1]), len(lane), np.linalg.norm([lane[0][0] - lane[-1][0], lane[0][1] - lane[-1][1]])
        ):
            continue
        islane_list = []
        lanetype_list = []
        lanecolor_list = []
        for coord in lane:
            x, y = int(coord[0]), int(coord[1])
            x = max(0, min(x, feat_width - 1))
            y = max(0, min(y, feat_height - 1))
            if isinstance(lanetype_mat, np.ndarray):
                lanetype = np.argmax(lanetype_mat[y, x]) + 1
                if lanetype == 3:
                    lanetype = 6
                if lanetype == 4:
                    lanetype = 7
            else:
                assert isinstance(lanetype_mat, int)
                lanetype = lanetype_mat

            if isinstance(lanecolor_mat, np.ndarray):
                lanecolor = np.argmax(lanecolor_mat[y, x])
            else:
                assert isinstance(lanecolor_mat, int)
                lanecolor = lanecolor_mat
            islane_list.append(int(lanetype != 0))
            lanetype_list.append(lanetype - 1)
            lanecolor_list.append(lanecolor)
        lane[:, 0] *= samp_factor_x
        lane[:, 1] *= samp_factor_y
        lane[:, 0]
        lane[:, 1] += crop_offset[1]
        islane_list = np.asarray(islane_list)
        lanetype_list = np.asarray(lanetype_list)
        lanecolor_list = np.asarray(lanecolor_list)
        argsort_idx = np.argsort(lane[:, 1])
        lane = lane[argsort_idx]
        lanetype_list = lanetype_list[argsort_idx]
        lanecolor_list = lanecolor_list[argsort_idx]
        islane_list = islane_list[argsort_idx]
        f = CubicSpline(lane[:, 1], lane[:, 0], extrapolate=False)
        if f is not None:
            smooth_lane = []
            smooth_xs = [f(y) for y in lane[:, 1]]
            coord_idx = 0
            for _x, _y in zip(smooth_xs, lane[:, 1]):
                if np.isnan(_x):
                    continue
                else:
                    smooth_lane.append(
                        {
                            'x': _x,
                            'y': _y,
                            'islane': islane_list[coord_idx],
                            'lanetype': lanetype_list[coord_idx],
                            'lanecolor': lanecolor_list[coord_idx],
                        }
                    )
                coord_idx += 1
            results.append(smooth_lane)
    return results


def last_point_fixed(lane, front_start_idx=5, front_end_idx=9, back_start_idx=-6, back_end_idx=-10):
    if len(lane.coords) < 10:
        return lane
    # start_pts
    front_delta_x = (lane.coords[front_end_idx][0] - lane.coords[front_start_idx][0]) / (
        front_end_idx - front_start_idx
    )
    front_delta_y = (lane.coords[front_end_idx][1] - lane.coords[front_start_idx][1]) / (
        front_end_idx - front_start_idx
    )
    # end_pts
    back_delta_x = (lane.coords[back_start_idx][0] - lane.coords[back_end_idx][0]) / (back_start_idx - back_end_idx)
    back_delta_y = (lane.coords[back_start_idx][1] - lane.coords[back_end_idx][1]) / (back_start_idx - back_end_idx)

    for front_idx in range(front_start_idx):
        lane.coords[front_idx][0] = lane.coords[front_start_idx][0] - (front_start_idx - front_idx) * front_delta_x
        lane.coords[front_idx][1] = lane.coords[front_start_idx][1] - (front_start_idx - front_idx) * front_delta_y

    for back_idx in range(back_start_idx + 1, 0):
        lane.coords[back_idx][0] = lane.coords[back_start_idx][0] + (back_idx - back_start_idx) * back_delta_x
        lane.coords[back_idx][1] = lane.coords[back_start_idx][1] + (back_idx - back_start_idx) * back_delta_y
    return lane


def decodePoses_bias(BW, VAF, HAF, hbias, vbias, fg_thresh, err_thresh=5, config=dict()):
    corr_pts = list()
    height, width = BW.shape[0], BW.shape[1]
    # start decoding from last row to first
    for row in range(BW.shape[0] - 1, -1, -1):  # 从最后一行
        centers = [[[], []]]
        center_points = []
        cols = np.where(BW[row, :] > fg_thresh)[0]  # get fg cols
        clusters = [[]]
        if cols.size > 0:
            prev_col = cols[0]
        # parse horizontally 水平方向解析
        for col in cols:
            cur_center = [hbias[row, col], vbias[row, col]]
            if col - prev_col > err_thresh:  # if too far away from last point 如果水平距离太远 则构成一个cluster
                if len(clusters) > 0:
                    center_points.append([np.mean(centers[-1][0]), np.mean(centers[-1][1])])
                clusters.append([])
                clusters[-1].append(col)
                centers.append([[], []])
                centers[-1][0].append(cur_center[0])
                centers[-1][1].append(cur_center[1])
                prev_col = col
                continue
            if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0:  # keep moving to the right > 0 说明向右， 终点还在右边 向右移动
                clusters[-1].append(col)
                centers[-1][0].append(cur_center[0])
                centers[-1][1].append(cur_center[1])
                prev_col = col
                continue
            elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0:  # found lane center, process VAF ， 找到了中点
                clusters[-1].append(col)
                centers[-1][0].append(cur_center[0])
                centers[-1][1].append(cur_center[1])
                prev_col = col
            elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0:  # found lane end, spawn new lane 找了了新的车道线 产生新的cluster
                if len(clusters) > 0:
                    center_points.append([np.mean(centers[-1][0]), np.mean(centers[-1][1])])
                clusters.append([])
                clusters[-1].append(col)
                centers.append([[], []])
                centers[-1][0].append(cur_center[0])
                centers[-1][1].append(cur_center[1])
                prev_col = col
                continue
            elif HAF[row, prev_col] < 0 and HAF[row, col] < 0:  # keep moving to the right
                clusters[-1].append(col)
                centers[-1][0].append(cur_center[0])
                centers[-1][1].append(cur_center[1])
                prev_col = col
                continue
        if len(clusters) > 0:
            center_points.append([np.mean(centers[-1][0]), np.mean(centers[-1][1])])
        # parse vertically
        # assign existing lanes
        cols = list()
        cur_centers_x = list()
        cur_centers_y = list()
        down_offsets = list()
        up_offsets = list()
        for cluster, center_point in zip(clusters, center_points):
            if len(cluster) == 0:
                continue
            col = np.mean(cluster)
            down_offset = VAF[row, int(col), 0]
            up_offset = VAF[row, int(col), 1]
            cols.append(col)
            down_offsets.append(down_offset)
            up_offsets.append(up_offset)
            cur_centers_x.append(-center_point[0] / 100 * width + int(col))
            cur_centers_y.append(-center_point[1] / 100 * height + row)
        corr_pts.append(
            [
                row,
                np.asarray(cols),
                np.asarray(up_offsets),
                np.asarray(down_offsets),
                np.asarray(cur_centers_x),
                np.asarray(cur_centers_y),
            ]
        )
    export_lanes = list()
    lanes = list()
    if len(corr_pts) > 0:
        for col, center_x, center_y in zip(corr_pts[0][1], corr_pts[0][4], corr_pts[0][5]):
            lane = Lane()
            lane.add_coord([col, corr_pts[0][0]])
            lane.set_center(center_x, center_y)
            lanes.append(lane)
    for idx in range(len(corr_pts) - 1):
        bottom_items = corr_pts[idx]
        up_items = corr_pts[idx + 1]
        bottom_predicts = bottom_items[1].astype(int) + bottom_items[2]
        up_predicts = up_items[1].astype(int) + up_items[3]
        bottom_predicts = bottom_predicts[:, np.newaxis]  # rows is bottom
        up_orig_items = up_items[1][np.newaxis, :]
        bottom_orig_items = bottom_items[1][:, np.newaxis]
        up_predicts = up_predicts[np.newaxis, :]
        dist_map_a = np.abs(bottom_predicts - up_orig_items)
        dist_map_b = np.abs(bottom_orig_items - up_predicts)
        dist_map = 0.5 * dist_map_a + 0.5 * dist_map_b
        err_thresh_ad = err_thresh * ((up_items[0] / (BW.shape[0] - 1))) + 2
        match_res = greedy_search(dist_map, err_thresh_ad)
        bottom_2_up = dict()
        up_2_bottom = dict()
        for match_pair in match_res:
            bottom_2_up[match_pair[0]] = match_pair[1]
            up_2_bottom[match_pair[1]] = match_pair[0]
        for bottom_idx in range(len(bottom_items[1])):
            if bottom_idx not in bottom_2_up:
                lanes[bottom_idx].mark_miss()
            else:
                corr_up_idx = bottom_2_up[bottom_idx]
                lanes[bottom_idx].add_coord([up_items[1][corr_up_idx], up_items[0]])
                lanes[bottom_idx].set_center(up_items[4][corr_up_idx], up_items[5][corr_up_idx])
        for up_idx in range(len(up_items[1])):
            if up_idx not in up_2_bottom:
                lane = Lane()
                lane.add_coord([up_items[1][up_idx], up_items[0]])
                lane.set_center(up_items[4][up_idx], up_items[5][up_idx])
                lanes.append(lane)
        export_lanes.extend([x for x in lanes if not x.is_matched])
        lanes = [x for x in lanes if x.is_matched]
        lanes.sort(key=lambda x: x.coords[-1][0])
    export_lanes.extend(lanes)
    for lane in lanes:
        lane.sort_coords()
    # 使用bias预测连接断连的车道线
    out_lanes = list()
    for i in range(len(export_lanes) - 1):
        for j in range(i + 1, len(export_lanes)):
            lane1 = export_lanes[i]
            lane2 = export_lanes[j]
            if lane1.is_valid and lane2.is_valid:
                lane1_y = []
                lane1_x = []
                lane2_y = []
                lane2_x = []
                for coord in lane1.coords:
                    lane1_x.append(coord[0])
                    lane1_y.append(coord[1])
                for coord in lane2.coords:
                    lane2_x.append(coord[0])
                    lane2_y.append(coord[1])
                lane1_y_argsort_idx = np.argsort(lane1_y)
                lane1_y_max = lane1_y[lane1_y_argsort_idx[-1]]
                lane1_y_min = lane1_y[lane1_y_argsort_idx[0]]
                lane1_y_max_corr_x = lane1_x[lane1_y_argsort_idx[-1]]
                lane1_y_min_corr_x = lane1_x[lane1_y_argsort_idx[0]]
                lane2_y_argsort_idx = np.argsort(lane2_y)
                lane2_y_max = lane2_y[lane2_y_argsort_idx[-1]]
                lane2_y_min = lane2_y[lane2_y_argsort_idx[0]]
                lane2_y_max_corr_x = lane2_x[lane2_y_argsort_idx[-1]]
                lane2_y_min_corr_x = lane2_x[lane2_y_argsort_idx[0]]
                # 若2条线在y轴上存在重合则不连接
                is_covered = False
                if lane1_y_max >= lane2_y_max and lane1_y_min <= lane2_y_max:
                    is_covered = True
                if lane2_y_max >= lane1_y_max and lane2_y_min <= lane1_y_max:
                    is_covered = True
                if len(lane1.coords) <= 3 or len(lane2.coords) <= 3:
                    is_covered = True
                # 判断两条线y轴最近的两点的距离，距离近则进行连接
                if not is_covered:
                    if lane2_y_max > lane1_y_max:
                        merged_pts = (
                            [lane2_y_max_corr_x, lane2_y_max],
                            [lane2_y_min_corr_x, lane2_y_min],
                            [lane1_y_max_corr_x, lane1_y_max],
                            [lane1_y_min_corr_x, lane1_y_min],
                        )
                    else:
                        merged_pts = (
                            [lane1_y_max_corr_x, lane1_y_max],
                            [lane1_y_min_corr_x, lane1_y_min],
                            [lane2_y_max_corr_x, lane2_y_max],
                            [lane2_y_min_corr_x, lane2_y_min],
                        )
                    vec1 = np.asarray([merged_pts[0][0] - merged_pts[3][0], merged_pts[0][1] - merged_pts[3][1]])
                    vec2 = np.asarray([merged_pts[1][0] - merged_pts[2][0], merged_pts[1][1] - merged_pts[2][1]])
                    cosangle = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cosangle = np.arccos(cosangle) / np.pi * 180
                    vec2_len = np.linalg.norm(vec2)
                    lane1_center_x0 = lane1.center_x[0]
                    lane1_center_x1 = lane1.center_x[-1]
                    lane2_center_x0 = lane2.center_x[0]
                    lane2_center_x1 = lane2.center_x[-1]
                    lane1_center_y0 = lane1.center_y[0]
                    lane1_center_y1 = lane1.center_y[-1]
                    lane2_center_y0 = lane2.center_y[0]
                    lane2_center_y1 = lane2.center_y[-1]
                    merge_lane = False
                    if (
                        lane1_y_max > lane2_y_max
                        and abs(lane1_center_x1 - lane2_center_x0) < 20
                        and abs(lane1_center_y1 - lane2_center_y0) < 10
                    ):
                        merge_lane = True
                    if (
                        lane1_y_max < lane2_y_max
                        and abs(lane1_center_x0 - lane2_center_x1) < 20
                        and abs(lane1_center_y0 - lane2_center_y1) < 10
                    ):
                        merge_lane = True
                    if merge_lane and cosangle > 90 and vec2_len > 9:
                        merge_lane = False
                    if merge_lane:
                        for coord in lane2.coords:
                            lane1.add_coord(coord)
                        lane1.coords.sort(key=lambda x: x[1])
                        lane2.is_valid = False
    for lane in export_lanes:
        if lane.is_valid:
            out_lanes.append(lane)
    export_lanes = out_lanes
    if 'do_last_points_fixed' in config and config['do_last_points_fixed']:
        export_lanes = [last_point_fixed(x) for x in export_lanes]
    output = [lane.coords for lane in export_lanes]
    return output


def tensor2image(tensor):
    image = np.transpose(tensor[0].cpu().float().numpy(), (1, 2, 0))  # (C, H, W) to (H, W, C)
    return image


def get_frequenest_item(items):
    values, cnts = np.unique(items, return_counts=True)
    return values[np.argmax(cnts)]


def get_results(parsed_lanes, parsed_double_ids, valid_pts, image_shape, scale):
    assert len(parsed_lanes) == len(parsed_double_ids)
    lanes = list()
    lane_types = list()
    lane_colors = list()
    is_lanes = list()
    double_ids = list()
    for i, lane in enumerate(parsed_lanes):
        parsed_lane = list()
        islane_list = list()
        lanetype_list = list()
        lanecolor_list = list()
        min_y = 100000000
        max_y = -1
        for j in range(len(lane)):
            x = int(lane[j]['x'] * scale[1])
            y = int(lane[j]['y'] * scale[0])
            x = max(0, min(x, image_shape[1] - 1))
            y = max(0, min(y, image_shape[0] - 1))
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            islane_list.append(lane[j]['islane'])
            lanetype_list.append(lane[j]['lanetype'])
            lanecolor_list.append(lane[j]['lanecolor'])
            parsed_lane.extend([int(x), int(y)])
        if min_y < int(image_shape[0] * 0.5) and max_y < int(image_shape[0] * 0.9):
            continue
        freq_islane = get_frequenest_item(islane_list)
        freq_lanetype = get_frequenest_item(lanetype_list)
        freq_lanecolor = get_frequenest_item(lanecolor_list)
        lanes.append(parsed_lane)
        lane_types.append(freq_lanetype)
        lane_colors.append(freq_lanecolor)
        is_lanes.append(freq_islane)
        double_ids.append(parsed_double_ids[i])
    lane_names = []
    for j in range(len(lane_types)):
        if is_lanes[j] == 0:
            assert lane_types[j] == -1
            lane_name = ['road_edge']
        elif is_lanes[j] == 1:
            assert lane_types[j] != -1
            if double_ids[j] != -1:
                assert double_ids[j] in (1, 2)
                if lane_types[j] in (0, 5):
                    lane_name = ['lane_line', 'double', 'solid']
                elif lane_types[j] in (1, 4, 6):
                    lane_name = ['lane_line', 'double', 'dot']
                else:
                    assert False
            elif lane_types[j] == 0:
                lane_name = ['lane_line', 'single', 'solid']
            elif lane_types[j] == 1:
                lane_name = ['lane_line', 'single', 'dot']
            elif lane_types[j] == 2:
                lane_name = ['stop_line']
            elif lane_types[j] == 4:
                lane_name = ['lane_line', 'diamoid', 'solid_dot']
            elif lane_types[j] == 5:
                lane_name = ['lane_line', 'diamoid', 'solid']
            elif lane_types[j] == 6:
                lane_name = ['lane_line', 'diamoid', 'dot']
            else:
                assert False
        else:
            assert False
        if lane_colors[j] == 0:
            lane_name.append('white')
        elif lane_colors[j] == 1:
            lane_name.append('yellow')
        else:
            assert False
        lane_names.append(lane_name)
    return lanes, lane_names
