"""Point-cloud reconstruction metrics (chamfer, Hausdorff, F-score)."""

import numpy as np


def compute_chamfer_distance_from_pcds(pcd1, pcd2):
    """
    Computes the Chamfer Distance between two Open3D point clouds.
    Chamfer Distance = mean(min_dist(pcd1 -> pcd2)) + mean(min_dist(pcd2 -> pcd1))
    """
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        if len(pcd1.points) == len(pcd2.points):  # both empty
            return 0.0, 0.0, 0.0
        return float("inf"), float("inf"), float("inf")

    dists1_to_2 = pcd1.compute_point_cloud_distance(pcd2)
    mean_dist1_to_2 = np.mean(np.asarray(dists1_to_2))

    dists2_to_1 = pcd2.compute_point_cloud_distance(pcd1)
    mean_dist2_to_1 = np.mean(np.asarray(dists2_to_1))

    chamfer_distance = mean_dist1_to_2 + mean_dist2_to_1
    return chamfer_distance, mean_dist1_to_2, mean_dist2_to_1


def compute_hausdorff_distance_from_pcds(pcd1, pcd2):
    """
    Computes the Hausdorff Distance between two Open3D point clouds.
    Hausdorff Distance = max(max(min_dist(pcd1 -> pcd2)), max(min_dist(pcd2 -> pcd1)))
    """
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        if len(pcd1.points) == len(pcd2.points):  # both empty
            return 0.0
        return float("inf")

    dists1_to_2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    hausdorff_1_to_2 = np.max(dists1_to_2)

    dists2_to_1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    hausdorff_2_to_1 = np.max(dists2_to_1)

    hausdorff_distance = max(hausdorff_1_to_2, hausdorff_2_to_1)
    return hausdorff_distance


def compute_f1_score_from_pcds(pcd_pred, pcd_gt, threshold):
    """
    Computes the F1-Score between a predicted point cloud and a ground truth point cloud.
    Precision: Percentage of points in pcd_pred within threshold to pcd_gt.
    Recall: Percentage of points in pcd_gt within threshold to pcd_pred.
    """
    len_pred = len(pcd_pred.points)
    len_gt = len(pcd_gt.points)

    if len_pred == 0 and len_gt == 0:
        return 1.0, 1.0, 1.0

    if len_pred == 0:
        precision = 1.0
        recall = 0.0
        return 0.0, precision, recall

    if len_gt == 0:
        precision = 0.0
        recall = 1.0
        return 0.0, precision, recall

    dists_pred_to_gt = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    precision = np.sum(dists_pred_to_gt <= threshold) / len_pred

    dists_gt_to_pred = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    recall = np.sum(dists_gt_to_pred <= threshold) / len_gt

    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score, precision, recall
