import torch
import torch.nn.functional as F


def window_sum(cam_sum, cam_cnt, cam, anchor_coordinate=(0, 0)):
    c, h, w = cam_sum.shape
    _, window_width_x, window_width_y = cam.shape
    anchor_x, anchor_y = anchor_coordinate
    a_start_x = max(0, -anchor_x)
    a_start_y = max(0, -anchor_y)
    b_start_x = max(0, anchor_x)
    b_start_y = max(0, anchor_y)
    a_end_x = min(window_width_x, a_start_x + h - b_start_x)
    a_end_y = min(window_width_y, a_start_y + w - b_start_y)

    cam_sum[:, b_start_x:b_start_x + (a_end_x - a_start_x), b_start_y:b_start_y + (a_end_y - a_start_y)] += \
        cam[:, a_start_x:a_end_x, a_start_y:a_end_y]
    cam_cnt[:, b_start_x:b_start_x + (a_end_x - a_start_x), b_start_y:b_start_y + (a_end_y - a_start_y)] += 1

    return cam_sum, cam_cnt


class RebuildMultiScaledCAM(object):
    def __init__(self, target_size=448, n_scales=3):
        self.target_size = target_size
        assert n_scales in [1, 3, 5, 7]
        self.n_scales = n_scales

    def __call__(self, cams, target_h, target_w):
        b, c, h, w = cams.shape
        edge_l = max(target_h, target_w)
        edge_s = min(target_h, target_w)

        # long edge scale
        anchor_x = min((target_h - edge_l) // 2, 0)
        anchor_y = min((target_w - edge_l) // 2, 0)

        cam_sum = F.interpolate(cams[0:1], size=(edge_l, edge_l), mode='bilinear').squeeze()
        cam_sum = cam_sum[:, -anchor_x:h - anchor_x, -anchor_y:w - anchor_y]
        cam_cnt = torch.ones_like(cam_sum).type(dtype=torch.uint8)

        if self.n_scales > 2:  # short edge scale
            cur_cams = F.interpolate(cams[1:3], size=(edge_s, edge_s), mode='bilinear')
            anchor_x = max(target_h - edge_s, 0)
            anchor_y = max(target_w - edge_s, 0)
            cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[0], anchor_coordinate=(0, 0))
            cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[1], anchor_coordinate=(anchor_x, anchor_y))

        if self.n_scales > 4:  # half long edge scale
            if self.n_scales == 5:  # 2 samples
                cur_cams = F.interpolate(cams[3:5], size=(edge_l // 2, edge_l // 2), mode='bilinear')
                if target_w > target_h:
                    anchor_x = (target_h - edge_l // 2) // 2
                    anchor_y = edge_l // 2
                    cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[0], anchor_coordinate=(anchor_x, 0))
                    cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[1], anchor_coordinate=(anchor_x, anchor_y))
                else:
                    anchor_y = (target_w - edge_l // 2) // 2
                    anchor_x = edge_l // 2
                    cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[0], anchor_coordinate=(0, anchor_y))
                    cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[1], anchor_coordinate=(anchor_x, anchor_y))

            else:  # 4 samples
                cur_cams = F.interpolate(cams[3:7], size=(edge_l // 2, edge_l // 2), mode='bilinear')
                anchor_x = target_h - edge_l // 2
                anchor_y = target_w - edge_l // 2
                cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[0], anchor_coordinate=(0, 0))
                cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[1], anchor_coordinate=(anchor_x, 0))
                cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[2], anchor_coordinate=(0, anchor_y))
                cam_sum, cam_cnt = window_sum(cam_sum, cam_cnt, cur_cams[3], anchor_coordinate=(anchor_x, anchor_y))

        return cam_sum / cam_cnt

