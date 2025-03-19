import cv2
import torch
# 以下参数需调整
bg_color = 0
panel_w, panel_h, panel_color = 16, 4, 29

panel = torch.ones((panel_h, panel_w)) * panel_color
panel_wrap = torch.ones((panel_h + 2, panel_w + 2)) * bg_color
panel_wrap[1:panel_h + 1, 1:panel_w + 1] = panel
kernel_panel_left = panel_wrap[:, :9] # 这里的 // 2可以调整
kernel_panel_right = panel_wrap[:, -9:] # 这里的 // 2可以调整

ball_w, ball_h, ball_color = 2, 4, 29
kernal_ball = torch.ones((ball_h, ball_w)) * ball_color
kernal_ball_wrap = torch.ones((ball_h + 2, ball_w + 2)) * bg_color
kernal_ball_wrap[1:ball_h + 1, 1:ball_w + 1] = kernal_ball
kernal_ball_left = kernal_ball_wrap[:, :2] # 这里的 2可以调整
kernel_ball_right = kernal_ball_wrap[:, -2:] # 这里的 2可以调整
kernal_ball_top = kernal_ball_wrap[:2, :] # 这里的 2可以调整
kernal_ball_bottom = kernal_ball_wrap[-2:, :] # 这里的 2可以调整

def find_subimg(img, sub):
    h, w = sub.shape
    for i in range(img.shape[0] - h + 1):
        for j in range(img.shape[1] - w + 1):
            sub_region = img[i:i+h, j:j+w]
            if torch.equal(torch.from_numpy(sub_region), sub):
                found_small_image = True
                return i, j
    return None, None

def getCoord(img):
    ball_x, ball_y, panel_x, panel_y = None, None, None, None # xy为物体左上角相对于图片左上角坐标
    if find_subimg(img, kernal_ball_left)[0] is not None:
        ball_x, ball_y = find_subimg(img, kernal_ball_left)
        ball_x += 1
        ball_y += 1
    elif find_subimg(img, kernel_ball_right)[0] is not None:
        ball_x, ball_y = find_subimg(img, kernel_ball_right)
        ball_x = ball_x + kernel_ball_right.shape[1] - 1 - ball_w
        ball_y += 1
    elif find_subimg(img, kernal_ball_top)[0] is not None:
        ball_x, ball_y = find_subimg(img, kernal_ball_top)
        ball_x += 1
        ball_y += 1
    elif find_subimg(img, kernal_ball_bottom)[0] is not None:
        ball_x, ball_y = find_subimg(img, kernal_ball_bottom)
        ball_x += 1
        ball_y = ball_y + kernal_ball_top.shape[0] - 1 - ball_h

    if find_subimg(img, kernel_panel_left)[0] is not None:
        panel_x, panel_y = find_subimg(img, kernel_panel_left)
        panel_x += 1
        panel_y += 1
    elif find_subimg(img, kernel_panel_right)[0] is not None:
        panel_x, panel_y = find_subimg(img, kernel_panel_right)
        panel_x = panel_x + kernel_panel_right.shape[1] - 1 - panel_w
        panel_y += 1
    if ball_x == panel_x and ball_y == panel_y:
        ball_x, ball_y = None, None
    return ball_x, ball_y, panel_x, panel_y