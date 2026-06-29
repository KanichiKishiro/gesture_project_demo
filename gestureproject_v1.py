#!/usr/bin/env python3
import cv2
import time
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import can
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


# =========================
# 全局参数
# =========================
TARGET_FPS = 30
frame_dt   = 1.0 / TARGET_FPS

is_following = False
lost_frames  = 0
invalid_depth_frames = 0     # 连续深度为 0 的帧计数
MAX_INVALID_DEPTH = 7        # 连续 7 帧深度无效 → 停车
MAX_LOST_FRAMES  = 10        # 连续 10 帧无pose → 停车

last_cmdVL = 0
last_cmdVR = 0

TOO_CLOSE_STOP = 1.0    # <1m 强制停止
HAND_FORWARD_Z = 0.20   # 伸手阈值 30cm
ARM_UP_Y       = 0.05   # 前左右动作的“抬高手腕”阈值（相对手肘）


# =========================
# 指さしモード（第1段階：画像座標で左右方位のみ）
# =========================
# 設計方針:
#   - 操舵は steer_from_lateral(lateral, forward) に一本化。
#     第1段階は lateral を「指先の画像左右位置」から作る。
#     第2段階で 3D 方位角に、第3段階で forward を仰角に差し替えれば、
#     この操舵・安全・UI 部分はそのまま再利用できる。
#   - 開始条件は既存ジェスチャー（伸手の甲/掌=追従、両手前=後退、
#     手首を肘より上=前進/左右）と衝突しないように選ぶ。

POINTING_ENABLED   = True     # 指さしモードを有効にするか
POINT_HOLD_FRAMES  = 8        # 指さし姿勢を連続何フレーム保持で「確定」とみなすか
POINT_RELEASE_FRAMES = 5      # 指さし姿勢が連続何フレーム消えたら解除するか
POINT_FINGER_EXT   = 0.06     # 人差し指が「伸びている」と判定する TIP-MCP 距離(正規化)
POINT_WRIST_BELOW_SH = 0.03   # 手首が肩より下にある（=上げていない）余裕
POINT_GAIN         = 12.0     # 指さし方位 → 旋回ゲイン（追従の 10 に対し少し強め）
POINT_BASE_SPEED   = 2500     # 指さし誘導中の前進ベース速度
POINT_LR_DEADBAND  = 0.04     # 中央デッドバンド（正規化 |lateral| がこれ未満は直進）

is_pointing       = False     # 現在指さし誘導中か
point_hold_cnt    = 0         # 指さし姿勢の連続検出カウント
point_release_cnt = 0         # 指さし姿勢の連続消失カウント
point_side        = None      # 'L' or 'R' どちらの手で指しているか
point_lateral_view = 0.0   # UI表示用：直近の指さし左右量（追加）


# =========================
# Mediapipe 初始化
# =========================
mp_holistic = mp.solutions.holistic
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles


# =========================
# RealSense 初始化
# =========================
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile        = pipeline.start(config)
depth_scale    = profile.get_device().first_depth_sensor().get_depth_scale()
align_to_color = rs.align(rs.stream.color)


# =========================
# 视频保存初始化
# =========================
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cur_time = time.strftime("%m%d%H%M", time.localtime())
out = cv2.VideoWriter(f"/home/shinichiro/research/gesture_project/recorder/{cur_time}.avi",
                      fourcc, TARGET_FPS, (640, 480))


#bus = can.interface.Bus(channel='can0', bustype='socketcan')


# =========================
# CAN
# =========================
def send_can(cmdVL, cmdVR):
    Bd0 = (cmdVL >> 8) & 0x00ff
    Bd1 = cmdVL & 0x00ff
    Bd2 = (cmdVR >> 8) & 0x00ff
    Bd3 = cmdVR & 0x00ff

    message = can.Message(
        arbitration_id=0x8F01110,
        is_extended_id=True,
        data=[Bd0, Bd1, Bd2, Bd3, 0, 0, 0, 0]
    )
    try:
        bus.send(message)
    except:
        pass


# =========================
# 深度查询函数（坑A修复：只依赖 depth_img 自己的尺寸）
# =========================
def depth_at(pt, depth_img):
    h, w = depth_img.shape[:2]
    u = int(np.clip(pt.x * w, 0, w - 1))
    v = int(np.clip(pt.y * h, 0, h - 1))
    z = depth_img[v, u]
    return float(z) * depth_scale if z > 0 else 0.0


def depth_at_xy(x_norm, y_norm, depth_img):
    h, w = depth_img.shape[:2]
    u = int(np.clip(x_norm * w, 0, w - 1))
    v = int(np.clip(y_norm * h, 0, h - 1))
    z = depth_img[v, u]
    return float(z) * depth_scale if z > 0 else 0.0


# =========================
# 判断是否伸手（用“人体中心深度” vs “手腕深度”）
# =========================
def is_hand_forward(person_z, wrist_pt, depth_img):
    z_wr = depth_at(wrist_pt, depth_img)
    if person_z <= 0 or z_wr <= 0:
        return False
    return (person_z - z_wr) > HAND_FORWARD_Z


# =========================
# 左右手手心/手背（区分左右）
# =========================
def right_hand_facing(hand):
    if hand is None:
        return None
    rh = hand.landmark
    thumb = rh[mp_holistic.HandLandmark.THUMB_TIP]
    pinky = rh[mp_holistic.HandLandmark.PINKY_TIP]
    return "back" if thumb.x < pinky.x else "palm"


def left_hand_facing(hand):
    if hand is None:
        return None
    lh = hand.landmark
    thumb = lh[mp_holistic.HandLandmark.THUMB_TIP]
    pinky = lh[mp_holistic.HandLandmark.PINKY_TIP]
    return "back" if thumb.x > pinky.x else "palm"


# =========================
# 指さし検出（第1段階）
# =========================
def index_finger_extended(hand):
    """人差し指が伸びているか（TIP が MCP から十分離れている）。
    第1段階では指さし"方向"までは使わず、"指さしているか"の判定にのみ使う。"""
    if hand is None:
        return False
    lm = hand.landmark
    tip = lm[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    mcp = lm[mp_holistic.HandLandmark.INDEX_FINGER_MCP]
    d = ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5
    return d > POINT_FINGER_EXT


def detect_pointing(lm, left_hand, right_hand):
    """指さし姿勢を検出し、(side, lateral) を返す。
    side    : 'L' / 'R' / None
    lateral : 指さしの左右量。正=右、負=左。画面中心からの正規化オフセット。
    既存ジェスチャーと衝突しない条件:
      - 対象の手首が肩より下（=「手首を肘より上」前進系と非衝突）
      - 手が前に大きく突き出ていない（=伸手の追従/後退と非衝突）は
        呼び出し側で is_hand_forward を見て弾く
      - 人差し指が伸びている（単なる手振りと区別）
    """
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # y は下向きが大きい。手首が肩より下 = wrist.y > shoulder.y + 余裕
    l_below = l_wr.y > (l_sh.y + POINT_WRIST_BELOW_SH)
    r_below = r_wr.y > (r_sh.y + POINT_WRIST_BELOW_SH)

    l_point = l_below and index_finger_extended(left_hand)
    r_point = r_below and index_finger_extended(right_hand)

    # 両手同時に指さし姿勢なら曖昧 → 不採用（誤動作回避）
    if l_point and not r_point:
        # 指先の画像 x を左右量に。中心 0.5 からのオフセット。
        tip = left_hand.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        return "L", (tip.x - 0.5)
    if r_point and not l_point:
        tip = right_hand.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        return "R", (tip.x - 0.5)
    return None, 0.0


# =========================
# 操舵（共通・将来差し替え点）
# =========================
def steer_from_lateral(lateral, forward=POINT_BASE_SPEED):
    """left/right の偏差 lateral（正=右にずれている）と前進量 forward から
    左右輪速度を返す。第2段階で lateral を 3D 方位角に、第3段階で forward を
    仰角由来に差し替えても、この関数はそのまま使える。"""
    if abs(lateral) < POINT_LR_DEADBAND:
        lateral = 0.0
    turn = POINT_GAIN * lateral * 1000.0   # 正規化オフセット→速度差スケール
    # lateral 正（指先が右）→ 右へ向く＝右輪を落とす
    cmdVL = int(forward + turn)
    cmdVR = int(forward - turn)
    return cmdVL, cmdVR


# =========================
# 主动作处理
# =========================
def handle_action(results, depth_img, w, h):
    global is_following, lost_frames, last_cmdVL, last_cmdVR
    global invalid_depth_frames
    global is_pointing, point_hold_cnt, point_release_cnt, point_side
    global point_lateral_view 

    def with_src(text, src):
        return f"{text}[{src}]"

    # =========================
    # 人体丢失处理
    # =========================
    if results is None or results.pose_landmarks is None:
        lost_frames += 1
        if lost_frames <= MAX_LOST_FRAMES:
            return last_cmdVL, last_cmdVR, with_src("KEEP LAST (NO POSE)", "NO POSE"), 0.0
        else:
            is_following = False
            return 0, 0, with_src("STOP (NO POSE)", "NO POSE"), 0.0

    lost_frames = 0
    lm = results.pose_landmarks.landmark

    # 基础关键点
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    hip_cx = (l_hip.x + r_hip.x) * 0.5
    hip_cy = (l_hip.y + r_hip.y) * 0.5
    sh_cx  = (l_sh.x + r_sh.x) * 0.5
    sh_cy  = (l_sh.y + r_sh.y) * 0.5

    # =========================
    # 深度取点：髋中心 → 鼻子 → 肩中心（带可见性判断）
    # =========================
    VIS_TH = 0.35

    def ok(pt):
        vis = getattr(pt, "visibility", 1.0)
        return (0.0 <= pt.x <= 1.0) and (0.0 <= pt.y <= 1.0) and (vis >= VIS_TH)

    hip_ok  = ok(l_hip) and ok(r_hip)
    nose_ok = ok(nose)
    sh_ok   = ok(l_sh) and ok(r_sh)

    if hip_ok:
        depth_x, depth_y = hip_cx, hip_cy
        depth_src = "HIP"
    elif nose_ok:
        depth_x, depth_y = nose.x, nose.y
        depth_src = "NOSE"
    elif sh_ok:
        depth_x, depth_y = sh_cx, sh_cy
        depth_src = "SHOULDER"
    else:
        # 三者都不可靠：先用鼻子坐标尝试（即便低可见性），失败再判定为无深度
        depth_x, depth_y = nose.x, nose.y
        depth_src = "FALLBACK"

    distance = depth_at_xy(depth_x, depth_y, depth_img)

    # 如果深度=0，再按同优先级换点尝试一次，避免“洞/无效像素”
    if distance <= 0:
        if depth_src != "HIP" and hip_ok:
            distance = depth_at_xy(hip_cx, hip_cy, depth_img)
            depth_src = "HIP*"
        if distance <= 0 and depth_src != "NOSE" and nose_ok:
            distance = depth_at_xy(nose.x, nose.y, depth_img)
            depth_src = "NOSE*"
        if distance <= 0 and sh_ok:
            distance = depth_at_xy(sh_cx, sh_cy, depth_img)
            depth_src = "SHOULDER*"

    # ===== 深度无效：保持动作（最多 7 帧）=====
    if distance <= 0:
        invalid_depth_frames += 1
        if invalid_depth_frames <= MAX_INVALID_DEPTH:
            return last_cmdVL, last_cmdVR, with_src("KEEP LAST (NO DEPTH)", "NO DEPTH"), 0.0
        else:
            is_following = False
            return 0, 0, with_src("STOP (NO DEPTH)", "NO DEPTH"), 0.0
    else:
        invalid_depth_frames = 0

    # ===== 太近：强制停止 =====
    if distance < TOO_CLOSE_STOP:
        is_following = False
        return 0, 0, with_src("TOO CLOSE", depth_src), distance

    # =========================
    # 手势检测（伸手：人体深度 vs 手腕深度）
    # =========================
    L_forward = is_hand_forward(distance, l_wr, depth_img)
    R_forward = is_hand_forward(distance, r_wr, depth_img)

    L_face = left_hand_facing(results.left_hand_landmarks)
    R_face = right_hand_facing(results.right_hand_landmarks)

    follow_start = (R_forward and R_face == "back") or (L_forward and L_face == "back")
    follow_stop  = (R_forward and R_face == "palm") or (L_forward and L_face == "palm")

    # =========================
    # 指さしモード（第1段階：画像左右のみ）
    # =========================
    # 追従中／伸手中は指さしを評価しない（既存ジェスチャーと排他）。
    point_side_now, point_lateral = (None, 0.0)
    if POINTING_ENABLED and not is_following and not (L_forward or R_forward):
        point_side_now, point_lateral = detect_pointing(
            lm, results.left_hand_landmarks, results.right_hand_landmarks
        )

    if point_side_now is not None:
        point_hold_cnt += 1
        point_release_cnt = 0
        point_side = point_side_now
    else:
        point_release_cnt += 1
        if point_release_cnt >= POINT_RELEASE_FRAMES:
            point_hold_cnt = 0
            if is_pointing:
                point_lateral_view = point_lateral
                # 指さしを下ろした → 即停止（安全要件）
                is_pointing = False
                last_cmdVL, last_cmdVR = 0, 0
                return 0, 0, with_src("POINT RELEASE (STOP)", depth_src), distance

    # 一定時間保持で確定 → 指さし誘導開始
    if point_hold_cnt >= POINT_HOLD_FRAMES:
        is_pointing = True

    # ===== 指さし誘導中 =====
    if is_pointing:
        # 第1段階：指先の画像左右オフセットを lateral に使う
        point_lateral_view = point_lateral          # ← この行を追加
        cmdVL, cmdVR = steer_from_lateral(point_lateral, POINT_BASE_SPEED)
        last_cmdVL, last_cmdVR = cmdVL, cmdVR
        side_txt = "R" if point_side == "R" else "L"
        return cmdVL, cmdVR, with_src(f"POINTING({side_txt})", depth_src), distance

    # ===== 跟随开始（坑B修复：不 return 0，直接进入当帧跟随控制）=====
    if follow_start:    
        is_following = True

    # ===== 只有跟随时才允许停止（避免干扰倒车）=====
    if is_following and follow_stop:
        is_following = False
        return 0, 0, with_src("FOLLOW STOP", depth_src), distance

    # =========== 跟随模式 ===========
    if is_following:
        # 用深度源对应的 x 来做偏差（更一致）
        if depth_src.startswith("HIP"):
            x_for_turn = hip_cx
        elif depth_src.startswith("NOSE") or depth_src == "FALLBACK":
            x_for_turn = nose.x
        else:
            x_for_turn = sh_cx

        dist_px = (x_for_turn - 0.5) * w
        cmdVL = int(3000 + 10 * dist_px)
        cmdVR = int(3000 - 10 * dist_px)
        last_cmdVL, last_cmdVR = cmdVL, cmdVR
        return cmdVL, cmdVR, with_src("FOLLOWING", depth_src), distance

    # =========== 非跟随：倒车 ===========
    both_forward = (L_forward and R_forward)
    if both_forward:
        base = -1500
        dy = (l_wr.y - r_wr.y)
        turn = int(dy * 6000)

        cmdVL = int(base + turn)
        cmdVR = int(base - turn)

        if turn == 0:
            action_text = "BACKWARD"
        else:
            action_text = "BACKWARD LEFT" if turn > 0 else "BACKWARD RIGHT"

        last_cmdVL, last_cmdVR = cmdVL, cmdVR
        return cmdVL, cmdVR, with_src(action_text, depth_src), distance

    # =========== 非跟随：前进/左右 ===========
    L_up = (l_wr.y < l_el.y - ARM_UP_Y)
    R_up = (r_wr.y < r_el.y - ARM_UP_Y)

    if L_up and R_up:
        dy1 = (l_wr.y - r_wr.y)
        cmdVL = int(2500 + dy1 * 6000)
        cmdVR = int(2500 - dy1 * 6000)
        action_text = "GO AHEAD"
    elif R_up:
        cmdVL = 1500
        cmdVR = 3000
        action_text = "LEFT"
    elif L_up:
        cmdVL = 3000
        cmdVR = 1500
        action_text = "RIGHT"
    else:
        cmdVL = 0
        cmdVR = 0
        action_text = "STOP"

    last_cmdVL, last_cmdVR = cmdVL, cmdVR
    return cmdVL, cmdVR, with_src(action_text, depth_src), distance


# =========================
# 主节点 (ROS2 Humble)
# =========================
rclpy.init()
node = Node("pose_tracker")
pub  = node.create_publisher(String, "/car_state", 10)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holo:

  try:
    while rclpy.ok():
        frame_start = time.perf_counter()

        frames = pipeline.wait_for_frames()
        aligned  = align_to_color.process(frames)
        color_fr = aligned.get_color_frame()
        depth_fr = aligned.get_depth_frame()
        if not color_fr or not depth_fr:
            continue

        color_img = np.asanyarray(color_fr.get_data())
        depth_img = np.asanyarray(depth_fr.get_data())
        h, w, _ = color_img.shape

        # =========================
        # 3m 以上 ×2 双三次插值（只用于提升识别）
        # =========================
        z_center = depth_img[h//2, w//2]
        approx_dis = z_center * depth_scale if z_center > 0 else 0.0

        if approx_dis >= 3.0:
            proc_img = cv2.resize(color_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            proc_img = color_img

        rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        results = holo.process(rgb)

        # =========================
        # 同时计算三套坐标（UI观察用）
        # =========================
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE.value]
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            l_sh  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh  = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            hip_coords  = ((l_hip.x + r_hip.x) * 0.5, (l_hip.y + r_hip.y) * 0.5)
            nose_coords = (nose.x, nose.y)
            sh_coords   = ((l_sh.x + r_sh.x) * 0.5, (l_sh.y + r_sh.y) * 0.5)
        else:
            hip_coords = nose_coords = sh_coords = (0.0, 0.0)

        # =========================
        # 动作处理
        # =========================
        cmdVL, cmdVR, action_text, distance = handle_action(results, depth_img, w, h)
        #send_can(cmdVL, cmdVR)

        # =========================
        # 绘制骨架
        # =========================
        vis = proc_img.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                vis, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(vis, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(vis, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        vis = cv2.resize(vis, (640, 480))
        fps_display = 1.0 / (time.perf_counter() - frame_start)

        # =========================
        # UI
        # =========================
        y0, dy = 40, 26
        yellow = (0, 255, 255)

        if is_pointing:
            mode_str = f"Pointing({point_side})"
        elif is_following:
            mode_str = "Following"
        else:
            mode_str = "Idle"
        cv2.putText(vis, f"EVP: {mode_str}",
                    (30, y0 + 0*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2)
        cv2.putText(vis, f"Pose: {action_text}",
                    (30, y0 + 1*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2)
        cv2.putText(vis, f"Dis: {distance:.3f} m",
                    (30, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2)

        cv2.putText(vis, f"FPS: {fps_display:.2f}",
                    (30, y0 + 3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"cmdVL: {cmdVL}",
                    (30, y0 + 4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(vis, f"cmdVR: {cmdVR}",
                    (30, y0 + 5*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 三个坐标同时显示（便于判断丢点）
        cv2.putText(vis, f"HipC : ({hip_coords[0]:.3f},{hip_coords[1]:.3f})",
                    (30, y0 + 6*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, yellow, 2)
        cv2.putText(vis, f"Nose : ({nose_coords[0]:.3f},{nose_coords[1]:.3f})",
                    (30, y0 + 7*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, yellow, 2)
        cv2.putText(vis, f"ShC  : ({sh_coords[0]:.3f},{sh_coords[1]:.3f})",
                    (30, y0 + 8*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, yellow, 2)
        
        # ===== 指さし可視化（追加）=====
        # 画面中央の基準線とデッドバンド帯
        cx = 320  # 640幅の中央
        db_px = int(POINT_LR_DEADBAND * 640)
        cv2.line(vis, (cx, 0), (cx, 480), (200, 200, 200), 1)
        cv2.line(vis, (cx - db_px, 0), (cx - db_px, 480), (120, 120, 120), 1)
        cv2.line(vis, (cx + db_px, 0), (cx + db_px, 480), (120, 120, 120), 1)

        # 指先マーカー（指さし中のみ、対応する手の人差し指TIPに円）
        if is_pointing and results.pose_landmarks:
            hand = (results.right_hand_landmarks if point_side == "R"
                    else results.left_hand_landmarks)
            if hand is not None:
                tip = hand.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
                tx = int(tip.x * 640)
                ty = int(tip.y * 480)
                cv2.circle(vis, (tx, ty), 12, (0, 0, 255), 2)
                cv2.circle(vis, (tx, ty), 3, (0, 0, 255), -1)

        # lateral 数値（右上あたり）
        cv2.putText(vis, f"Point LR: {point_lateral_view:+.3f}",
                    (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        msg = String()
        msg.data = json.dumps({
            "distance": round(distance, 3),
            "approx_dis": round(approx_dis, 3),
            "fps": round(fps_display, 2),
            "cmdVL": cmdVL,
            "cmdVR": cmdVR,
            "act_text": action_text,
            "is_pointing": bool(is_pointing),
            "point_side": point_side if is_pointing else None
        })
        pub.publish(msg)

        out.write(vis)
        cv2.imshow("EVP System", vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        used = time.perf_counter() - frame_start
        if used < frame_dt:
            time.sleep(frame_dt - used)

  except KeyboardInterrupt:
    # Ctrl+C で止めても下の finally が必ず実行される
    print("Interrupted by user (Ctrl+C)")
  finally:
    # 正常終了・Ctrl+C・例外のいずれでも必ず後始末する
    print("Cleaning up: releasing camera, video file, and ROS2 node...")
    try:
        pipeline.stop()
    except Exception:
        pass
    try:
        out.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()
    print("Cleanup done. Video saved safely.")

