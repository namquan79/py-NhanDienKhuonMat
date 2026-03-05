import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
#Vietnamese
VI_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
def put_vietnamese_text(img_bgr, text, org, font_size=28, color=(255, 255, 255), thickness=2):
    """
    Vẽ text tiếng Việt lên ảnh OpenCV bằng PIL.
    img_bgr: ảnh OpenCV (BGR)
    org: (x, y) góc trên-trái của text
    color: BGR
    """
    x, y = org
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(VI_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    rgb = (color[2], color[1], color[0])

    # viền chữ
    if thickness > 0:
        outline_rgb = (0, 0, 0)
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_rgb)

    draw.text((x, y), text, font=font, fill=rgb)

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out
#endVietnamese

def draw_progress_circle(frame, progress01: float, center=None, radius=45, thickness=10):
    """
    Vẽ vòng tròn tiến độ + % ở giữa. (return frame)
    progress01: 0..1
    """
    h, w = frame.shape[:2]
    if center is None:
        center = (w - 70, 70)  # góc phải trên

    p = float(np.clip(progress01, 0.0, 1.0))
    pct = int(round(p * 100))

    # vòng nền
    cv2.circle(frame, center, radius, (70, 70, 70), thickness)

    # vòng tiến độ
    angle = int(360 * p)
    cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 255, 255), thickness)

    # text % (PIL)
    frame = put_vietnamese_text(
        frame, f"{pct}%", (center[0] - 22, center[1] - 18),
        font_size=26, color=(255, 255, 255), thickness=2
    )
    return frame


