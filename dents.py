import cv2
import numpy as np
import random
import requests


# -----------------------------
# Load image from CDN URL
# -----------------------------
def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decoding failed")
    return img


# -----------------------------
# Generate center-positioned bounding box (half image size)
# -----------------------------
def random_bbox(img, min_ratio=0.45, max_ratio=0.55):
    """Create dent region around image center (45-55% of image = half size)"""
    h, w = img.shape[:2]
    
    # Dent size: 45-55% of image dimensions (approximately half)
    dent_w = int(random.uniform(min_ratio, max_ratio) * w)
    dent_h = int(random.uniform(min_ratio, max_ratio) * h)
    
    # Center the dent position
    cx = w // 2
    cy = h // 2
    
    # Offset center slightly for variation
    cx += random.randint(-int(w * 0.05), int(w * 0.05))
    cy += random.randint(-int(h * 0.05), int(h * 0.05))
    
    # Calculate top-left corner
    x = max(0, min(cx - dent_w // 2, w - dent_w))
    y = max(0, min(cy - dent_h // 2, h - dent_h))
    
    return x, y, dent_w, dent_h


# -----------------------------
# Post-effect helper
# -----------------------------
def apply_post_effect_to_roi(roi, effect=None, intensity=1.0):
    """Apply a post-effect to an ROI.

    effect: None | 'blacken' | 'grayscale'
    intensity: blending amount between original and effect (0.0-1.0)
    """
    if effect is None:
        return roi

    intensity = float(np.clip(intensity, 0.0, 1.0))

    if effect == "blacken":
        black = np.zeros_like(roi)
        return cv2.addWeighted(roi, 1.0 - intensity, black, intensity, 0)

    if effect == "grayscale":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(roi, 1.0 - intensity, gray_bgr, intensity, 0)

    return roi


# -----------------------------
# TECHNIQUE 1: Radial Displacement Dent (Your Original)
# -----------------------------
def apply_radial_dent(img, bbox, strength=600, post_effect=None, effect_intensity=1.0):
    """Creates a smooth radial indentation with shading - VERY STRONG

    Optional post-effect: `post_effect` can be 'blacken' or 'grayscale'.
    `effect_intensity` controls blending (0.0-1.0).
    """
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w].copy()

    rh, rw = roi.shape[:2]
    # Place dent at center of the region
    cx = rw // 2
    cy = rh // 2
    # Large radius covering most of the dent region
    radius = min(rw, rh) * 0.4

    yy, xx = np.indices((rh, rw), dtype=np.float32)
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)

    mask = dist < radius
    displacement = strength * (1 - dist / radius)
    displacement[~mask] = 0

    dxn = dx / (dist + 1e-6)
    dyn = dy / (dist + 1e-6)

    map_x = (xx + dxn * displacement).astype(np.float32)
    map_y = (yy + dyn * displacement).astype(np.float32)

    # Ensure maps are within bounds
    map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

    dent = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    # Shading for depth - VERY STRONG
    shade = cv2.GaussianBlur(displacement / (strength + 1e-6), (21, 21), 0)
    dent = dent.astype(np.float32)
    dent -= shade[..., None] * 120
    dent += (1 - shade[..., None]) * 30
    dent = np.clip(dent, 0, 255).astype(np.uint8)

    # Apply optional post-effect
    if post_effect in ("blacken", "grayscale"):
        dent = apply_post_effect_to_roi(dent, post_effect, effect_intensity)

    img[y:y + h, x:x + w] = dent
    return img


# -----------------------------
# TECHNIQUE 2: Creased/Linear Dent
# -----------------------------
def apply_creased_dent(img, bbox, strength=600, post_effect=None, effect_intensity=1.0):
    """Creates a linear crease-like dent (like a fold) - VERY STRONG

    Optional post-effect: `post_effect` can be 'blacken' or 'grayscale'.
    `effect_intensity` controls blending (0.0-1.0).
    """
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w].copy()

    rh, rw = roi.shape[:2]

    # Create a linear gradient through center
    angle = random.uniform(0, np.pi)
    cx, cy = rw // 2, rh // 2

    yy, xx = np.indices((rh, rw), dtype=np.float32)

    # Rotate coordinates
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rx = (xx - cx) * cos_a + (yy - cy) * sin_a
    ry = -(xx - cx) * sin_a + (yy - cy) * sin_a

    # Create crease pattern
    width = min(rw, rh) // 3
    dist_from_line = np.abs(ry)

    mask = dist_from_line < width
    displacement = strength * (1 - dist_from_line / (width + 1e-6)) ** 2
    displacement[~mask] = 0

    # Apply displacement perpendicular to crease
    map_x = (xx - sin_a * displacement).astype(np.float32)
    map_y = (yy + cos_a * displacement).astype(np.float32)

    # Ensure maps are within bounds
    map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

    dent = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    # Add shadow along the crease - VERY STRONG
    shade = cv2.GaussianBlur(displacement / (strength + 1e-6), (15, 15), 0)
    dent = dent.astype(np.float32)
    dent -= shade[..., None] * 150
    dent = np.clip(dent, 0, 255).astype(np.uint8)

    # Apply optional post-effect
    if post_effect in ("blacken", "grayscale"):
        dent = apply_post_effect_to_roi(dent, post_effect, effect_intensity)

    img[y:y + h, x:x + w] = dent
    return img


# -----------------------------
# TECHNIQUE 3: Bumpy/Hail Damage Dent
# -----------------------------
def apply_hail_damage(img, bbox, num_dents=150, post_effect=None, effect_intensity=1.0):
    """Creates multiple dense circular dents around center - VERY STRONG

    Optional post-effect: `post_effect` can be 'blacken' or 'grayscale'.
    `effect_intensity` controls blending (0.0-1.0).
    """
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w].copy()
    rh, rw = roi.shape[:2]
    
    # Center region for dent placement
    cx_center, cy_center = rw // 2, rh // 2
    region_size = min(rw, rh) // 3

    for _ in range(num_dents):
        # Place dents around center
        cx = random.randint(max(0, cx_center - region_size), min(rw, cx_center + region_size))
        cy = random.randint(max(0, cy_center - region_size), min(rh, cy_center + region_size))
        radius = random.randint(min(rw, rh) // 15, min(rw, rh) // 6)
        strength = random.uniform(25, 40)

        yy, xx = np.indices((rh, rw), dtype=np.float32)
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx ** 2 + dy ** 2)

        mask = dist < radius
        displacement = strength * (1 - dist / (radius + 1e-6)) ** 2
        displacement[~mask] = 0

        dxn = dx / (dist + 1e-6)
        dyn = dy / (dist + 1e-6)

        map_x = (xx + dxn * displacement).astype(np.float32)
        map_y = (yy + dyn * displacement).astype(np.float32)

        # Ensure maps are within bounds
        map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

        roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

        # Add VERY STRONG shading
        shade = cv2.GaussianBlur(displacement / (strength + 1e-6), (11, 11), 0)
        roi = roi.astype(np.float32)
        roi -= shade[..., None] * 90
        roi = np.clip(roi, 0, 255).astype(np.uint8)

    # Apply optional post-effect
    if post_effect in ("blacken", "grayscale"):
        roi = apply_post_effect_to_roi(roi, post_effect, effect_intensity)

    img[y:y + h, x:x + w] = roi
    return img


# -----------------------------
# TECHNIQUE 4: Wave/Ripple Distortion
# -----------------------------
def apply_wave_dent(img, bbox, strength=80, post_effect=None, effect_intensity=1.0):
    """Creates wavy distortion (like metal buckling) - VERY STRONG

    Optional post-effect: `post_effect` can be 'blacken' or 'grayscale'.
    `effect_intensity` controls blending (0.0-1.0).
    """
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w].copy()

    rh, rw = roi.shape[:2]

    yy, xx = np.indices((rh, rw), dtype=np.float32)

    # Create wave pattern
    wave_length = max(min(rw, rh) // 4, 1)
    amplitude = strength

    # Combine horizontal and vertical waves
    offset_x = amplitude * np.sin(2 * np.pi * yy / wave_length)
    offset_y = amplitude * np.sin(2 * np.pi * xx / wave_length)

    # Create envelope to limit effect to center
    cx, cy = rw // 2, rh // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = min(rw, rh) // 2
    envelope = np.clip(1 - dist / (max_dist + 1e-6), 0, 1)

    map_x = (xx + offset_x * envelope).astype(np.float32)
    map_y = (yy + offset_y * envelope).astype(np.float32)

    # Ensure maps are within bounds
    map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

    dent = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    # Add VERY STRONG shading
    shade = cv2.GaussianBlur(envelope, (21, 21), 0) * 0.5
    dent = dent.astype(np.float32)
    dent -= shade[..., None] * 80
    dent = np.clip(dent, 0, 255).astype(np.uint8)

    # Apply optional post-effect
    if post_effect in ("blacken", "grayscale"):
        dent = apply_post_effect_to_roi(dent, post_effect, effect_intensity)

    img[y:y + h, x:x + w] = dent
    return img


# -----------------------------
# TECHNIQUE 5: Depth Map Based Dent
# -----------------------------
def apply_depth_dent(img, bbox, strength=100, post_effect=None, effect_intensity=1.0):
    """Uses a depth map for more realistic lighting - VERY STRONG, centered

    Optional post-effect: `post_effect` can be 'blacken' or 'grayscale'.
    `effect_intensity` controls blending (0.0-1.0).
    """
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w].copy()

    rh, rw = roi.shape[:2]
    # Center position
    cx = rw // 2
    cy = rh // 2

    yy, xx = np.indices((rh, rw), dtype=np.float32)

    # Create elliptical dent
    a = random.randint(rw // 4, rw // 2)  # horizontal radius
    b = random.randint(rh // 4, rh // 2)  # vertical radius

    dist = np.sqrt(((xx - cx) / (a + 1e-6)) ** 2 + ((yy - cy) / (b + 1e-6)) ** 2)

    # Create smooth depth map
    depth = np.zeros((rh, rw), dtype=np.float32)
    mask = dist < 1
    depth[mask] = (1 - dist[mask]) ** 2

    depth = cv2.GaussianBlur(depth, (21, 21), 0)

    # Apply displacement
    dx = xx - cx
    dy = yy - cy
    dist_from_center = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

    map_x = (xx + (dx / dist_from_center) * depth * strength).astype(np.float32)
    map_y = (yy + (dy / dist_from_center) * depth * strength).astype(np.float32)

    # Ensure maps are within bounds
    map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

    dent = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    # Advanced shading based on depth gradient - VERY STRONG
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

    # Simulate light from top-left
    light_dir = np.array([-1, -1])
    shading = -(grad_x * light_dir[0] + grad_y * light_dir[1])
    shading = cv2.GaussianBlur(shading, (11, 11), 0)

    dent = dent.astype(np.float32)
    dent += shading[..., None] * 200
    dent = np.clip(dent, 0, 255).astype(np.uint8)

    # Apply optional post-effect
    if post_effect in ("blacken", "grayscale"):
        dent = apply_post_effect_to_roi(dent, post_effect, effect_intensity)

    img[y:y + h, x:x + w] = dent
    return img


# -----------------------------
# TECHNIQUE 6: Scratch + Dent Combination
# -----------------------------
def apply_scratch_dent(img, bbox, strength=80, post_effect=None, effect_intensity=1.0):
    """Combines scratches with a dent - VERY STRONG, centered

    Optional post-effect: `post_effect` can be 'blacken' or 'grayscale'.
    `effect_intensity` controls blending (0.0-1.0).
    """
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w].copy()

    rh, rw = roi.shape[:2]

    # First apply a basic dent at center
    cx = rw // 2
    cy = rh // 2
    radius = min(rw, rh) * 0.35

    yy, xx = np.indices((rh, rw), dtype=np.float32)
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)

    mask = dist < radius
    displacement = strength * (1 - dist / (radius + 1e-6)) ** 2
    displacement[~mask] = 0

    dxn = dx / (dist + 1e-6)
    dyn = dy / (dist + 1e-6)

    map_x = (xx + dxn * displacement).astype(np.float32)
    map_y = (yy + dyn * displacement).astype(np.float32)

    # Ensure maps are within bounds
    map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

    dent = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    # Add scratches
    num_scratches = random.randint(2, 4)
    for _ in range(num_scratches):
        # Random scratch parameters
        x1 = random.randint(0, rw - 1)
        y1 = random.randint(0, rh - 1)
        length = random.randint(rw // 6, rw // 3)
        angle = random.uniform(0, 2 * np.pi)

        x2 = int(np.clip(x1 + length * np.cos(angle), 0, rw - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, rh - 1))

        # Draw scratch
        thickness = random.randint(1, 2)
        color = tuple([int(c * 0.3) for c in dent[y1, x1]])  # Darker version
        cv2.line(dent, (x1, y1), (x2, y2), color, thickness)

        # Add slight highlight next to scratch
        offset_x = int(-np.sin(angle))
        offset_y = int(np.cos(angle))
        highlight_x1 = int(np.clip(x1 + offset_x, 0, rw - 1))
        highlight_y1 = int(np.clip(y1 + offset_y, 0, rh - 1))
        highlight_x2 = int(np.clip(x2 + offset_x, 0, rw - 1))
        highlight_y2 = int(np.clip(y2 + offset_y, 0, rh - 1))
        cv2.line(dent, (highlight_x1, highlight_y1),
                 (highlight_x2, highlight_y2), (255, 255, 255), 1, cv2.LINE_AA)

    # Overall shading - VERY STRONG
    shade = cv2.GaussianBlur(displacement / (strength + 1e-6), (21, 21), 0)
    dent = dent.astype(np.float32)
    dent -= shade[..., None] * 140
    dent = np.clip(dent, 0, 255).astype(np.uint8)

    # Apply optional post-effect
    if post_effect in ("blacken", "grayscale"):
        dent = apply_post_effect_to_roi(dent, post_effect, effect_intensity)

    img[y:y + h, x:x + w] = dent
    return img


# -----------------------------
# MAIN - Demonstrate all techniques
# -----------------------------
if __name__ == "__main__":

    CDN_IMAGE_URL = "https://www.motoringresearch.com/wp-content/uploads/2017/09/01_ordinary_extraordinary-1.jpg"

    techniques = [
        ("1. Radial Dent", apply_radial_dent),
        ("2. Creased Dent", apply_creased_dent),
        ("3. Hail Damage", apply_hail_damage),
        ("4. Wave/Ripple", apply_wave_dent),
        ("5. Depth Map Dent", apply_depth_dent),
        ("6. Scratch + Dent", apply_scratch_dent),
    ]

    for name, technique_func in techniques:
        print(f"Applying: {name}")

        img = load_image_from_url(CDN_IMAGE_URL)
        bbox = random_bbox(img)

        # Apply the technique
        if technique_func == apply_hail_damage:
            img = technique_func(img, bbox, num_dents=5)
        else:
            img = technique_func(img, bbox)

        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text label
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow(name, img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()