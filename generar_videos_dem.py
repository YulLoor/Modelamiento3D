r"""Genera hillshade, curvas 3D y videos orbitales desde cuatro DEM GeoTIFF.

Uso (PowerShell, desde la carpeta del proyecto):
    .\.venv-volcanes3d\Scripts\python.exe .\4-Scripts\generar_videos_dem.py --todos
    .\.venv-volcanes3d\Scripts\python.exe .\4-Scripts\generar_videos_dem.py --volcan Cerro_Tusa
    .\.venv-volcanes3d\Scripts\python.exe .\4-Scripts\generar_videos_dem.py --preview

Requiere numpy, rasterio, pyvista, vtk, matplotlib, pillow, imageio e imageio-ffmpeg.
Los GeoTIFF de entrada tienen celdas cero de borde, que aquí se tratan como NoData.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyvista as pv
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import xy


PROJECT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT / "1-Insumos"
RASTER_DIR = PROJECT / "2-Rasters"
VECTOR_DIR = PROJECT / "3-Vectores"
OUTPUT_DIR = PROJECT / "5-Salidas" / "Video_3D"
NODATA_VALUE = 0.0
CONTOUR_INTERVAL_M = 30.0
HILLSHADE_AZIMUTH_DEG = 315.0
HILLSHADE_ALTITUDE_DEG = 45.0
HILLSHADE_STRENGTH = 0.65
CAMERA_VIEW_ANGLE_DEG = 35.0
VERTICAL_EXAGGERATION = 1.0

VOLCANOES = {
    "Cerro_Tusa": "DEM_Cerro_Tusa.tif",
    "Cerro_Bravo": "DEM_Cerro_Bravo.tif",
    "Cerro_Sillon": "DEM_Cerro_Sillon_2.tif",
    "Morro_Alegre": "DEM_Morro_Alegre_1.tif",
}


def valid_dem(array: np.ndarray) -> np.ndarray:
    """Máscara de elevaciones reales; en estos cuatro recortes, cero es NoData."""
    return np.isfinite(array) & (array != NODATA_VALUE)


def calculate_hillshade(elevation: np.ndarray, valid: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Sombreado Horn de 8 bits, calculado en metros desde los propios valores DEM."""
    z = np.asarray(elevation, dtype=np.float64)
    z_safe = np.where(valid, z, 0.0)
    z_padded = np.pad(z_safe, 1, mode="edge")

    z1 = z_padded[:-2, :-2]
    z2 = z_padded[:-2, 1:-1]
    z3 = z_padded[:-2, 2:]
    z4 = z_padded[1:-1, :-2]
    z6 = z_padded[1:-1, 2:]
    z7 = z_padded[2:, :-2]
    z8 = z_padded[2:, 1:-1]
    z9 = z_padded[2:, 2:]

    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * abs(cell_x))
    dz_dy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8 * abs(cell_y))

    normal_length = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
    nx = -dz_dx / normal_length
    ny = -dz_dy / normal_length
    nz = 1.0 / normal_length

    azimuth = math.radians(HILLSHADE_AZIMUTH_DEG)
    altitude = math.radians(HILLSHADE_ALTITUDE_DEG)
    lx = math.cos(altitude) * math.sin(azimuth)
    ly = math.cos(altitude) * math.cos(azimuth)
    lz = math.sin(altitude)
    intensity = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

    # Ignorar una celda a cada lado del borde evita usar vecinos que sean NoData.
    safe_neighborhood = np.ones(valid.shape, dtype=bool)
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            shifted = np.roll(valid, (row_offset, col_offset), axis=(0, 1))
            if row_offset < 0:
                shifted[row_offset:, :] = False
            elif row_offset > 0:
                shifted[:row_offset, :] = False
            if col_offset < 0:
                shifted[:, col_offset:] = False
            elif col_offset > 0:
                shifted[:, :col_offset] = False
            safe_neighborhood &= shifted

    result = np.rint(intensity * 255.0).astype(np.uint8)
    result[~(valid & safe_neighborhood)] = 0
    return result


def save_hillshade(path: Path, profile: dict, hillshade: np.ndarray) -> None:
    profile_out = profile.copy()
    profile_out.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
    )
    with rasterio.open(path, "w", **profile_out) as destination:
        destination.write(hillshade, 1)


def wkb_multilinestring_z(lines: list[list[tuple[float, float, float]]]) -> bytes:
    """Codifica un MultiLineString Z en WKB little-endian ISO."""
    parts = [b"\x01", struct.pack("<I", 1005), struct.pack("<I", len(lines))]
    for line in lines:
        parts.extend((b"\x01", struct.pack("<II", 1002, len(line))))
        for x_coord, y_coord, z_coord in line:
            parts.append(struct.pack("<ddd", x_coord, y_coord, z_coord))
    return b"".join(parts)


def create_contour_geopackage(
    path: Path,
    crs_wkt: str,
    contour_features: dict[int, list[list[tuple[float, float, float]]]],
) -> int:
    """Escribe las isolíneas 3D en GeoPackage usando SQLite y WKB estándar."""
    if path.exists():
        path.unlink()

    all_points = [point for lines in contour_features.values() for line in lines for point in line]
    if all_points:
        min_x = min(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_x = max(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)
    else:
        min_x = min_y = max_x = max_y = None

    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA application_id = 1196444487;
            PRAGMA user_version = 10300;
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            );
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL,
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER,
                FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            );
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                PRIMARY KEY (table_name, column_name),
                FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            );
            CREATE TABLE curvas_30m (
                fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                geom MULTILINESTRING,
                cota_m REAL NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("Undefined Cartesian SRS", -1, "NONE", -1, "undefined", "undefined"),
                ("Undefined Geographic SRS", 0, "NONE", 0, "undefined", "undefined"),
                ("MAGNA-SIRGAS 2018 / Origen-Nacional", 9377, "EPSG", 9377, crs_wkt, ""),
            ],
        )
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        connection.execute(
            "INSERT INTO gpkg_contents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("curvas_30m", "features", "Curvas de nivel cada 30 m", "Cotas absolutas en metros", timestamp,
             min_x, min_y, max_x, max_y, 9377),
        )
        connection.execute(
            "INSERT INTO gpkg_geometry_columns VALUES (?, ?, ?, ?, ?, ?)",
            ("curvas_30m", "geom", "MULTILINESTRING", 9377, 1, 0),
        )

        inserted = 0
        for contour_level, lines in sorted(contour_features.items()):
            if not lines:
                continue
            geometry = b"GP" + bytes((0, 1)) + struct.pack("<i", 9377) + wkb_multilinestring_z(lines)
            connection.execute(
                "INSERT INTO curvas_30m (geom, cota_m) VALUES (?, ?)",
                (sqlite3.Binary(geometry), float(contour_level)),
            )
            inserted += 1
        connection.commit()
        return inserted
    finally:
        connection.close()


def build_surface(
    elevation: np.ndarray,
    valid: np.ndarray,
    transform,
    shade: np.ndarray,
    global_min: float,
    global_max: float,
) -> tuple[pv.PolyData, float, float, float, float, float, float]:
    """Construye una malla 5 m en coordenadas locales, sin caras sobre NoData."""
    rows, cols = np.indices(elevation.shape)
    world_x, world_y = xy(transform, rows, cols, offset="center")
    world_x = np.asarray(world_x, dtype=np.float64).reshape(elevation.shape)
    world_y = np.asarray(world_y, dtype=np.float64).reshape(elevation.shape)
    valid_rows, valid_cols = np.nonzero(valid)

    center_x = float((world_x[valid].min() + world_x[valid].max()) / 2.0)
    center_y = float((world_y[valid].min() + world_y[valid].max()) / 2.0)
    base_elevation = float(np.min(elevation[valid]))

    vertex_id = np.full(elevation.shape, -1, dtype=np.int64)
    vertex_id[valid] = np.arange(valid_rows.size, dtype=np.int64)
    points = np.column_stack(
        (
            world_x[valid] - center_x,
            world_y[valid] - center_y,
            (elevation[valid] - base_elevation) * VERTICAL_EXAGGERATION,
        )
    )

    a = valid[:-1, :-1]
    b = valid[:-1, 1:]
    c = valid[1:, 1:]
    d = valid[1:, :-1]
    use_first = a & b & c
    use_second = a & c & d
    triangle_rows, triangle_cols = np.nonzero(use_first)
    first = np.column_stack(
        (
            vertex_id[triangle_rows, triangle_cols],
            vertex_id[triangle_rows, triangle_cols + 1],
            vertex_id[triangle_rows + 1, triangle_cols + 1],
        )
    )
    triangle_rows, triangle_cols = np.nonzero(use_second)
    second = np.column_stack(
        (
            vertex_id[triangle_rows, triangle_cols],
            vertex_id[triangle_rows + 1, triangle_cols + 1],
            vertex_id[triangle_rows + 1, triangle_cols],
        )
    )
    triangles = np.vstack((first, second))
    faces = np.column_stack((np.full(triangles.shape[0], 3), triangles)).astype(np.int64).ravel()

    terrain = pv.PolyData(points, faces)
    terrain.point_data["elevation_m"] = elevation[valid].astype(np.float32)
    terrain.field_data["center_easting_m"] = np.array([center_x], dtype=np.float64)
    terrain.field_data["center_northing_m"] = np.array([center_y], dtype=np.float64)
    terrain.field_data["base_elevation_m"] = np.array([base_elevation], dtype=np.float64)
    terrain.field_data["horizontal_crs_epsg"] = np.array([9377], dtype=np.int32)
    terrain.field_data["vertical_exaggeration"] = np.array([VERTICAL_EXAGGERATION], dtype=np.float64)

    span = max(global_max - global_min, 1.0)
    normalized_elevation = np.clip((elevation[valid] - global_min) / span, 0.0, 1.0)
    terrain_colors = matplotlib.colormaps["terrain"](normalized_elevation)[:, :3]
    local_shading = 1.0 - HILLSHADE_STRENGTH + HILLSHADE_STRENGTH * (shade[valid].astype(np.float32) / 255.0)
    terrain.point_data["RGB"] = np.rint(terrain_colors * local_shading[:, None] * 255).clip(0, 255).astype(np.uint8)

    half_width = float((world_x[valid].max() - world_x[valid].min()) / 2.0)
    half_height = float((world_y[valid].max() - world_y[valid].min()) / 2.0)
    relief = float((np.max(elevation[valid]) - base_elevation) * VERTICAL_EXAGGERATION)
    return terrain, center_x, center_y, base_elevation, max(half_width, 1.0), max(half_height, 1.0), relief


def extract_contour_features(
    terrain: pv.PolyData,
    center_x: float,
    center_y: float,
    base_elevation: float,
    levels: np.ndarray,
) -> tuple[pv.PolyData | None, dict[int, list[list[tuple[float, float, float]]]]]:
    """Extrae contornos de la malla y agrupa líneas para GeoPackage y render."""
    if levels.size == 0:
        return None, {}

    contour_mesh = terrain.contour(isosurfaces=levels.tolist(), scalars="elevation_m")
    if contour_mesh.n_points == 0 or not hasattr(contour_mesh, "lines"):
        return None, {}

    features: dict[int, list[list[tuple[float, float, float]]]] = {int(level): [] for level in levels}
    line_values = np.asarray(contour_mesh.lines, dtype=np.int64)
    offset = 0
    # PolyData.lines es una secuencia: [n, id_0, ..., n, id_0, ...].
    while offset < line_values.size:
        count = int(line_values[offset])
        point_ids = line_values[offset + 1 : offset + 1 + count]
        offset += count + 1
        if point_ids.size < 2:
            continue
        scalar_values = np.asarray(contour_mesh.point_data["elevation_m"])[point_ids]
        contour_level = int(round(float(np.mean(scalar_values)) / CONTOUR_INTERVAL_M) * CONTOUR_INTERVAL_M)
        if contour_level not in features:
            contour_level = int(levels[np.argmin(np.abs(levels - np.mean(scalar_values)))])
        local_points = np.asarray(contour_mesh.points)[point_ids]
        absolute_line = [
            (float(p[0] + center_x), float(p[1] + center_y), float(p[2] / VERTICAL_EXAGGERATION + base_elevation))
            for p in local_points
        ]
        features[contour_level].append(absolute_line)
    return contour_mesh, features


def ease(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def camera_state(progress: float) -> tuple[float, float]:
    """Ángulo de rumbo y elevación: una órbita, cenital norte-arriba y regreso."""
    if progress <= 0.66:
        return 360.0 * ease(progress / 0.66), 35.0
    if progress <= 0.83:
        u = ease((progress - 0.66) / 0.17)
        return 360.0 - 90.0 * u, 35.0 + 55.0 * u
    if progress <= 0.86:
        return 270.0, 90.0
    u = ease((progress - 0.86) / 0.14)
    return 270.0 + 90.0 * u, 90.0 - 55.0 * u


def load_overlay_fonts(width: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """Carga fuentes escaladas al ancho del fotograma."""
    size = max(11, min(19, int(width * 0.016)))
    try:
        return ImageFont.truetype("arial.ttf", size), ImageFont.truetype("arial.ttf", size + 2)
    except OSError:
        return ImageFont.load_default(size=size), ImageFont.load_default(size=size + 2)


def annotate_frame(
    frame: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    color_min: float,
    color_max: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    compass_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> np.ndarray:
    """Añade brújula norte y barra de elevación según la orientación de cámara."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Proyección del vector norte sobre los ejes derecho y arriba de la cámara.
    azimuth = math.radians(azimuth_deg)
    elevation_angle = math.radians(elevation_deg)
    cosine = math.cos(elevation_angle)
    forward = np.array((-cosine * math.cos(azimuth), -cosine * math.sin(azimuth), -math.sin(elevation_angle)))
    view_up = np.array((-math.sin(elevation_angle) * math.cos(azimuth), -math.sin(elevation_angle) * math.sin(azimuth), cosine))
    right = np.cross(forward, view_up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    north_screen = np.array((right[1], -view_up[1]), dtype=np.float64)
    north_screen /= max(float(np.linalg.norm(north_screen)), 1e-9)

    compass_x = width - max(66, int(width * 0.08))
    compass_y = max(74, int(height * 0.075))
    compass_radius = max(27, int(width * 0.043))
    arrow_length = compass_radius * 1.15
    start = (compass_x - north_screen[0] * arrow_length / 2, compass_y - north_screen[1] * arrow_length / 2)
    tip = (compass_x + north_screen[0] * arrow_length / 2, compass_y + north_screen[1] * arrow_length / 2)
    perpendicular = np.array((-north_screen[1], north_screen[0]))
    head_base = np.array(tip) - north_screen * compass_radius * 0.42
    head_width = compass_radius * 0.30
    triangle = [
        tip,
        tuple(head_base + perpendicular * head_width),
        tuple(head_base - perpendicular * head_width),
    ]
    dark = (42, 49, 45)
    draw.ellipse(
        (compass_x - compass_radius, compass_y - compass_radius, compass_x + compass_radius, compass_y + compass_radius),
        fill=(250, 250, 246),
        outline=(105, 111, 105),
        width=max(1, width // 540),
    )
    draw.line((start, tip), fill=dark, width=max(3, width // 180))
    draw.polygon(triangle, fill=dark)
    draw.text((compass_x, compass_y + compass_radius + 2), "N", font=compass_font, fill=dark, anchor="mt")

    # Barra vertical con la misma paleta y el rango común usado para colorear los DEM.
    panel_width = max(100, int(width * 0.122))
    panel_right = width - max(10, int(width * 0.012))
    panel_left = panel_right - panel_width
    bar_x = panel_left + max(8, int(width * 0.009))
    bar_y = int(height * 0.515)
    bar_width = max(13, int(width * 0.018))
    bar_height = max(150, int(height * 0.19))
    title_height = font.size + 11
    panel_top = bar_y - title_height
    panel_bottom = bar_y + bar_height + font.size // 2 + 7
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=max(5, int(width * 0.008)),
        fill=(250, 250, 246),
        outline=(155, 159, 151),
        width=max(1, width // 800),
    )
    draw.text(((panel_left + panel_right) // 2, panel_top + 5), "Elevación (m)", font=font, fill=dark, anchor="mt")

    stops = np.linspace(1.0, 0.0, bar_height)
    gradient = np.rint(matplotlib.colormaps["terrain"](stops)[:, :3] * 255).astype(np.uint8)
    gradient = np.repeat(gradient[:, None, :], bar_width, axis=1)
    gradient_image = Image.fromarray(gradient, mode="RGB")
    image.paste(gradient_image, (bar_x, bar_y))
    draw.rectangle((bar_x - 1, bar_y - 1, bar_x + bar_width, bar_y + bar_height), outline=dark, width=1)

    label_x = bar_x + bar_width + max(5, int(width * 0.006))
    span = max(color_max - color_min, 1.0)
    for fraction in np.linspace(1.0, 0.0, 5):
        y = int(bar_y + (1.0 - fraction) * (bar_height - 1))
        value = color_min + fraction * span
        draw.line((bar_x - 3, y, bar_x + bar_width + 3, y), fill=dark, width=1)
        draw.text((label_x, y), f"{value:,.0f}", font=font, fill=dark, anchor="lm")

    return np.asarray(image)


def render_movie(
    terrain: pv.PolyData,
    contour_mesh: pv.PolyData | None,
    name: str,
    output_path: Path,
    center_x: float,
    center_y: float,
    half_width: float,
    half_height: float,
    relief: float,
    min_elevation: float,
    max_elevation: float,
    color_min: float,
    color_max: float,
    duration_seconds: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Renderiza una vuelta completa y la vista cenital en MP4/H.264."""
    plotter = pv.Plotter(off_screen=True, window_size=(width, height), lighting=None)
    plotter.set_background("#f7f7f3")
    plotter.add_mesh(
        terrain,
        scalars="RGB",
        rgb=True,
        lighting=False,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    if contour_mesh is not None and contour_mesh.n_points:
        contour_render = contour_mesh.copy(deep=True)
        contour_render.points[:, 2] += 0.65
        plotter.add_mesh(
            contour_render,
            color="#342e25",
            line_width=1.5,
            render_lines_as_tubes=True,
            lighting=False,
        )

    title_size = max(16, min(28, int(width * 0.026)))
    caption_size = max(14, min(20, int(width * 0.018)))
    plotter.add_text(name.replace("_", " "), position="upper_edge", font_size=title_size, color="#20241f")
    plotter.add_text(
        f"Elevación {min_elevation:,.0f}–{max_elevation:,.0f} m\nCurvas: 30 m\nEscala vertical: 1×",
        position=(24, max(60, int(height * 0.055))),
        font_size=caption_size,
        color="#30342f",
    )
    focal = (0.0, 0.0, relief * 0.48)
    horizontal_radius = math.hypot(half_width, half_height)
    horizontal_half_fov = math.radians(CAMERA_VIEW_ANGLE_DEG * (width / height)) / 2.0
    camera_distance = max(
        horizontal_radius / max(math.tan(horizontal_half_fov), 0.02) * 1.10,
        relief * 1.45,
        500.0,
    )
    frame_count = duration_seconds * fps
    overlay_font, compass_font = load_overlay_fonts(width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            str(output_path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            quality=8,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
        ) as writer:
            for frame in range(frame_count):
                progress = frame / max(frame_count - 1, 1)
                azimuth_deg, elevation_deg = camera_state(progress)
                azimuth = math.radians(azimuth_deg)
                elevation_angle = math.radians(elevation_deg)
                cos_elevation = math.cos(elevation_angle)
                eye = (
                    focal[0] + camera_distance * cos_elevation * math.cos(azimuth),
                    focal[1] + camera_distance * cos_elevation * math.sin(azimuth),
                    focal[2] + camera_distance * math.sin(elevation_angle),
                )
                view_up = (
                    -math.sin(elevation_angle) * math.cos(azimuth),
                    -math.sin(elevation_angle) * math.sin(azimuth),
                    math.cos(elevation_angle),
                )
                plotter.camera_position = [eye, focal, view_up]
                plotter.render()
                rendered_frame = plotter.screenshot(return_img=True, transparent_background=False)
                annotated_frame = annotate_frame(
                    rendered_frame,
                    azimuth_deg,
                    elevation_deg,
                    color_min,
                    color_max,
                    overlay_font,
                    compass_font,
                )
                writer.append_data(annotated_frame)
                if (frame + 1) % max(fps * 5, 1) == 0 or frame + 1 == frame_count:
                    print(f"  {name}: fotograma {frame + 1}/{frame_count}", flush=True)
    finally:
        plotter.close()


def process_volcano(
    volcano: str,
    filename: str,
    global_min: float,
    global_max: float,
    duration: int,
    fps: int,
    width: int,
    height: int,
    preview: bool,
) -> None:
    source_path = INPUT_DIR / filename
    if not source_path.exists():
        raise FileNotFoundError(f"No se encontró el DEM: {source_path}")

    print(f"[{volcano}] Leyendo {source_path.name}", flush=True)
    with rasterio.open(source_path) as source:
        elevation = source.read(1).astype(np.float32, copy=False)
        profile = source.profile.copy()
        transform = source.transform
        crs = source.crs

    valid = valid_dem(elevation)
    if not np.any(valid):
        raise ValueError(f"El DEM {source_path.name} no tiene celdas de elevación válidas.")
    minimum = float(np.min(elevation[valid]))
    maximum = float(np.max(elevation[valid]))
    shade = calculate_hillshade(elevation, valid, profile["transform"].a, profile["transform"].e)

    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = volcano.lower()
    hillshade_path = RASTER_DIR / f"Hillshade_{volcano}.tif"
    save_hillshade(hillshade_path, profile, shade)

    print(f"[{volcano}] Construyendo malla 3D de 5 m", flush=True)
    terrain, center_x, center_y, base, half_width, half_height, relief = build_surface(
        elevation, valid, transform, shade, global_min, global_max
    )
    low_level = int(math.ceil(minimum / CONTOUR_INTERVAL_M) * CONTOUR_INTERVAL_M)
    high_level = int(math.floor(maximum / CONTOUR_INTERVAL_M) * CONTOUR_INTERVAL_M)
    contour_levels = np.arange(low_level, high_level + 1, CONTOUR_INTERVAL_M, dtype=np.float64)
    contour_mesh, contour_features = extract_contour_features(
        terrain, center_x, center_y, base, contour_levels
    )

    contours_path = VECTOR_DIR / f"Curvas_30m_{volcano}.gpkg"
    number_levels = create_contour_geopackage(contours_path, crs.to_wkt(), contour_features)
    print(f"[{volcano}] Curvas: {number_levels} cotas en {contours_path.name}", flush=True)

    mesh_path = OUTPUT_DIR / f"Modelo_3D_{volcano}.vtp"
    terrain.save(mesh_path)
    print(f"[{volcano}] Malla: {terrain.n_points:,} vértices, {terrain.n_cells:,} caras", flush=True)

    if preview:
        width, height, duration, fps = 540, 960, 4, 12
        video_path = OUTPUT_DIR / f"Video_DEM_{volcano}_preview.mp4"
    else:
        video_path = OUTPUT_DIR / f"Video_DEM_{volcano}_360.mp4"
    render_movie(
        terrain,
        contour_mesh,
        volcano,
        video_path,
        center_x,
        center_y,
        half_width,
        half_height,
        relief,
        minimum,
        maximum,
        global_min,
        global_max,
        duration,
        fps,
        width,
        height,
    )
    print(f"[{volcano}] Video guardado: {video_path}", flush=True)


def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera curvas de 30 m, hillshade y videos verticales 3D a partir de DEM GeoTIFF."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--todos", action="store_true", help="Procesa y renderiza los cuatro volcanes.")
    selection.add_argument("--volcan", choices=sorted(VOLCANOES), help="Procesa un solo volcán.")
    selection.add_argument("--preview", action="store_true", help="Vista previa breve vertical de Cerro Tusa.")
    return parser.parse_args()


def main() -> None:
    args = read_arguments()
    if args.todos:
        selected = list(VOLCANOES.items())
        duration, fps, width, height, preview = 30, 30, 1080, 1920, False
    elif args.preview:
        selected = [("Cerro_Tusa", VOLCANOES["Cerro_Tusa"])]
        duration, fps, width, height, preview = 4, 12, 540, 960, True
    else:
        selected = [(args.volcan, VOLCANOES[args.volcan])]
        duration, fps, width, height, preview = 30, 30, 1080, 1920, False

    extrema = []
    for _, filename in selected:
        path = INPUT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el DEM: {path}")
        with rasterio.open(path) as source:
            values = source.read(1, masked=False)
            mask = valid_dem(values)
            if not np.any(mask):
                raise ValueError(f"El DEM {path.name} no tiene valores válidos.")
            extrema.append((float(np.min(values[mask])), float(np.max(values[mask]))))
    global_min = min(item[0] for item in extrema)
    global_max = max(item[1] for item in extrema)
    print(f"Rango de color común: {global_min:.2f}–{global_max:.2f} m", flush=True)

    pv.OFF_SCREEN = True
    for volcano, filename in selected:
        process_volcano(
            volcano,
            filename,
            global_min,
            global_max,
            duration,
            fps,
            width,
            height,
            preview,
        )


if __name__ == "__main__":
    main()
