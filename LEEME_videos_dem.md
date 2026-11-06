# Videos 3D de los DEM

El programa procesa los cuatro DEM GeoTIFF de `1-Insumos` con Python. No requiere QGIS.

## Instalar

Desde PowerShell, ubicado en la carpeta del proyecto:

```powershell
python -m venv .venv-volcanes3d
.\.venv-volcanes3d\Scripts\python.exe -m pip install -r .\4-Scripts\requirements_videos_dem.txt
```

La prueba local se realizó con Python 3.15, Rasterio 1.5.2, PyVista 0.49.2 y VTK 9.7.2.

## Ejecutar

```powershell
# Previsualización de Cerro Tusa (4 s, resolución vertical reducida)
.\.venv-volcanes3d\Scripts\python.exe .\4-Scripts\generar_videos_dem.py --preview

# Video final de un volcán
.\.venv-volcanes3d\Scripts\python.exe .\4-Scripts\generar_videos_dem.py --volcan Cerro_Tusa

# Los cuatro videos finales
.\.venv-volcanes3d\Scripts\python.exe .\4-Scripts\generar_videos_dem.py --todos
```

Los videos finales son verticales, de 1080 × 1920, 30 fps y 30 segundos. La cámara orbita 360°, sube a una vista cenital y vuelve al ángulo inicial. Una flecha se orienta al norte geográfico en cada fotograma; la escala vertical de colores muestra sus elevaciones en metros. La exageración vertical es 1×.

## Archivos generados

- `2-Rasters`: un hillshade GeoTIFF por DEM.
- `3-Vectores`: un GeoPackage por volcán, con curvas 3D de 30 m y campo `cota_m`.
- `5-Salidas/Video_3D`: los MP4, las mallas VTP y la vista previa.

Los GeoTIFF originales conservan sus valores; los ceros se ignoran como NoData según los bordes sin datos de estos cuatro recortes. El GeoPackage de curvas conserva coordenadas absolutas EPSG:9377. Las mallas VTP usan coordenadas locales y guardan en sus metadatos el origen, la cota base y el CRS.
