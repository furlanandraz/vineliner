"""The plugin inside QGIS: provider, parameters and outputs of Detect vine rows, on synthetic GeoTIFFs.

Run with QGIS's own Python (it has qgis and osgeo; it needs no pytest):

    "C:\\Program Files\\QGIS 3.44.15\\bin\\python-qgis-ltr.bat" tests/qgis/test_plugin.py -v

Under plain pytest these tests are skipped, since qgis cannot be imported there.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from osgeo import gdal, osr
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsFeature,
        QgsField,
        QgsGeometry,
        QgsPointXY,
        QgsProcessingContext,
        QgsProcessingFeedback,
        QgsRasterLayer,
        QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QMetaType
except ImportError as exc:  # plain Python: nothing to test here
    raise unittest.SkipTest(f"QGIS is not available: {exc}") from exc

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

THETA, SPACING, PIXEL, SIZE = 30.0, 1.5, 0.2, 300  # rows at 30 deg from east, 1.5 m apart, 60 m scene
X0, Y0 = 500000.0, 100060.0
_APP = None
_PROVIDER = None  # kept alive: QGIS cannot create the algorithms of a garbage-collected Python provider


def setUpModule():
    global _APP, _PROVIDER
    _APP = QgsApplication([], False, tempfile.mkdtemp(prefix="vineliner_qgis_profile_"))  # a throwaway profile
    _APP.initQgis()
    sys.path.append(os.path.join(QgsApplication.prefixPath(), "python", "plugins"))
    from processing.core.Processing import Processing

    Processing.initialize()
    from vineliner.processing_provider.provider import VinelinerProvider

    _PROVIDER = VinelinerProvider()
    QgsApplication.processingRegistry().addProvider(_PROVIDER)


def tearDownModule():
    import gc

    from qgis.core import QgsMapLayer, QgsProject

    QgsProject.instance().removeAllMapLayers()
    for cls in (DetectRowsTest, InputCheckTest):  # layers still alive at interpreter exit outlive QGIS and crash it
        for name, value in list(vars(cls).items()):
            if isinstance(value, QgsMapLayer):
                delattr(cls, name)
    gc.collect()
    _APP.exitQgis()


def stripes(half_width=0.3):
    rows, cols = np.indices((SIZE, SIZE))
    x = X0 + (cols + 0.5) * PIXEL
    y = Y0 - (rows + 0.5) * PIXEL
    t = math.radians(THETA)
    s = (x - X0) * -math.sin(t) + (y - Y0) * math.cos(t)
    return (np.abs(s - SPACING * np.round(s / SPACING)) <= half_width).astype(np.float32)


def write_tif(path, bands, epsg=3794, nodata=None, geographic=False):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), SIZE, SIZE, len(bands), gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    if geographic:  # the same scene, in degrees near Cormons
        ds.SetGeoTransform((13.46, PIXEL / 77360.0, 0, 45.97, 0, -PIXEL / 111130.0))
    else:
        ds.SetGeoTransform((X0, PIXEL, 0, Y0, 0, -PIXEL))
    ds.SetProjection(srs.ExportToWkt())
    for i, band in enumerate(bands, start=1):
        b = ds.GetRasterBand(i)
        b.WriteArray(band)
        if nodata is not None:
            b.SetNoDataValue(nodata)
    ds = None
    return str(path)


def polygon_layer(crs="EPSG:3794", coords=None):
    layer = QgsVectorLayer(f"Polygon?crs={crs}", "vineyards", "memory")
    layer.dataProvider().addAttributes([QgsField("name", QMetaType.Type.QString)])
    layer.updateFields()
    coords = coords or [(X0 + 8, Y0 - 10), (X0 + 50, Y0 - 6), (X0 + 54, Y0 - 50), (X0 + 10, Y0 - 52)]
    f = QgsFeature(layer.fields())
    f.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in coords]]))
    f.setAttribute("name", "block A")
    layer.dataProvider().addFeatures([f])
    return layer


def lon_lat():
    """The test block's corners in degrees near Cormons (matching write_tif(geographic=True))."""
    return [(13.46 + (x - X0) / 77360.0, 45.97 + (y - Y0) / 111130.0)
            for x, y in [(X0 + 8, Y0 - 10), (X0 + 50, Y0 - 6), (X0 + 54, Y0 - 50), (X0 + 10, Y0 - 52)]]


class DetectRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vineliner_qgis_test_"))
        V = stripes()
        rng = np.random.default_rng(0)
        gray = np.clip(200 - 110 * V + rng.normal(0, 8, V.shape), 0, 254).astype(np.float32)
        nir = (1000 + 1200 * V + rng.normal(0, 60, V.shape)).astype(np.float32)
        cls.rgb = QgsRasterLayer(write_tif(cls.tmp / "rgb.tif", [gray, gray, gray], nodata=255), "rgb")
        cls.nrg = QgsRasterLayer(write_tif(cls.tmp / "nrg.tif", [nir, gray, gray]), "nrg")
        ndvi = np.clip(0.3 + 0.5 * V + rng.normal(0, 0.04, V.shape), -1, 1).astype(np.float32)
        cls.ndvi = QgsRasterLayer(write_tif(cls.tmp / "ndvi.tif", [ndvi]), "ndvi")
        cls.ndvi_geo = QgsRasterLayer(write_tif(cls.tmp / "ndvi_4326.tif", [ndvi], epsg=4326, geographic=True), "ndvi_4326")
        cls.blocks = polygon_layer()

    def run_alg(self, **layers):
        import processing

        params = {
            "RGB": layers.get("rgb"), "NIR_R_G": layers.get("nrg"), "NDVI": layers.get("ndvi"),
            "VINEYARDS": layers.get("blocks", self.blocks), "ROW_WIDTH": layers.get("width", [1.4, 1.6]),
            "OUTPUT": "memory:rows", "BLOCKS_OUTPUT": "memory:blocks",
            "REPORT": str(self.tmp / f"report_{len(os.listdir(self.tmp))}.json"),
        }
        context, feedback = QgsProcessingContext(), QgsProcessingFeedback()
        result = processing.run("vineliner:detect_rows", params, context=context, feedback=feedback)

        def layer(value):  # processing.run hands back memory layers themselves; ids are taken from the context
            return context.takeResultLayer(value) if isinstance(value, str) else value

        return layer(result["OUTPUT"]), layer(result["BLOCKS_OUTPUT"]), result

    def test_rows_have_exactly_one_boolean_attribute(self):
        rows, blocks, result = self.run_alg(rgb=self.rgb)
        self.assertEqual([f.name() for f in rows.fields()], ["needs_review"])
        self.assertEqual(rows.fields()[0].type(), QMetaType.Type.Bool)
        self.assertGreaterEqual(rows.featureCount(), 30)  # the block holds about 36 rows
        flags = [f["needs_review"] for f in rows.getFeatures()]
        self.assertTrue(all(isinstance(v, bool) for v in flags) and not all(flags))
        self.assertTrue(Path(result["REPORT"]).exists())

    def test_polygons_keep_their_attributes_and_get_the_statistics(self):
        _, blocks, _ = self.run_alg(rgb=self.rgb)
        names = [f.name() for f in blocks.fields()]
        self.assertEqual(names, ["name", "average_row_length", "average_row_direction", "average_row_width", "confidence_interval"])
        f = next(blocks.getFeatures())
        self.assertEqual(f["name"], "block A")
        self.assertAlmostEqual(f["average_row_direction"], 60.0, delta=0.5)  # 30 deg from east = 60 deg from north
        self.assertAlmostEqual(f["average_row_width"], SPACING, delta=0.03)
        self.assertTrue(0 < f["confidence_interval"] < 0.05)
        self.assertGreater(f["average_row_length"], 20)
        # Row ends are vertices of the output polygon.
        vertices = {(round(p.x(), 6), round(p.y(), 6)) for p in f.geometry().vertices()}
        rows, _, _ = self.run_alg(rgb=self.rgb)
        ends = [(round(p.x(), 6), round(p.y(), 6)) for r in rows.getFeatures() for p in (r.geometry().vertexAt(0),)]
        self.assertTrue(set(ends) <= vertices)

    def test_nir_r_g_alone_and_ndvi_alone(self):
        for kwargs in ({"nrg": self.nrg}, {"ndvi": self.ndvi}, {"rgb": self.rgb, "nrg": self.nrg}):
            rows, _, result = self.run_alg(**kwargs)
            self.assertGreaterEqual(rows.featureCount(), 30, kwargs)
        import json

        report = json.loads(Path(result["REPORT"]).read_text())
        self.assertEqual(report["inputs"]["components"], ["rgb", "nir", "ndvi"])
        self.assertEqual(report["inputs"]["ndvi"], "computed from NIR-R-G")
        self.assertEqual(report["inputs"]["analysis_grid"]["on_native_grid_of"], "rgb")

    def test_a_geographic_layer_is_put_on_a_metric_grid(self):
        rows, blocks, result = self.run_alg(ndvi=self.ndvi_geo, blocks=polygon_layer("EPSG:4326", lon_lat()))
        self.assertTrue(rows.crs().authid().startswith("EPSG:326"))  # WGS 84 / UTM
        self.assertGreaterEqual(rows.featureCount(), 25)
        f = next(blocks.getFeatures())
        self.assertAlmostEqual(f["average_row_width"], SPACING, delta=0.06)

    def test_osr_transforms_match_what_qgis_gives_when_it_works(self):
        from qgis.core import QgsCoordinateTransform, QgsProject, QgsRectangle

        from vineliner.qgis_io.transform import Transform

        wgs84, utm = QgsCoordinateReferenceSystem("EPSG:4326"), QgsCoordinateReferenceSystem("EPSG:32633")
        qgis = QgsCoordinateTransform(wgs84, utm, QgsProject.instance()).transform(QgsPointXY(13.461, 45.971))
        x, y = Transform(wgs84, utm, "test").point(13.461, 45.971)
        self.assertAlmostEqual(x, qgis.x(), delta=1e-3)
        self.assertAlmostEqual(y, qgis.y(), delta=1e-3)
        rect = QgsRectangle(13.46, 45.97, 13.47, 45.98)
        ours = Transform(wgs84, utm, "test").rectangle(rect)
        theirs = QgsCoordinateTransform(wgs84, utm, QgsProject.instance()).transformBoundingBox(rect)
        for a, b in zip((ours.xMinimum(), ours.yMinimum(), ours.xMaximum(), ours.yMaximum()),
                        (theirs.xMinimum(), theirs.yMinimum(), theirs.xMaximum(), theirs.yMaximum())):
            self.assertAlmostEqual(a, b, delta=0.01)

    def test_untransformed_coordinates_are_caught(self):
        from qgis.core import QgsProcessingException

        from vineliner.qgis_io.transform import require_transformed

        wgs84, utm = QgsCoordinateReferenceSystem("EPSG:4326"), QgsCoordinateReferenceSystem("EPSG:32633")
        require_transformed(wgs84, utm, (13.461, 45.971), (380769.9, 5091976.9))  # transformed: fine
        require_transformed(utm, QgsCoordinateReferenceSystem("EPSG:3794"), (1.0, 2.0), (1.0, 2.0))  # both metric: may coincide
        with self.assertRaisesRegex(QgsProcessingException, "returned the coordinates unchanged"):
            require_transformed(wgs84, utm, (13.461, 45.971), (13.461, 45.971))

    def test_at_least_one_raster_layer_is_required(self):
        from vineliner.processing_provider.detect_rows import DetectRowsAlgorithm

        alg = DetectRowsAlgorithm()
        alg.initAlgorithm()
        outputs = {"OUTPUT": "memory:rows", "BLOCKS_OUTPUT": "memory:blocks"}
        ok, message = alg.checkParameterValues({"VINEYARDS": self.blocks, "ROW_WIDTH": [1.4, 1.6]} | outputs, QgsProcessingContext())
        self.assertFalse(ok)
        self.assertIn("at least one raster layer", message)
        ok, message = alg.checkParameterValues(
            {"NDVI": self.rgb, "RGB": self.ndvi, "VINEYARDS": self.blocks, "ROW_WIDTH": [1.4, 1.6]} | outputs, QgsProcessingContext())
        self.assertFalse(ok)  # a one-band layer in the RGB slot
        self.assertIn("needs 3", message)


class InputCheckTest(unittest.TestCase):
    """Layers the algorithm cannot use are refused with a message saying why, before anything runs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vineliner_qgis_checks_"))
        V = stripes()
        cls.blocks = polygon_layer()
        # A colour rendering of NDVI: 4 bands of 8-bit colours, like DJI Terra's index_map_color/*.tif.
        colours = [(V * 200 + 30).astype(np.float32)] * 3 + [np.full_like(V, 255)]
        cls.colour_ndvi = QgsRasterLayer(write_tif(cls.tmp / "ndvi_colour.tif", colours), "ndvi_colour")
        # A raster without a CRS, like DJI Terra's *_local.tif maps.
        path = cls.tmp / "no_crs.tif"
        ds = gdal.GetDriverByName("GTiff").Create(str(path), SIZE, SIZE, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((-67.0, 0.012, 0, 78.0, 0, -0.012))
        ds.GetRasterBand(1).WriteArray(V)
        ds = None
        cls.no_crs = QgsRasterLayer(str(path), "no_crs")
        cls.no_crs.setCrs(QgsCoordinateReferenceSystem())  # as QGIS leaves it when told to keep an unknown CRS

    def check(self, **params):
        from vineliner.processing_provider.detect_rows import DetectRowsAlgorithm

        alg = DetectRowsAlgorithm()
        alg.initAlgorithm()
        base = {"VINEYARDS": self.blocks, "ROW_WIDTH": [1.4, 1.6], "OUTPUT": "memory:rows", "BLOCKS_OUTPUT": "memory:blocks"}
        return alg.checkParameterValues(base | params, QgsProcessingContext())

    def test_a_colour_rendering_is_refused_as_ndvi(self):
        ok, message = self.check(NDVI=self.colour_ndvi)
        self.assertFalse(ok)
        self.assertIn("has 4 bands", message)
        self.assertIn("index_map/NDVI.tif", message)

    def test_a_raster_without_a_crs_is_refused(self):
        ok, message = self.check(NDVI=self.no_crs)
        self.assertFalse(ok)
        self.assertIn("no coordinate reference system", message)

    def test_a_local_frame_raster_saved_as_wgs84_is_refused(self):
        """DJI Terra's *_local.tif re-saved with WGS 84 attached: 1.2 cm pixels in metres, read as degrees (1.36 km)."""
        path = self.tmp / "ndvi_grayscale.tif"
        ds = gdal.GetDriverByName("GTiff").Create(str(path), SIZE, SIZE, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((-67.0608, 0.012222, 0, 78.2784, 0, -0.012222))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(stripes() * 255)
        ds = None
        ok, message = self.check(NDVI=QgsRasterLayer(str(path), "ndvi_grayscale"))
        self.assertFalse(ok)
        self.assertIn("about 1.36 km on the ground", message)
        self.assertIn("local frame", message)
        self.assertIn("does not move or rescale", message)

    def test_a_crs_that_contradicts_the_coordinates_is_refused(self):
        """A CRS set in Layer Properties > Source, over a file whose coordinates are in other units."""
        V = stripes()
        degrees_as_metres = QgsRasterLayer(write_tif(self.tmp / "ndvi_4326.tif", [V], epsg=4326, geographic=True), "ndvi")
        degrees_as_metres.setCrs(QgsCoordinateReferenceSystem("EPSG:32633"))
        ok, message = self.check(NDVI=degrees_as_metres)
        self.assertFalse(ok)
        self.assertIn("look like longitudes and latitudes", message)
        self.assertIn("Layer Properties > Source", message)
        metres_as_degrees = QgsRasterLayer(write_tif(self.tmp / "ndvi_3794.tif", [V]), "ndvi")
        metres_as_degrees.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        ok, message = self.check(NDVI=metres_as_degrees)
        self.assertFalse(ok)
        self.assertIn("they look like metres", message)
        ok, message = self.check(NDVI=QgsRasterLayer(write_tif(self.tmp / "ndvi_ok.tif", [V]), "ndvi"),
                                 VINEYARDS=polygon_layer("EPSG:3794", lon_lat()))
        self.assertFalse(ok)
        self.assertIn("vineyard polygon layer", message)

    def test_the_analysis_grid_refuses_implausible_sizes(self):
        """The checks behind the dialog's: whatever gets past it still stops with a reason, not a traceback."""
        from qgis.core import QgsProcessingException

        from vineliner.qgis_io import analysis_grid

        V = stripes()
        ndvi = QgsRasterLayer(write_tif(self.tmp / "grid_ndvi.tif", [V]), "ndvi")

        def grid(layer, polygons, pixel=0.0):
            return analysis_grid({"ndvi": layer}, polygons.extent(), polygons.crs(), None, pixel, 1.5, 1.6, QgsProcessingContext())

        self.assertEqual(grid(ndvi, self.blocks).pixel_m, 0.2)  # the scene itself is fine
        degrees = QgsRasterLayer(write_tif(self.tmp / "grid_ndvi_4326.tif", [V], epsg=4326, geographic=True), "ndvi")
        degrees.setCrs(QgsCoordinateReferenceSystem("EPSG:32633"))
        with self.assertRaisesRegex(QgsProcessingException, "pixels of 2.59e-06 m"):
            grid(degrees, polygon_layer("EPSG:32633", lon_lat()))
        with self.assertRaisesRegex(QgsProcessingException, "polygons span only 0.000"):
            grid(ndvi, polygon_layer("EPSG:3794", lon_lat()))
        beside = polygon_layer(coords=[(X0 - 24, Y0 - 10), (X0 - 14, Y0 - 10), (X0 - 14, Y0 - 20), (X0 - 24, Y0 - 20)])
        with self.assertRaisesRegex(QgsProcessingException, "covers only 0.8 x"):
            grid(ndvi, beside)  # 14.8 m of margin reaches 0.8 m into the layer
        with self.assertRaisesRegex(QgsProcessingException, "more than 50 million pixels"):
            grid(ndvi, self.blocks, pixel=0.001)

    def test_an_empty_polygon_layer_is_refused(self):
        class Errors(QgsProcessingFeedback):  # what the dialog's log shows
            def __init__(self):
                super().__init__()
                self.errors = []

            def reportError(self, error, fatalError=False):  # noqa: N802 (QGIS naming)
                self.errors.append(error)

        empty = QgsVectorLayer("Polygon?crs=EPSG:3794", "empty", "memory")
        rgb = QgsRasterLayer(write_tif(self.tmp / "rgb.tif", [stripes()] * 3), "rgb")
        alg = QgsApplication.processingRegistry().createAlgorithmById("vineliner:detect_rows")
        feedback = Errors()
        params = {"RGB": rgb, "VINEYARDS": empty, "ROW_WIDTH": [1.4, 1.6], "OUTPUT": "memory:rows", "BLOCKS_OUTPUT": "memory:blocks"}
        self.assertFalse(alg.prepare(params, QgsProcessingContext(), feedback))  # QGIS reports the reason to the feedback
        self.assertTrue(any("no features" in e for e in feedback.errors), feedback.errors)


class PluginPackageTest(unittest.TestCase):
    def test_class_factory_and_metadata(self):
        import configparser

        import vineliner

        self.assertTrue(callable(vineliner.classFactory))
        meta = configparser.ConfigParser()
        meta.read(ROOT / "vineliner" / "metadata.txt", encoding="utf-8")
        for key in ("name", "qgisMinimumVersion", "description", "about", "version", "author", "email", "repository"):
            self.assertTrue(meta["general"].get(key), key)
        self.assertEqual(meta["general"]["hasProcessingProvider"], "yes")

    def test_no_missing_dependencies_in_this_qgis(self):
        from vineliner.plugin import missing_dependencies

        self.assertEqual(missing_dependencies(), [])

    def test_init_gui_adds_the_raster_menu_entry_and_a_plugins_toolbar_icon(self):
        """The icon goes on the Plugins toolbar, which QGIS shows by default (the Raster toolbar is hidden)."""
        from unittest import mock

        import vineliner

        iface = mock.MagicMock()
        iface.mainWindow.return_value = None  # a real QAction needs a real (or no) parent
        plugin = vineliner.classFactory(iface)
        # The module already registered the provider; a second one with the same id would be refused.
        with mock.patch.object(type(plugin), "initProcessing", lambda self: None):
            plugin.initGui()
        try:
            iface.addPluginToRasterMenu.assert_called_once()
            iface.addToolBarIcon.assert_called_once_with(plugin.action)
            iface.addRasterToolBarIcon.assert_not_called()
            self.assertEqual(plugin.action.text(), "Detect vine rows…")
        finally:
            plugin.unload()
        iface.removeToolBarIcon.assert_called_once()
        iface.removePluginRasterMenu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
