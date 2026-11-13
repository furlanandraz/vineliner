"""Processing algorithm: Detect vine rows.

Inputs: raster layers (RGB, NIR-R-G, NDVI; at least one), vineyard polygons,
and the range of row widths to search. Outputs: the rows as lines with one
attribute, needs_review, and the vineyard polygons with row statistics.
"""

from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterCrs,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRange,
    QgsProcessingParameterRasterLayer,
    QgsApplication,
)
from qgis.PyQt.QtCore import QCoreApplication, QMetaType
from qgis.PyQt.QtGui import QIcon

from ..core.config import ConfigError, RunSettings, plugin_config
from ..core.export import BLOCK_FIELDS, ROW_FIELDS, row_record, write_report
from ..core.pipeline import LAYER_BANDS, Canceled, InputError, Raster, assemble_inputs, run
from ..qgis_io import analysis_grid, coordinates_problem, layer_problem, plan_read, read_blocks, read_layer, to_qgs_geometry

ICON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.png")
PREPARE_SHARE = 0.10  # progress bar share for reading the layers; the run takes the rest
LAYER_PARAMETERS = {"rgb": "RGB", "nir": "NIR_R_G", "ndvi": "NDVI"}  # core layer key -> parameter name


class DetectRowsAlgorithm(QgsProcessingAlgorithm):
    RGB = "RGB"
    NIR_R_G = "NIR_R_G"
    NDVI = "NDVI"
    VINEYARDS = "VINEYARDS"
    ROW_WIDTH = "ROW_WIDTH"
    ANALYSIS_CRS = "ANALYSIS_CRS"
    PIXEL_SIZE = "PIXEL_SIZE"
    OUTPUT = "OUTPUT"
    BLOCKS_OUTPUT = "BLOCKS_OUTPUT"
    REPORT = "REPORT"

    def tr(self, text: str) -> str:
        return QCoreApplication.translate("DetectRowsAlgorithm", text)

    def createInstance(self):  # noqa: N802 (QGIS naming)
        return DetectRowsAlgorithm()

    def name(self) -> str:
        return "detect_rows"

    def displayName(self) -> str:  # noqa: N802 (QGIS naming)
        return self.tr("Detect vine rows")

    def group(self) -> str:
        return ""

    def groupId(self) -> str:  # noqa: N802 (QGIS naming)
        return ""

    def icon(self) -> QIcon:
        return QIcon(ICON)

    def shortHelpString(self) -> str:  # noqa: N802 (QGIS naming)
        return self.tr(
            "Finds the vine rows inside vineyard polygons and draws each row as a line.\n\n"
            "Raster layers (at least one):\n"
            "• RGB: bands red, green, blue.\n"
            "• NIR-R-G: a false-colour layer with bands near-infrared, red, green. "
            "With it, NDVI is computed from its NIR and red bands unless an NDVI layer is given.\n"
            "• NDVI: one band, values from -1 to 1.\n"
            "Every layer given is used: RGB alone, RGB with NIR-R-G (plus computed NDVI), NDVI alone, or any mix.\n\n"
            "Vineyard polygons: one polygon per planting block; rows are searched and clipped inside each.\n\n"
            "Row width range: the distance between neighbouring rows to search, minimum to maximum, in metres. "
            "21 widths across the range are tried, and rows spaced outside it are flagged.\n\n"
            "Outputs:\n"
            "• Vine rows: lines with one attribute, needs_review, true where a person should check the row "
            "(a dangling end, a crossing, too much drawn without image support, spacing off, weak evidence, "
            "a short row, an uncertain position in the row order, or a row added between rows the image did not show).\n"
            "• Vineyard row statistics: the polygons with average_row_length (m), average_row_direction "
            "(degrees clockwise from north, 0-180), average_row_width (m) and confidence_interval: the half-width "
            "of the 95 % confidence interval of average_row_width (m).\n\n"
            "Advanced: the analysis CRS (default: the first layer's CRS if projected in metres, else the project's, "
            "else UTM) and pixel size (default: the coarsest layer's pixel, at least 1/13 of the middle row width)."
        )

    def initAlgorithm(self, config=None):  # noqa: N802 (QGIS naming)
        self.addParameter(QgsProcessingParameterRasterLayer(self.RGB, self.tr("RGB layer (red, green, blue)"), optional=True))
        self.addParameter(QgsProcessingParameterRasterLayer(self.NIR_R_G, self.tr("NIR-R-G layer (near-infrared, red, green)"), optional=True))
        self.addParameter(QgsProcessingParameterRasterLayer(self.NDVI, self.tr("NDVI layer"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.VINEYARDS, self.tr("Vineyard polygons"), [Qgis.ProcessingSourceType.VectorPolygon]))
        self.addParameter(QgsProcessingParameterRange(
            self.ROW_WIDTH, self.tr("Row width range, minimum to maximum (m)"), Qgis.ProcessingNumberParameterType.Double,
            defaultValue="1.8,2.2"))

        crs = QgsProcessingParameterCrs(self.ANALYSIS_CRS, self.tr("Analysis CRS (projected, metres; empty = automatic)"), optional=True)
        pixel = QgsProcessingParameterNumber(
            self.PIXEL_SIZE, self.tr("Analysis pixel size (m; 0 = automatic)"), Qgis.ProcessingNumberParameterType.Double,
            defaultValue=0.0, minValue=0.0)
        report = QgsProcessingParameterFileDestination(
            self.REPORT, self.tr("Run report (JSON)"), self.tr("JSON files (*.json)"), optional=True, createByDefault=False)
        for advanced in (crs, pixel, report):
            advanced.setFlags(advanced.flags() | Qgis.ProcessingParameterFlag.Advanced)
            self.addParameter(advanced)

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("Vine rows"), Qgis.ProcessingSourceType.VectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.BLOCKS_OUTPUT, self.tr("Vineyard row statistics"), Qgis.ProcessingSourceType.VectorPolygon))

    def checkParameterValues(self, parameters, context):  # noqa: N802 (QGIS naming)
        ok, message = super().checkParameterValues(parameters, context)
        if not ok:
            return ok, message
        layers = self._layers(parameters, context)
        if not layers:
            return False, self.tr("Select at least one raster layer: RGB, NIR-R-G or NDVI.")
        for key, layer in layers.items():
            problem = layer_problem(key, layer)
            if problem:
                return False, self.tr(problem)
        problem = self._polygons_problem(parameters, context)
        if problem:
            return False, self.tr(problem)
        lo, hi = self.parameterAsRange(parameters, self.ROW_WIDTH, context)
        if not 0 < lo <= hi:
            return False, self.tr("The row width range needs 0 < minimum <= maximum.")
        return True, ""

    def _polygons_problem(self, parameters, context) -> str | None:
        source = self.parameterAsSource(parameters, self.VINEYARDS, context)
        if source is None:
            return None  # the base class reports a missing source
        return coordinates_problem(f"vineyard polygon layer '{source.sourceName()}'", source.sourceCrs(), source.sourceExtent())

    def _layers(self, parameters, context) -> dict:
        out = {}
        for key, name in LAYER_PARAMETERS.items():
            layer = self.parameterAsRasterLayer(parameters, name, context)
            if layer is not None:
                out[key] = layer
        return out

    def prepareAlgorithm(self, parameters, context, feedback):  # noqa: N802 (QGIS naming)
        """Everything that touches QGIS layers and CRSs, on the main thread.

        QGIS always calls this on the main thread, then processAlgorithm on a
        worker thread; QGIS layers are not safe to use from other threads.
        Coordinates are transformed with GDAL's OSR (qgis_io/transform.py).

        A refusal is reported to the feedback (the dialog's log) and returns
        False rather than raising: an exception leaving this method through
        QGIS keeps its frames alive, and with them a feature source over a
        layer that is later deleted, which crashes QGIS when it exits.
        """
        try:
            self._prepared = self._prepare(parameters, context, feedback)
        except QgsProcessingException as exc:
            feedback.reportError(str(exc), True)
            return False
        return True

    def _prepare(self, parameters, context, feedback) -> dict:
        layers = self._layers(parameters, context)
        if not layers:
            raise QgsProcessingException(self.tr("Select at least one raster layer: RGB, NIR-R-G or NDVI."))
        for key, layer in layers.items():  # also when run without the dialog's checks
            problem = layer_problem(key, layer)
            if problem:
                raise QgsProcessingException(self.tr(problem))
        source = self.parameterAsSource(parameters, self.VINEYARDS, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.VINEYARDS))
        if source.featureCount() == 0:
            raise QgsProcessingException(self.tr(
                "The vineyard polygon layer has no features (or none selected, with 'Selected features only' on)."))
        problem = self._polygons_problem(parameters, context)
        if problem:
            raise QgsProcessingException(self.tr(problem))
        lo, hi = self.parameterAsRange(parameters, self.ROW_WIDTH, context)
        try:
            settings = RunSettings.from_config(plugin_config(lo, hi))
        except (ConfigError, ValueError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        # Analysis grid and inputs.
        target = analysis_grid(
            layers, source.sourceExtent(), source.sourceCrs(), self.parameterAsCrs(parameters, self.ANALYSIS_CRS, context),
            self.parameterAsDouble(parameters, self.PIXEL_SIZE, context), (lo + hi) / 2, hi, context,
        )
        feedback.pushInfo(
            f"Analysis grid: {target.crs.authid() or target.crs.description()}, {target.pixel_m:g} m pixels "
            f"({target.notes['pixel_from']}), {target.width} x {target.height} px"
        )
        plans = {key: plan_read(key, layer, list(range(1, len(LAYER_BANDS[key]) + 1)), target)
                 for key, layer in layers.items()}
        blocks, features = read_blocks(source, target.crs, context, feedback)
        return {"settings": settings, "target": target, "plans": plans, "blocks": blocks,
                "features": features, "fields": QgsFields(source.fields())}

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802 (QGIS naming)
        """The run, on the worker thread: GDAL reads, the NumPy core, and the outputs."""
        prepared = self._prepared
        settings, target, plans = prepared["settings"], prepared["target"], prepared["plans"]
        blocks, features = prepared["blocks"], prepared["features"]
        grid = target.grid()
        rasters = {}
        for i, (key, plan) in enumerate(plans.items()):
            if feedback.isCanceled():
                return {}
            feedback.pushInfo(f"Reading the {key.upper()} layer '{plan.name}' ...")
            data, metadata = read_layer(plan, target)
            rasters[key] = Raster(data, grid, LAYER_BANDS[key], metadata)
            feedback.pushInfo(f"  {metadata['resampling']} from {metadata['native_pixel_m']:.4g} m; {100 * metadata['valid_share']:.1f}% of the grid has data")
            feedback.setProgress(100 * PREPARE_SHARE * (i + 1) / (len(plans) + 1))
        try:
            inputs = assemble_inputs(rasters, blocks)
        except (InputError, ValueError) as exc:
            raise QgsProcessingException(str(exc)) from exc
        feedback.setProgress(100 * PREPARE_SHARE)

        # The run.
        cache_dir = os.path.join(QgsApplication.qgisSettingsDirPath(), "cache", "vineliner", "kernels")

        def log(message: str):  # the core's warnings show as warnings in the Processing log
            (feedback.pushWarning if "WARNING" in message else feedback.pushInfo)(message)

        try:
            result = run(
                settings, inputs, cache_dir=cache_dir, log=log,
                progress=lambda f: feedback.setProgress(100 * (PREPARE_SHARE + (1 - PREPARE_SHARE) * f)),
                is_canceled=feedback.isCanceled,
            )
        except Canceled:
            return {}
        except InputError as exc:
            raise QgsProcessingException(str(exc)) from exc

        # Vine rows: one attribute.
        row_fields = QgsFields()
        row_fields.append(QgsField(ROW_FIELDS[0], QMetaType.Type.Bool))
        sink, rows_id = self.parameterAsSink(parameters, self.OUTPUT, context, row_fields, Qgis.WkbType.LineString, target.crs)
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))
        for section, review in zip(result.sections, result.reviews):
            feature = QgsFeature(row_fields)
            feature.setGeometry(to_qgs_geometry(section.geometry))
            feature.setAttribute(ROW_FIELDS[0], row_record(review)[ROW_FIELDS[0]])
            sink.addFeature(feature, QgsFeatureSink.Flag.FastInsert)

        # Vineyard polygons: their own attributes plus the row statistics.
        block_fields = QgsFields()
        kept = [f for f in prepared["fields"] if f.name() not in BLOCK_FIELDS]
        for f in kept:
            block_fields.append(f)
        for name in BLOCK_FIELDS:
            block_fields.append(QgsField(name, QMetaType.Type.Double))
        block_sink, blocks_id = self.parameterAsSink(
            parameters, self.BLOCKS_OUTPUT, context, block_fields, Qgis.WkbType.MultiPolygon, target.crs)
        if block_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.BLOCKS_OUTPUT))
        by_id = {b.block.id: b for b in result.blocks}
        for feature in features:
            b = by_id[int(feature.id())]
            out = QgsFeature(block_fields)
            geometry = to_qgs_geometry(b.polygon)
            geometry.convertToMultiType()
            out.setGeometry(geometry)
            out.setAttributes([feature[f.name()] for f in kept] + [b.statistics[name] for name in BLOCK_FIELDS])
            block_sink.addFeature(out, QgsFeatureSink.Flag.FastInsert)
            s = b.statistics
            if s["rows"]:
                ci = f" +/- {s['confidence_interval']:.3f}" if s["confidence_interval"] is not None else ""
                width = f"{s['average_row_width']:.3f}{ci} m" if s["average_row_width"] is not None else "n/a"
                feedback.pushInfo(
                    f"Polygon {b.block.id}: {s['rows']} rows ({s['rows_needing_review']} need review), "
                    f"length {s['average_row_length']:.1f} m, direction {s['average_row_direction']:.1f} deg, width {width}"
                )
            else:
                feedback.pushWarning(f"Polygon {b.block.id}: no rows found.")

        outputs = {self.OUTPUT: rows_id, self.BLOCKS_OUTPUT: blocks_id}
        report_path = self.parameterAsFileOutput(parameters, self.REPORT, context)
        if report_path:
            result.report["inputs"]["analysis_grid"] = target.describe()
            write_report(report_path, result.report)
            outputs[self.REPORT] = report_path
        return outputs
