import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import importlib

import numpy as np
import sitkUtils
import slicer
import SimpleITK as sitk
import vtk
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer import vtkMRMLScalarVolumeNode


class TAPSeg(ScriptedLoadableModule):


    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("TAPSeg")
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Y Zhang"]
        self.parent.helpText = _(
            'Tooth and pulp segmentation from CBCT volumes, with tooth hard-tissue mask generation. '
            '<a href="https://github.com/simzhangbest/TAPSeg">Documentation</a>.'
        )
        self.parent.acknowledgementText = _("""
 This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
 and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
 """)


@parameterNodeWrapper
class TAPSegParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode


class TAPSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.loadingWidget = slicer.util.loadUI(self.resourcePath('UI/loading.ui'))
        self.uiWidget = slicer.util.loadUI(self.resourcePath("UI/TAPSeg.ui"))
        self.ui = slicer.util.childWidgetVariables(self.uiWidget)
        self.loadingUi = slicer.util.childWidgetVariables(self.loadingWidget)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        self.layout.addWidget(self.uiWidget)

        self.uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = TAPSegLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.applyButton_1.connect("clicked(bool)", self.onToothPulp)
        self.ui.applyButton_2.connect("clicked(bool)", self.toothPulpBoolean)
        self.ui.batchApplyButton.connect("clicked(bool)", self.onBatchApplyButton)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        pass

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        pass

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode()

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode()

    def setParameterNode(self) -> None:
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        pass

    def _checkCanApply(self, caller=None, event=None) -> None:
        pass

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.process(self.logic.getToothWeightsDir(),
                               self.ui.inputSelector.currentNode(),
                               callback=self.pop_msg)
            slicer.util.messageBox("Tooth segmentation completed.")

    def toothPulpBoolean(self):
        "Generate the tooth hard-tissue mask by subtracting pulp."
        ok = False
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            ok = bool(self.logic.generateToothHardTissueMask(self.ui.inputSelector.currentNode(), callback=self.pop_msg))
        if ok:
            slicer.util.messageBox("Tooth hard tissue segmentation completed.")

    def onBatchApplyButton(self) -> None:
        "Run tooth, pulp, and hard-tissue segmentation for a directory."
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.batch_process(
                self.logic.getToothWeightsDir(),
                self.ui.input_dir.currentPath,
                self.ui.output_dir.currentPath,
                callback=self.pop_msg,
            )
        slicer.util.messageBox("Batch processing completed.")

    def pop_msg(self, prog=None, msg=""):
        self.loadingUi.loading_text.text = msg

    def onToothPulp(self):
        "Run nnU-Net v2 pulp segmentation."
        ok = False
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            ok = bool(self.logic.processPulp(self.ui.inputSelector.currentNode(), callback=self.pop_msg))
        if ok:
            slicer.util.messageBox("Pulp segmentation completed.")


class TAPSegLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return TAPSegParameterNode(super().getParameterNode())

    @staticmethod
    def _thisFileDir() -> Path:
        return Path(__file__).resolve().parent

    def getToothWeightsDir(self) -> str:
        "Locate tooth weights in module resources or the repository model directory."
        def _is_valid_weights_dir(p: Path) -> bool:
            if not p or not p.exists():
                return False
            if not (p / "roi").exists():
                return False
            has_cnt = (p / "cnt").exists() or (p / "skl_off").exists()
            has_ins = (p / "sin").exists() or any(
                (p / cand).exists() for cand in ("inst_v1_bd_test_zy", "inst_v1_bd_test", "inst_noise")
            )
            return bool(has_cnt and has_ins)

        forced = os.environ.get("TAPSEG_TOOTH_WEIGHTS_DIR", "").strip()
        if forced:
            forced_path = Path(forced).expanduser()
            if _is_valid_weights_dir(forced_path):
                return str(forced_path)

        candidate = self._thisFileDir() / "Resources" / "ML" / "tooth"
        if _is_valid_weights_dir(candidate):
            return str(candidate)

        for parent in self._thisFileDir().parents:
            candidate = parent / "model" / "tooth"
            if _is_valid_weights_dir(candidate):
                return str(candidate)

        raise FileNotFoundError(
            "Tooth weights directory not found.\n"
            "Expected structure:\n"
            "  - roi/\n"
            "  - cnt/ or skl_off/\n"
            "  - sin/ or inst_v1_bd_test_zy/inst_v1_bd_test/inst_noise/\n"
            "You can also set the environment variable TAPSEG_TOOTH_WEIGHTS_DIR before launching Slicer."
        )

    def getPulpModelDir(self) -> str:
        "Locate the pulp model, optionally overridden by TAPSEG_PULP_MODEL_DIR."

        def _is_valid_model_dir(p: Path) -> bool:
            if not p or not p.exists():
                return False
            if not (p / "plans.json").exists():
                return False
            if not (p / "dataset.json").exists():
                return False
            if not any((p / f"fold_{i}").exists() for i in range(5)) and not (p / "fold_all").exists():
                return False
            return True

        forced = os.environ.get("TAPSEG_PULP_MODEL_DIR", "").strip()
        if forced:
            forced_path = Path(forced).expanduser().resolve()
            if forced_path.name.lower().startswith("fold_"):
                forced_path = forced_path.parent
            if _is_valid_model_dir(forced_path):
                return str(forced_path)

            cand = forced_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
            if _is_valid_model_dir(cand):
                return str(cand)

        candidate = self._thisFileDir() / "Resources" / "ML" / "pulp" / "model"
        if _is_valid_model_dir(candidate):
            return str(candidate)

        candidate = self._thisFileDir() / "Resources" / "ML" / "pulp" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        if _is_valid_model_dir(candidate):
            return str(candidate)

        raise FileNotFoundError(
            "Pulp nnU-Net model directory not found.\n"
            "Expected path like: .../nnUNetTrainer__nnUNetPlans__3d_fullres "
            "(contains dataset.json, plans.json, fold_all/checkpoint_best.pth).\n"
            "You can also set the environment variable TAPSEG_PULP_MODEL_DIR before launching Slicer."
        )

    def _importToothBackend(self):
        "Import the bundled tooth backend without conflicting medic packages."
        moduleDir = self._thisFileDir()
        moduleDirStr = str(moduleDir)

        def _norm(p: str) -> str:
            try:
                return os.path.normcase(os.path.normpath(p))
            except Exception:
                return p

        sys.path[:] = [p for p in sys.path if _norm(p) != _norm(moduleDirStr)]
        sys.path.insert(0, moduleDirStr)
        importlib.invalidate_caches()

        loaded_medic = sys.modules.get("medic")
        if loaded_medic is not None:
            ok = False
            try:
                medic_paths = getattr(loaded_medic, "__path__", None)
                if medic_paths:
                    ok = any(_norm(str(moduleDir / "medic")) in _norm(str(p)) for p in medic_paths)
                else:
                    medic_file = getattr(loaded_medic, "__file__", "") or ""
                    ok = _norm(moduleDirStr) in _norm(medic_file)
            except Exception:
                ok = False

            if not ok:
                for k in list(sys.modules.keys()):
                    if k == "medic" or k.startswith("medic."):
                        del sys.modules[k]

        try:
            return importlib.import_module("medic.tooth_backend_simzhang")
        except SystemExit as e:
            raise RuntimeError(
                "SystemExit raised while importing the tooth backend "
                "(it may be parsing command-line arguments at import time)."
            ) from e
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Failed to import tooth backend: 'medic.tooth_backend_simzhang' not found.\n"
                "Common causes:\n"
                "1) Incomplete module folder (missing TAPSeg/medic/tooth_backend_simzhang.py).\n"
                "2) Another extension provides a top-level package named 'medic', causing import conflicts.\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Failed to import tooth backend. Dependencies may be missing (e.g., torch/scipy/scikit-image), "
                "or weights/code are incompatible."
            ) from e

    def process(self, model_dir, input_volume: vtkMRMLScalarVolumeNode, callback=None) -> None:
        if not input_volume:
            slicer.util.errorDisplay("Please select an input volume.")
            return

        print("[TAPSeg] Tooth segmentation started.", flush=True)

        weights_dir = model_dir
        try:
            if not weights_dir or os.path.normpath(weights_dir) in [".", "./model", ".\\model"]:
                weights_dir = self.getToothWeightsDir()
        except Exception:
            weights_dir = self.getToothWeightsDir()

        with tempfile.TemporaryDirectory(prefix="TAPSeg_") as tempDir:
            in_file = os.path.join(tempDir, "input.nrrd")
            out_file = os.path.join(tempDir, "pred.nrrd")

            input_volume_itk = sitkUtils.PullVolumeFromSlicer(input_volume)
            images = sitk.GetArrayFromImage(input_volume_itk)
            image = sitk.GetImageFromArray(images)
            image.SetSpacing(input_volume_itk.GetSpacing())
            image.SetDirection(input_volume_itk.GetDirection())
            image.SetOrigin(input_volume_itk.GetOrigin())
            sitk.WriteImage(image, in_file)

            if os.path.exists(in_file):
                self.compute_file("", weights_dir, in_file, out_file)
                if os.path.exists(out_file):
                    out_img = sitk.ReadImage(out_file)
                    seg_name = f"{input_volume.GetName()}_TAPSeg_tooth"
                    segNode = self._labelImageToSegmentationNode(
                        out_img,
                        reference_volume=input_volume,
                        seg_node_name=seg_name,
                        single_segment_name=None,
                    )

                    self._removeNodeByNameIfExists(f"{input_volume.GetName()}_TAPSeg_tooth_label")

                    try:
                        slicer.util.selectModule("SegmentEditor")
                        segEditorWidget = slicer.util.findChild(slicer.util.mainWindow(), "SegmentEditorWidget")
                        if segEditorWidget:
                            segEditorWidget.setMasterVolumeNode(input_volume)
                            segEditorWidget.setSegmentationNode(segNode)
                    except Exception:
                        pass
                else:
                    raise RuntimeError("Tooth segmentation did not generate an output file.")
            else:
                raise RuntimeError("Failed to export input volume for tooth segmentation.")

        print("[TAPSeg] Tooth segmentation finished.", flush=True)

    def batch_process(self, model_dir, input_dir, output_dir, callback=None):
        "Write tooth, pulp, and hard-tissue outputs; honor TAPSEG_BATCH_* overrides."

        def _as_path(p: str) -> Path:
            if p is None:
                return Path()
            p = str(p).strip()
            if not p:
                return Path()
            try:
                return Path(p).expanduser().resolve()
            except Exception:
                return Path(p)

        def _is_image_file(path: Path) -> bool:
            name = path.name.lower()
            return name.endswith((".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".nhdr"))

        def _strip_image_suffix(filename: str) -> str:
            lower = str(filename).lower()
            if lower.endswith(".nii.gz"):
                return filename[:-7]
            if lower.endswith(".nii"):
                return filename[:-4]
            return Path(filename).stem

        input_dir = str(input_dir) if input_dir is not None else ""
        output_dir = str(output_dir) if output_dir is not None else ""

        in_dir = _as_path(input_dir)
        out_dir = _as_path(output_dir)

        if not in_dir or not in_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {in_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        disable_pulp = os.environ.get("TAPSEG_BATCH_DISABLE_PULP", "").strip() in ("1", "true", "True", "YES", "yes")
        disable_hard = os.environ.get("TAPSEG_BATCH_DISABLE_HARD", "").strip() in ("1", "true", "True", "YES", "yes")
        recursive = os.environ.get("TAPSEG_BATCH_RECURSIVE", "").strip() in ("1", "true", "True", "YES", "yes")

        tooth_out_dir = out_dir / "tooth"
        pulp_out_dir = out_dir / "pulp"
        hard_out_dir = out_dir / "tooth_hard"
        tooth_out_dir.mkdir(parents=True, exist_ok=True)
        if not disable_pulp:
            pulp_out_dir.mkdir(parents=True, exist_ok=True)
        if not disable_hard:
            hard_out_dir.mkdir(parents=True, exist_ok=True)

        if recursive:
            candidates = [p for p in in_dir.rglob("*") if p.is_file() and _is_image_file(p)]
        else:
            candidates = [p for p in in_dir.iterdir() if p.is_file() and _is_image_file(p)]
        candidates.sort(key=lambda p: p.name.lower())

        if not candidates:
            raise FileNotFoundError(f"No image files found in input directory (nii/nii.gz/nrrd/...): {in_dir}")

        backend = self._importToothBackend()
        pulp_model_dir = None
        if not disable_pulp:
            pulp_model_dir = self.getPulpModelDir()

        total = len(candidates)
        ok_tooth = 0
        ok_pulp = 0
        ok_hard = 0
        failed = []

        print("[TAPSeg] Batch processing started.", flush=True)
        for idx, in_path in enumerate(candidates, start=1):
            stem = _strip_image_suffix(in_path.name)
            tooth_out = tooth_out_dir / f"{stem}.nii.gz"
            pulp_out = pulp_out_dir / f"{stem}.nii.gz"
            hard_out = hard_out_dir / f"{stem}.nii.gz"

            try:
                if callback:
                    callback((idx - 1) / max(total, 1), f"[{idx}/{total}] Segment Teeth: {in_path.name}")
                backend.exec_math_one("", str(in_path), str(tooth_out), str(model_dir))
                ok_tooth += 1
            except Exception as e:
                failed.append((in_path.name, f"tooth failed: {type(e).__name__}: {e}"))
                continue

            pulp_ready = False
            if not disable_pulp:
                try:
                    if callback:
                        callback((idx - 1) / max(total, 1), f"[{idx}/{total}] Segment Pulp: {in_path.name}")

                    in_for_pulp = str(in_path)
                    if not in_path.name.lower().endswith(".nii.gz"):
                        with tempfile.TemporaryDirectory(prefix="TAPSeg_BatchPulp_") as tempDir:
                            tmp_in = Path(tempDir) / "input.nii.gz"
                            img = sitk.ReadImage(str(in_path))
                            sitk.WriteImage(img, str(tmp_in), True)
                            in_for_pulp = str(tmp_in)
                            ok = self.nnunetInference(in_for_pulp, str(pulp_out), str(pulp_model_dir))
                    else:
                        ok = self.nnunetInference(in_for_pulp, str(pulp_out), str(pulp_model_dir))

                    if not ok or not pulp_out.exists():
                        raise RuntimeError("Output file was not generated.")

                    try:
                        tooth_img = sitk.ReadImage(str(tooth_out))
                        pulp_img = sitk.ReadImage(str(pulp_out))
                        tooth_lab = self._as_uint16_label(tooth_img)
                        pulp_lab = self._as_uint16_label(pulp_img)
                        if not self._same_sitk_geometry(tooth_lab, pulp_lab):
                            pulp_lab = sitk.Resample(
                                pulp_lab,
                                tooth_lab,
                                sitk.Transform(),
                                sitk.sitkNearestNeighbor,
                                0,
                                sitk.sitkUInt16,
                            )
                        tooth_arr = sitk.GetArrayFromImage(tooth_lab).astype(np.uint16, copy=False)
                        pulp_bin = (sitk.GetArrayFromImage(pulp_lab) != 0)
                        assigned = self._assignPulpComponentsToToothLabels(
                            tooth_labels=tooth_arr,
                            pulp_binary=pulp_bin,
                            reference_sitk_image=tooth_lab,
                        )
                        sitk.WriteImage(assigned, str(pulp_out), True)
                    except Exception as e:
                        failed.append((in_path.name, f"pulp assignment failed: {type(e).__name__}: {e}"))
                        pulp_ready = False
                        continue

                    ok_pulp += 1
                    pulp_ready = True
                except Exception as e:
                    failed.append((in_path.name, f"pulp failed: {type(e).__name__}: {e}"))
                    pulp_ready = False

            if not disable_hard:
                if not pulp_ready and not pulp_out.exists():
                    failed.append((in_path.name, "tooth_hard skipped: missing pulp output"))
                else:
                    try:
                        if callback:
                            callback((idx - 1) / max(total, 1), f"[{idx}/{total}] Generate Hard Tissue: {in_path.name}")

                        tooth_img = sitk.ReadImage(str(tooth_out))
                        pulp_img = sitk.ReadImage(str(pulp_out))
                        tooth_lab = self._as_uint16_label(tooth_img)
                        pulp_lab = self._as_uint16_label(pulp_img)

                        if not self._same_sitk_geometry(tooth_lab, pulp_lab):
                            pulp_lab = sitk.Resample(
                                pulp_lab,
                                tooth_lab,
                                sitk.Transform(),
                                sitk.sitkNearestNeighbor,
                                0,
                                sitk.sitkUInt16,
                            )

                        pulp_mask = sitk.Cast(pulp_lab != 0, sitk.sitkUInt8)
                        tooth_hard = sitk.Mask(tooth_lab, sitk.BinaryNot(pulp_mask), outsideValue=0)
                        tooth_hard = sitk.Cast(tooth_hard, sitk.sitkUInt16)
                        tooth_hard.CopyInformation(tooth_lab)

                        sitk.WriteImage(tooth_hard, str(hard_out), True)
                        ok_hard += 1
                    except Exception as e:
                        failed.append((in_path.name, f"tooth_hard failed: {type(e).__name__}: {e}"))

            try:
                slicer.app.processEvents()
            except Exception:
                pass

        if callback:
            callback(
                1.0,
                f"Batch completed: tooth={ok_tooth}/{total} pulp={ok_pulp}/{total} tooth_hard={ok_hard}/{total}",
            )

        print(
            f"[TAPSeg] Batch processing finished. "
            f"tooth={ok_tooth}/{total} pulp={ok_pulp}/{total} hard={ok_hard}/{total} failed={len(failed)}",
            flush=True,
        )

        write_report = os.environ.get("TAPSEG_BATCH_WRITE_REPORT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if write_report:
            try:
                report_path = out_dir / "tapseg_batch_report.txt"
                lines = [
                    "TAPSeg Batch Report",
                    f"input_dir={in_dir}",
                    f"output_dir={out_dir}",
                    f"total={total}",
                    f"tooth_ok={ok_tooth}",
                    f"pulp_ok={ok_pulp}",
                    f"tooth_hard_ok={ok_hard}",
                    f"failed={len(failed)}",
                    "",
                ]
                if failed:
                    lines.append("Failed/Skipped cases:")
                    for name, reason in failed:
                        lines.append(f"- {name}: {reason}")
                report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except Exception:
                pass

        return

    def compute_file(self, sign, model_dir, input_file, output_file, callback=None):
        backend = self._importToothBackend()
        backend.exec_math_one(sign, input_file, output_file, model_dir)
        return

    def toothPulpBooleanOperation(self, toothModelNode, pulpModelNode, operationType="DIFFERENCE"):
        "Apply UNION, DIFFERENCE, or INTERSECTION to tooth and pulp model nodes."
        if not toothModelNode or not pulpModelNode:
            slicer.util.errorDisplay("Tooth and pulp model nodes are required.")
            return None
        if not isinstance(toothModelNode, slicer.vtkMRMLModelNode) or not isinstance(pulpModelNode, slicer.vtkMRMLModelNode):
            slicer.util.errorDisplay("Inputs must be vtkMRMLModelNode.")
            return None

        toothPolyData = toothModelNode.GetPolyData()
        pulpPolyData = pulpModelNode.GetPolyData()
        if toothPolyData.GetNumberOfCells() == 0 or pulpPolyData.GetNumberOfCells() == 0:
            slicer.util.errorDisplay("Model polydata is empty; cannot perform boolean operation.")
            return None

        booleanFilter = vtk.vtkBooleanOperationPolyDataFilter()
        booleanFilter.SetInputData(0, toothPolyData)
        booleanFilter.SetInputData(1, pulpPolyData)

        if operationType.upper() == "UNION":
            booleanFilter.SetOperationToUnion()
        elif operationType.upper() == "DIFFERENCE":
            booleanFilter.SetOperationToDifference()
        elif operationType.upper() == "INTERSECTION":
            booleanFilter.SetOperationToIntersection()
        else:
            slicer.util.errorDisplay(
                f"Unsupported operation type: {operationType}. Supported: UNION/DIFFERENCE/INTERSECTION"
            )
            return None

        booleanFilter.SetTolerance(0.001)
        booleanFilter.Update()

        resultPolyData = booleanFilter.GetOutput()
        if resultPolyData.GetNumberOfCells() == 0:
            slicer.util.errorDisplay("Boolean operation result is empty. Please check if models are closed and overlapping.")
            return None

        resultNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        resultNode.SetName(f"{toothModelNode.GetName()}_{operationType}")
        resultNode.SetPolyData(resultPolyData)

        resultDisplayNode = resultNode.GetDisplayNode()
        if not resultDisplayNode:
            resultDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            resultNode.SetAndObserveDisplayNodeID(resultDisplayNode.GetID())
        resultDisplayNode.SetColor(1.0, 0.0, 0.0)
        resultDisplayNode.SetOpacity(1.0)

        return resultNode

    def slicerVolumeToNifti(self, volumeNode, outputNiftiPath):
        "Export a Slicer volume to NIfTI for nnU-Net inference."
        sitkImage = sitkUtils.PullVolumeFromSlicer(volumeNode)
        sitk.WriteImage(sitkImage, outputNiftiPath, True)
        return outputNiftiPath

    def nnunetInference(
        self,
        inputNiftiPath: str,
        outputNiftiPath: str,
        model_dir: str,
        *,
        folds=(0,),
        checkpoint: str = "checkpoint_best.pth",
        device: str = "auto",
        cuda_id: int = 0,
        perform_on_device: bool = False,
        num_proc_pre: int = 1,
        num_proc_export: int = 1,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> bool:

        try:
            folds = tuple(folds) if isinstance(folds, (list, tuple)) else (folds,)
            model_dir_path = Path(model_dir)
            if (
                folds == (0,)
                and (model_dir_path / "fold_all").exists()
                and not (model_dir_path / "fold_0").exists()
            ):
                folds = ("all",)
        except Exception:
            pass

        def _python_slicer_exe() -> str:
            exe = Path(sys.executable).resolve()
            if exe.name.lower() in ("pythonslicer.exe", "pythonslicer") and exe.is_file():
                return str(exe)
            candidates = [exe.parent]
            try:
                app_dir = Path(slicer.app.applicationDirPath())
            except Exception:
                try:
                    app_dir = Path(slicer.app.applicationDirPath)
                except Exception:
                    app_dir = None
            if app_dir:
                candidates.append(app_dir)
            for directory in candidates:
                for name in ("PythonSlicer.exe", "PythonSlicer"):
                    candidate = directory / name
                    if candidate.is_file():
                        return str(candidate)
            raise FileNotFoundError("PythonSlicer launcher not found (cannot run nnU-Net inference).")

        try:
            import time
            import traceback
            import json

            model_dir = str(Path(model_dir).resolve())
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

            python_slicer = _python_slicer_exe()

            if not str(inputNiftiPath).lower().endswith(".nii.gz"):
                raise ValueError(f"Input file must be .nii.gz: {inputNiftiPath}")
            if not str(outputNiftiPath).lower().endswith(".nii.gz"):
                raise ValueError(f"Output file must be .nii.gz: {outputNiftiPath}")

            inputNiftiPath = os.path.abspath(str(inputNiftiPath))
            finalOutputNiftiPath = os.path.abspath(str(outputNiftiPath))
            os.makedirs(os.path.dirname(finalOutputNiftiPath), exist_ok=True)

            if not overwrite and os.path.exists(finalOutputNiftiPath):
                return True

            with tempfile.TemporaryDirectory(prefix="TAPSeg_nnUNetOut_") as tempOutDir:
                tmpOutputNiftiPath = os.path.join(tempOutDir, os.path.basename(finalOutputNiftiPath))

                code = "\n".join(
                    [
                        "import os, time, json",
                        "os.environ.setdefault('nnUNet_n_proc_DA', '12')",
                        "import torch",
                        "from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor",
                        f"model_dir = {json.dumps(model_dir)}",
                        f"in_file = {json.dumps(inputNiftiPath)}",
                        f"out_file = {json.dumps(tmpOutputNiftiPath)}",
                        f"folds = {tuple(folds)!r}",
                        f"checkpoint = {json.dumps(str(checkpoint))}",
                        f"device_opt = {json.dumps(str(device))}",
                        f"cuda_id = {int(cuda_id)}",
                        f"perform_on_device = {bool(perform_on_device)}",
                        f"num_proc_pre = {int(num_proc_pre)}",
                        f"num_proc_export = {int(num_proc_export)}",
                        f"overwrite = {bool(overwrite)}",
                        f"verbose = {bool(verbose)}",
                        "",
                        "if device_opt == 'cuda':",
                        "    dev = torch.device('cuda', cuda_id)",
                        "elif device_opt == 'cpu':",
                        "    dev = torch.device('cpu')",
                        "else:",
                        "    dev = torch.device('cuda', cuda_id) if torch.cuda.is_available() else torch.device('cpu')",
                        "",
                        "predictor = nnUNetPredictor(",
                        "    tile_step_size=1.0,",
                        "    use_gaussian=True,",
                        "    use_mirroring=False,",
                        "    perform_everything_on_device=bool(perform_on_device) and dev.type == 'cuda',",
                        "    device=dev,",
                        "    verbose=bool(verbose),",
                        "    verbose_preprocessing=bool(verbose),",
                        "    allow_tqdm=False,",
                        ")",
                        "predictor.initialize_from_trained_model_folder(",
                        "    model_training_output_dir=model_dir,",
                        "    use_folds=tuple(folds),",
                        "    checkpoint_name=str(checkpoint),",
                        ")",
                        "file_ending = predictor.dataset_json.get('file_ending', '.nii.gz')",
                        "if not in_file.lower().endswith(file_ending.lower()):",
                        "    raise ValueError(f'Input suffix mismatch: expected {file_ending}, got {in_file}')",
                        "if not out_file.lower().endswith(file_ending.lower()):",
                        "    raise ValueError(f'Output suffix mismatch: expected {file_ending}, got {out_file}')",
                        "out_trunc = [out_file[:-len(file_ending)]]",
                        "predictor.predict_from_files(",
                        "    list_of_lists_or_source_folder=[[in_file]],",
                        "    output_folder_or_list_of_truncated_output_files=out_trunc,",
                        "    save_probabilities=False,",
                        "    overwrite=bool(overwrite),",
                        "    num_processes_preprocessing=int(num_proc_pre),",
                        "    num_processes_segmentation_export=int(num_proc_export),",
                        "    folder_with_segs_from_prev_stage=None,",
                        "    num_parts=1,",
                        "    part_id=0,",
                        ")",
                        "",
                        "deadline = time.time() + 1800",
                        "last_size = -1",
                        "stable_hits = 0",
                        "while time.time() < deadline:",
                        "    if os.path.exists(out_file):",
                        "        sz = os.path.getsize(out_file)",
                        "        if sz > 0 and sz == last_size:",
                        "            stable_hits += 1",
                        "        else:",
                        "            stable_hits = 0",
                        "        last_size = sz",
                        "        if sz > 0 and stable_hits >= 2:",
                        "            break",
                        "    time.sleep(0.5)",
                        "if not os.path.exists(out_file):",
                        "    raise RuntimeError('Inference finished but output file was not found: ' + out_file)",
                    ]
                )

                env = os.environ.copy()
                env.setdefault("nnUNet_n_proc_DA", "12")

                result = subprocess.run(
                    [python_slicer, "-c", code],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        "nnU-Net subprocess inference failed.\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}\n"
                    )

                if not os.path.exists(tmpOutputNiftiPath):
                    raise RuntimeError(f"nnU-Net did not generate output file: {tmpOutputNiftiPath}")

                shutil.copyfile(tmpOutputNiftiPath, finalOutputNiftiPath)

            return os.path.exists(finalOutputNiftiPath)
        except Exception as e:
            try:
                print(traceback.format_exc(), flush=True)
            except Exception:
                pass
            slicer.util.errorDisplay(f"nnU-Net inference failed: {e}")
            return False

    def processPulp(self, input_volume: vtkMRMLScalarVolumeNode, callback=None) -> bool:
        "Run pulp inference, assign instances to teeth, and load the segmentation."
        if not input_volume:
            slicer.util.errorDisplay("Please select an input volume.")
            return False

        print("[TAPSeg] Pulp segmentation started.", flush=True)

        tooth_seg_name = f"{input_volume.GetName()}_TAPSeg_tooth"
        tooth_seg_node = self._get_first_node_by_name(tooth_seg_name, "vtkMRMLSegmentationNode")
        if tooth_seg_node is None:
            tooth_seg_node = self._find_first_node_endswith("_TAPSeg_tooth", "vtkMRMLSegmentationNode")
        if tooth_seg_node is None:
            slicer.util.errorDisplay(
                "Tooth segmentation (Segmentation node) was not found. Please click 'Segment Teeth' first.\n"
                f"Expected node name: {tooth_seg_name}"
            )
            return False

        model_dir = self.getPulpModelDir()
        with tempfile.TemporaryDirectory(prefix="TAPSeg_Pulp_") as tempDir:
            in_file = os.path.join(tempDir, "input.nii.gz")
            out_file = os.path.join(tempDir, "input.nii.gz")

            self.slicerVolumeToNifti(input_volume, in_file)
            if not os.path.exists(in_file):
                raise RuntimeError("Failed to export NIfTI: input file was not created.")

            ok = self.nnunetInference(in_file, out_file, model_dir)
            if not ok or not os.path.exists(out_file):
                raise RuntimeError("nnU-Net inference failed: output file was not created.")

            out_img = sitk.ReadImage(out_file)
            assigned_img = self._assignPulpToTeeth(
                tooth_seg_node=tooth_seg_node,
                pulp_label_img=out_img,
                reference_volume=input_volume,
                callback=callback,
            )

            seg_name = f"{input_volume.GetName()}_TAPSeg_pulp"
            segNode = self._labelImageToSegmentationNode(
                assigned_img,
                reference_volume=input_volume,
                seg_node_name=seg_name,
            )

            self._removeNodeByNameIfExists(f"{input_volume.GetName()}_TAPSeg_pulp_label")

            try:
                slicer.util.selectModule("SegmentEditor")
                segEditorWidget = slicer.util.findChild(slicer.util.mainWindow(), "SegmentEditorWidget")
                if segEditorWidget:
                    segEditorWidget.setMasterVolumeNode(input_volume)
                    segEditorWidget.setSegmentationNode(segNode)
            except Exception:
                pass

        print("[TAPSeg] Pulp segmentation finished.", flush=True)
        return True

    @staticmethod
    def _pulp_assign_min_overlap_ratio() -> float:
        """
        Minimum overlap ratio threshold for assigning a pulp connected component to a tooth.
        A pulp component is discarded if max(|C∩M_tooth|)/|C| is below this value.

        Override via environment variable:
          TAPSEG_PULP_ASSIGN_MIN_OVERLAP_RATIO=0.01
        """
        v = os.environ.get("TAPSEG_PULP_ASSIGN_MIN_OVERLAP_RATIO", "").strip()
        if not v:
            return 0.01
        try:
            return max(0.0, float(v))
        except Exception:
            return 0.01

    def _toothSegmentationToLabelArray(self, *, tooth_seg_node, reference_volume: vtkMRMLScalarVolumeNode) -> np.ndarray:
        """
        Convert a tooth Segmentation node to a uint16 label array on the reference_volume grid.
        Segment names are expected to be numeric FDI codes (e.g., \"11\"). Voxels are labeled with that integer.
        """
        seg = tooth_seg_node.GetSegmentation()
        ids = vtk.vtkStringArray()
        seg.GetSegmentIDs(ids)

        ref_arr = slicer.util.arrayFromVolume(reference_volume)
        out = np.zeros_like(ref_arr, dtype=np.uint16)

        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            s = seg.GetSegment(sid)
            if s is None:
                continue
            name = str(s.GetName() or "").strip()
            try:
                fdi = int(name)
            except Exception:
                continue
            if fdi <= 0:
                continue
            m = slicer.util.arrayFromSegmentBinaryLabelmap(tooth_seg_node, sid, reference_volume) != 0
            out[m] = np.uint16(fdi)

        return out

    def _assignPulpComponentsToToothLabels(
        self,
        *,
        tooth_labels: np.ndarray,
        pulp_binary: np.ndarray,
        reference_sitk_image: sitk.Image,
    ) -> sitk.Image:
        """
        Core assignment implementation (numpy arrays):
        - tooth_labels: uint16 array (k,j,i) with tooth FDI labels (0=background)
        - pulp_binary: bool array (k,j,i) with pulp mask

        Output: sitk.Image (uint16) where kept pulp components are relabeled as (FDI + 100).
        One component per tooth is kept (highest overlap ratio; ties by overlap count).
        """
        tooth_labels = np.asarray(tooth_labels)
        pulp_binary = np.asarray(pulp_binary).astype(bool, copy=False)

        if pulp_binary.size == 0 or not bool(np.any(pulp_binary)):
            out = sitk.Image(reference_sitk_image.GetSize(), sitk.sitkUInt16)
            out.CopyInformation(reference_sitk_image)
            return out

        min_r = float(self._pulp_assign_min_overlap_ratio())

        pulp_bin_img = sitk.GetImageFromArray(pulp_binary.astype(np.uint8, copy=False))
        pulp_bin_img.CopyInformation(reference_sitk_image)
        cc_filter = sitk.ConnectedComponentImageFilter()
        try:
            cc_filter.SetFullyConnected(True)
        except Exception:
            pass
        cc = cc_filter.Execute(pulp_bin_img)
        cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
        cc_arr = sitk.GetArrayFromImage(cc).astype(np.int32, copy=False)

        num_cc = int(cc_arr.max()) if cc_arr.size else 0
        if num_cc <= 0:
            out = sitk.Image(reference_sitk_image.GetSize(), sitk.sitkUInt16)
            out.CopyInformation(reference_sitk_image)
            return out

        best_for_tooth = {}
        for comp_id in range(1, num_cc + 1):
            m = (cc_arr == comp_id)
            comp_size = int(np.count_nonzero(m))
            if comp_size <= 0:
                continue

            labels_in = tooth_labels[m]
            labels_in = labels_in[labels_in > 0]
            if labels_in.size == 0:
                continue

            counts = np.bincount(labels_in.astype(np.int64, copy=False))
            if counts.size <= 1:
                continue
            tooth_label = int(np.argmax(counts))
            inter = int(counts[tooth_label])
            if tooth_label <= 0 or inter <= 0:
                continue

            r = float(inter) / float(comp_size)
            if r < min_r:
                continue

            prev = best_for_tooth.get(tooth_label)
            cand = (inter, r, comp_size, comp_id)
            if prev is None:
                best_for_tooth[tooth_label] = cand
            else:
                if (cand[0] > prev[0]) or (cand[0] == prev[0] and cand[1] > prev[1]) or (
                    cand[0] == prev[0] and cand[1] == prev[1] and cand[2] > prev[2]
                ):
                    best_for_tooth[tooth_label] = cand

        out_arr = np.zeros_like(tooth_labels, dtype=np.uint16)
        for tooth_label, (_inter, _r, _sz, comp_id) in best_for_tooth.items():
            new_label = int(tooth_label) + 100
            out_arr[cc_arr == int(comp_id)] = np.uint16(new_label)

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(reference_sitk_image)
        return out_img

    def _assignPulpToTeeth(
        self,
        *,
        tooth_seg_node,
        pulp_label_img: sitk.Image,
        reference_volume: vtkMRMLScalarVolumeNode,
        callback=None,
    ) -> sitk.Image:
        """
        Instance-Level Pulp Assignment (UI path):
        - Requires tooth segmentation in the current scene.
        - Converts tooth segments to a label array (FDI values).
        - Splits pulp prediction into 3D connected components.
        - Assigns each component to the tooth with maximum overlap ratio R.
        - Keeps only one component per tooth and relabels it as (FDI + 100).
        """
        if callback:
            callback(0.1, "Post-processing...")

        ref_img = sitkUtils.PullVolumeFromSlicer(reference_volume)
        pulp_lab = self._as_uint16_label(pulp_label_img)
        if not self._same_sitk_geometry(pulp_lab, ref_img):
            pulp_lab = sitk.Resample(
                pulp_lab,
                ref_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt16,
            )

        pulp_arr = sitk.GetArrayFromImage(pulp_lab)
        pulp_bin = (pulp_arr != 0)
        tooth_labels = self._toothSegmentationToLabelArray(tooth_seg_node=tooth_seg_node, reference_volume=reference_volume)
        return self._assignPulpComponentsToToothLabels(
            tooth_labels=tooth_labels,
            pulp_binary=pulp_bin,
            reference_sitk_image=ref_img,
        )

    @staticmethod
    def _as_uint16_label(img: sitk.Image) -> sitk.Image:
        "Convert an image to uint16 labels, rounding floating-point values."
        pid = int(img.GetPixelIDValue())
        if pid in (
            sitk.sitkUInt8,
            sitk.sitkUInt16,
            sitk.sitkUInt32,
            sitk.sitkUInt64,
            sitk.sitkInt8,
            sitk.sitkInt16,
            sitk.sitkInt32,
            sitk.sitkInt64,
        ):
            return sitk.Cast(img, sitk.sitkUInt16)
        return sitk.Cast(sitk.Round(img), sitk.sitkUInt16)

    @staticmethod
    def _sitk_has_any_voxel(mask: sitk.Image) -> bool:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(mask)
        return float(stats.GetMaximum()) > 0.0

    @staticmethod
    def _same_sitk_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
        if tuple(a.GetSize()) != tuple(b.GetSize()):
            return False

        def _allclose(t1, t2) -> bool:
            if len(t1) != len(t2):
                return False
            for x, y in zip(t1, t2):
                if abs(float(x) - float(y)) > float(tol):
                    return False
            return True

        return _allclose(a.GetSpacing(), b.GetSpacing()) and _allclose(a.GetOrigin(), b.GetOrigin()) and _allclose(
            a.GetDirection(), b.GetDirection()
        )

    @staticmethod
    def _get_first_node_by_name(name: str, class_name=None):
        "Return the named node or None, optionally checking its class."
        try:
            node = slicer.util.getNode(name)
        except Exception:
            node = None
        if node is None or not class_name:
            return node
        try:
            if node.IsA(class_name):
                return node
        except Exception:
            pass
        return None

    @staticmethod
    def _find_first_node_endswith(suffix: str, class_name: str):
        "Find a node by name suffix and class."
        try:
            nodes = slicer.util.getNodesByClass(class_name)
        except Exception:
            return None
        for n in nodes:
            try:
                if str(n.GetName()).endswith(str(suffix)):
                    return n
            except Exception:
                continue
        return None

    def _removeNodeByNameIfExists(self, name: str) -> None:
        node = self._get_first_node_by_name(name)
        if node is None:
            return
        try:
            slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass

    def _labelImageToSegmentationNode(
        self,
        label_img: sitk.Image,
        *,
        reference_volume: vtkMRMLScalarVolumeNode,
        seg_node_name: str,
        single_segment_name=None,
    ):
        "Create one segment per nonzero label, preserving FDI names and reference geometry."
        if label_img is None:
            raise ValueError("label_img must not be None")
        if not reference_volume:
            raise ValueError("reference_volume must not be None")
        if not seg_node_name:
            raise ValueError("seg_node_name must not be empty")

        label_lab = self._as_uint16_label(label_img)
        ref_img = sitkUtils.PullVolumeFromSlicer(reference_volume)
        if not self._same_sitk_geometry(label_lab, ref_img):
            label_lab = sitk.Resample(
                label_lab,
                ref_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt16,
            )

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(label_lab)
        labels = [int(l) for l in stats.GetLabels() if int(l) != 0]
        labels.sort()

        segNode = self._get_first_node_by_name(seg_node_name, "vtkMRMLSegmentationNode")
        if segNode is None:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", seg_node_name)
            segNode.CreateDefaultDisplayNodes()
        else:
            try:
                segNode.SetName(seg_node_name)
            except Exception:
                pass

        segNode.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume)
        try:
            segNode.GetSegmentation().RemoveAllSegments()
        except Exception:
            pass

        disp = segNode.GetDisplayNode()
        if disp is not None:
            try:
                disp.SetVisibility(True)
                disp.SetVisibility2D(True)
                disp.SetOpacity2DFill(0.5)
                disp.SetOpacity2DOutline(1.0)
            except Exception:
                pass

        if not labels:
            return segNode

        label_arr = sitk.GetArrayFromImage(label_lab)
        seg = segNode.GetSegmentation()

        import colorsys

        for lab in labels:
            mask = (label_arr == int(lab))
            if not bool(np.any(mask)):
                continue

            seg_id = seg.AddEmptySegment()
            segment = seg.GetSegment(seg_id)

            if single_segment_name and len(labels) == 1 and int(lab) == 1:
                seg_name = str(single_segment_name)
            else:
                seg_name = str(int(lab))
            try:
                segment.SetName(seg_name)
            except Exception:
                pass

            try:
                h = (float(int(lab)) * 0.618033988749895) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
                segment.SetColor(float(r), float(g), float(b))
            except Exception:
                pass

            slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), segNode, seg_id, reference_volume)

        return segNode

    def generateToothHardTissueMask(self, input_volume: vtkMRMLScalarVolumeNode, callback=None) -> bool:
        "Subtract the pulp union from each tooth segment while retaining its name."
        if not input_volume:
            slicer.util.errorDisplay("Please select an input volume.")
            return False

        tooth_seg_name = f"{input_volume.GetName()}_TAPSeg_tooth"
        pulp_seg_name = f"{input_volume.GetName()}_TAPSeg_pulp"

        tooth_seg_node = self._get_first_node_by_name(tooth_seg_name, "vtkMRMLSegmentationNode")
        if tooth_seg_node is None:
            tooth_seg_node = self._find_first_node_endswith("_TAPSeg_tooth", "vtkMRMLSegmentationNode")

        pulp_seg_node = self._get_first_node_by_name(pulp_seg_name, "vtkMRMLSegmentationNode")
        if pulp_seg_node is None:
            pulp_seg_node = self._find_first_node_endswith("_TAPSeg_pulp", "vtkMRMLSegmentationNode")

        if tooth_seg_node is None:
            slicer.util.errorDisplay(
                "Tooth segmentation (Segmentation node) was not found. Please click 'Segment Teeth' first.\n"
                f"Expected node name: {tooth_seg_name}"
            )
            return False
        if pulp_seg_node is None:
            slicer.util.errorDisplay(
                "Pulp segmentation (Segmentation node) was not found. Please click 'Segment Pulp' first.\n"
                f"Expected node name: {pulp_seg_name}"
            )
            return False

        print("[TAPSeg] Hard tissue mask generation started.", flush=True)

        if callback:
            callback(0.0, "Generating hard tissue: reading pulp segmentation...")

        pulp_seg = pulp_seg_node.GetSegmentation()
        pulp_ids = vtk.vtkStringArray()
        pulp_seg.GetSegmentIDs(pulp_ids)

        pulp_union = None
        for i in range(pulp_ids.GetNumberOfValues()):
            sid = pulp_ids.GetValue(i)
            arr = slicer.util.arrayFromSegmentBinaryLabelmap(pulp_seg_node, sid, input_volume)
            m = (arr != 0)
            pulp_union = m if pulp_union is None else (pulp_union | m)

        if pulp_union is None:
            pulp_union = np.zeros_like(slicer.util.arrayFromVolume(input_volume), dtype=bool)

        if callback:
            callback(0.3, "Generating hard tissue: subtracting pulp from each tooth segment...")

        out_seg_name = f"{input_volume.GetName()}_TAPSeg_tooth_hard"
        out_seg_node = self._get_first_node_by_name(out_seg_name, "vtkMRMLSegmentationNode")
        if out_seg_node is None:
            out_seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", out_seg_name)
            out_seg_node.CreateDefaultDisplayNodes()
        else:
            try:
                out_seg_node.GetSegmentation().RemoveAllSegments()
            except Exception:
                pass

        out_seg_node.SetReferenceImageGeometryParameterFromVolumeNode(input_volume)
        out_seg = out_seg_node.GetSegmentation()

        tooth_seg = tooth_seg_node.GetSegmentation()
        tooth_ids = vtk.vtkStringArray()
        tooth_seg.GetSegmentIDs(tooth_ids)

        kept = 0
        for i in range(tooth_ids.GetNumberOfValues()):
            sid = tooth_ids.GetValue(i)
            tooth_mask = slicer.util.arrayFromSegmentBinaryLabelmap(tooth_seg_node, sid, input_volume) != 0
            hard_mask = tooth_mask & (~pulp_union)
            if not bool(np.any(hard_mask)):
                continue

            new_id = out_seg.AddEmptySegment()
            new_seg = out_seg.GetSegment(new_id)
            src_seg = tooth_seg.GetSegment(sid)
            try:
                new_seg.SetName(src_seg.GetName())
            except Exception:
                pass
            try:
                c = [0.8, 0.8, 0.0]
                src_seg.GetColor(c)
                new_seg.SetColor(float(c[0]), float(c[1]), float(c[2]))
            except Exception:
                pass

            slicer.util.updateSegmentBinaryLabelmapFromArray(hard_mask.astype(np.uint8), out_seg_node, new_id, input_volume)
            kept += 1

        if kept == 0:
            slicer.util.warningDisplay(
                "Hard tissue segmentation is empty (pulp may cover all tooth regions, or input segmentation is empty)."
            )

        self._removeNodeByNameIfExists(f"{input_volume.GetName()}_TAPSeg_tooth_hard_label")

        try:
            slicer.util.selectModule("SegmentEditor")
            segEditorWidget = slicer.util.findChild(slicer.util.mainWindow(), "SegmentEditorWidget")
            if segEditorWidget:
                segEditorWidget.setMasterVolumeNode(input_volume)
                segEditorWidget.setSegmentationNode(out_seg_node)
        except Exception:
            pass

        if callback:
            callback(1.0, "Done")

        print("[TAPSeg] Hard tissue mask generation finished.", flush=True)
        return True
