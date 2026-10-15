set(MODULE_NAME TAPSeg)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  medic/__init__.py
  medic/relabel_smooth.py
  medic/tooth_backend_simzhang.py
  medic/simzhang/__init__.py
  medic/simzhang/roi_dect.py
  medic/simzhang/cnt_skl_dect.py
  medic/simzhang/ins_tooth_seg_v1.py
  medic/simzhang/networks/__init__.py
  medic/simzhang/networks/vnet_roi.py
  medic/simzhang/networks/vnet.py
  medic/simzhang/networks/vnet_ins_seg_bd.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/UI/loading.ui
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
