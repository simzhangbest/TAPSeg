cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(TAPSeg)

set(EXTENSION_HOMEPAGE "https://github.com/simzhangbest/TAPSeg")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Y Zhang")
set(EXTENSION_DESCRIPTION "Tooth and pulp segmentation from CBCT volumes, with tooth hard-tissue mask generation. License: Apache-2.0.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/simzhangbest/TAPSeg/main/TAPSeg/Resources/Icons/TAPSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/simzhangbest/TAPSeg/main/TAPSeg/Resources/Screenshots/TAPSeg.png")
set(EXTENSION_STATUS "Experimental")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(TAPSeg)

install(
  FILES requirements.txt LICENSE.txt
  DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  COMPONENT RuntimeLibraries
  )

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
