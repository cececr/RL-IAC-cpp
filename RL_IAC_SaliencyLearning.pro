#-------------------------------------------------
#
# Project created by QtCreator 2015-12-08T14:54:27
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RL_IAC_SaliencyLearning
TEMPLATE = app

#CONFIG += c++11
CONFIG += WITH_GPU

SOURCES += main.cpp \
    learning_module/Onlinerandomforest.cpp \
    learning_module/MyRandomForest.cpp \
    learning_module/LearningM.cpp \
    learning_module/dataformater.cpp \
    learning_module/classifier.cpp \
    foveal_segmentation/pt_cld_segmentation.cpp \
    foveal_segmentation/floortracker.cpp \
    feature_extractor/seeds2.cpp \
    feature_extractor/FeatureExtractor.cpp \
    bottom_up/VOCUS2.cpp \
    bottom_up/saliencyDetectionRudinac.cpp \
    bottom_up/saliencyDetectionItti.cpp \
    bottom_up/saliencyDetectionHou.cpp \
    bottom_up/saliencyDetectionBMS.cpp \
    bottom_up/CvGabor.cpp \
    bottom_up/bottomupsaliencymethod.cpp \
    RL_IAC/RegionsM.cpp \
    RL_IAC/Plot.cpp \
    RL_IAC/MetaM.cpp \
    RL_IAC/ActionSelectionM.cpp \
    SaliencyFileParser.cpp \
    saliencylearningexperiment.cpp \
    environment.cpp \
    feature_extractor/saliencyFeatureItti.cpp \
    experimentevaluation.cpp \
    feature_extractor/CvGabor2.cpp \
    worldmapgraph.cpp \
    feature_extractor/deepfeatureextractor.cpp \
    foveal_segmentation/image_geometry/pinhole_camera_model.cpp

HEADERS  += \
    learning_module/precomp.hpp \
    learning_module/Onlinerandomforest.h \
    learning_module/MyRandomForest.h \
    learning_module/LearningM.h \
    learning_module/dataformater.h \
    learning_module/classifier.h \
    foveal_segmentation/voxel_grid_fix.hpp \
    foveal_segmentation/voxel_grid_fix.h \
    foveal_segmentation/pt_cld_segmentation.hpp \
    foveal_segmentation/pt_cld_segmentation.h \
    foveal_segmentation/floortracker.h \
    foveal_segmentation/crop_box_fix.hpp \
    foveal_segmentation/crop_box_fix.h \
    feature_extractor/seeds2.h \
    feature_extractor/FeatureExtractor.h \
    bottom_up/VOCUS2.h \
    bottom_up/saliencyDetectionRudinac.h \
    bottom_up/saliencyDetectionItti.h \
    bottom_up/saliencyDetectionHou.h \
    bottom_up/saliencyDetectionBMS.h \
    bottom_up/CvGabor.h \
    bottom_up/bottomupsaliencymethod.h \
    RL_IAC/RegionsM.h \
    RL_IAC/Plot.h \
    RL_IAC/MetaM.h \
    RL_IAC/ActionSelectionM.h \
    SaliencyFileParser.h \
    printdebug.h \
    Evaluation.h \
    saliencylearningexperiment.h \
    environment.h \
    feature_extractor/saliencyFeatureItti.h \
    experimentevaluation.h \
    feature_extractor/CvGabor2.h \
    worldmapgraph.h \
    feature_extractor/deepfeatureextractor.h \
    foveal_segmentation/image_geometry/CameraInfoLight.h \
    foveal_segmentation/image_geometry/pinhole_camera_model.h \
    common.h


# PCL related
CONFIG += link_pkgconfig
PKGCONFIG += pcl_common-1.8
PKGCONFIG += pcl_filters-1.8
PKGCONFIG += pcl_sample_consensus-1.8
PKGCONFIG += pcl_segmentation-1.8
PKGCONFIG += pcl_features-1.8

LIBS += -lpcl_segmentation
LIBS += -lpcl_features

INCLUDEPATH += /usr/local/include/pcl-1.8
INCLUDEPATH += /usr/include/eigen3/
LIBS += -lboost_system

# OpenCV related
PKGCONFIG += opencv
LIBS += -L/usr/local/lib
LIBS += -L/usr/local/share/OpenCV/3rdparty/lib/

# Boost related
LIBS += -lboost_serialization
LIBS += -lboost_system

# Caffe related
###################
#WITH_GPU {
    LIBS += -L/usr/local/cuda-8.0/targets/x86_64-linux/lib/
    LIBS += -lcuda -lcublas -lcurand -lcudart
    INCLUDEPATH += /usr/local/cuda-8.0/targets/x86_64-linux/include/

    INCLUDEPATH += /home/celine/Libs/caffe/include
    INCLUDEPATH += /home/celine/Libs/caffe/.build_release/src

    LIBS += -L/home/celine/Libs/caffe/.build_release/lib
    DEFINES -= CPU_ONLY
#}
#else
#{
#    DEFINES += CPU_ONLY
#}
##################
LIBS += -lcaffe
LIBS += -lglog -lprotobuf

OTHER_FILES += \
    params.dat

