#-------------------------------------------------
#
# Project created by QtCreator 2015-05-23T12:40:20
#
#-------------------------------------------------

QT       = core
QT       += gui

TARGET = RC1
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

INCLUDEPATH += /usr/local/lib/

LIBS += `pkg-config opencv --libs`
LIBS += -lraspicam_cv
LIBS += -lraspicam

