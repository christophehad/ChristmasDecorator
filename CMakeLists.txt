CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(ChristmasDecorator)

ADD_EXECUTABLE(decorate decorate.h decorate.cpp corners.cpp corners.h windows.cpp windows.h ChristmasDecorator.h align.cpp align.h)
TARGET_LINK_LIBRARIES(decorate ${OpenCV_LIBS})