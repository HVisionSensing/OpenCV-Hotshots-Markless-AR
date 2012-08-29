add_executable(markerless_ar_demo ARDrawingContext.cpp
ARDrawingContext.hpp
CameraCalibration.cpp
CameraCalibration.hpp
GeometryTypes.cpp
GeometryTypes.hpp
main.cpp
ARPipeline.hpp
ARPipeline.cpp
Pattern.cpp
Pattern.hpp
PatternDetector.cpp
PatternDetector.hpp
)

add_executable(record_video record_video.cpp)

target_link_libraries( markerless_ar_demo ${OpenCV_LIBS} )
target_link_libraries( markerless_ar_demo ${OpenCV_LIBRARIES} )
target_link_libraries( markerless_ar_demo ${OPENGL_LIBRARIES} )

target_link_libraries( record_video ${OpenCV_LIBS} )
target_link_libraries( record_video ${OpenCV_LIBRARIES} )
      
install (TARGETS markerless_ar_demo DESTINATION bin)
install (TARGETS record_video DESTINATION bin)
