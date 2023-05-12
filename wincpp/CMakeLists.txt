  PROJECT("app1")
  
  if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
  endif()

  ADD_LIBRARY(m SHARED m.cpp)
  
  #ADD_EXECUTABLE(myexe main.cpp)
  #TARGET_LINK_LIBRARIES(myexe m)

