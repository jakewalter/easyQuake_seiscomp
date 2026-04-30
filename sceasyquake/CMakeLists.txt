cmake_minimum_required(VERSION 3.0)
project(sceasyquake)

# This project is intended to be deployed into a SeisComP installation
# Copy files to $SEISCOMP_ROOT/share/sceasyquake or use rsync

install(DIRECTORY bin/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/sceasyquake/bin)
install(DIRECTORY etc/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/sceasyquake/etc)
install(DIRECTORY lib/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/sceasyquake/lib)
