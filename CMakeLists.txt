# Root CMakeLists.txt - voice-assistant-idf
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(voice_assistant)

# Genereaza si flasheaza automat imaginea SPIFFS cu sample-urile TTS
# (folderul tts_output/ din radacina proiectului -> partitia "storage")
spiffs_create_partition_image(storage tts_output FLASH_IN_PROJECT)