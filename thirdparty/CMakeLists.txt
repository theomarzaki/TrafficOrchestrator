include(FetchContent)

set(LIBTORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.0.1.zip)
set(LIBTORCH_URL_MD5 9a653df4acf3c82aa31e69e818cfaffb)

FetchContent_Declare(libtorch
        URL ${LIBTORCH_URL}
        URL_MD5 ${LIBTORCH_URL_MD5}
        SOURCE_DIR ${CMAKE_BINARY_DIR}/lib)

# TODO use FetchContent_MakeAvailable instead of FetchContent_Populate with cmmake 3.14
FetchContent_GetProperties(libtorch)
if (NOT libtorch_POPULATED)
    message("we're downloading the torch library: wait many minutes!")
    FetchContent_Populate(libtorch)
    message("the source is available at" ${libtorch_SOURCE_DIR})
endif ()