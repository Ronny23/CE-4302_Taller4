LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := dot_product

LOCAL_SRC_FILES := dot_product.c

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    LOCAL_ARM_NEON := true
    LOCAL_CFLAGS :=  -O3 -DHAVE_NEON=1 -fopenmp -I./
endif

LOCAL_C_INCLUDES :=

LOCAL_STATIC_LIBRARIES := cpufeatures

LOCAL_LDLIBS := -llog -fopenmp

include $(BUILD_EXECUTABLE)

$(call import-module,android/cpufeatures)
