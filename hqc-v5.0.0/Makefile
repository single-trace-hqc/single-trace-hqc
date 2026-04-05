SHELL := /bin/sh

# This Makefile keeps upstream sources untouched by building from:
#   hqc-v5.0.0/build/ref/patched/next-release
THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UPSTREAM_DIR := $(THIS_DIR)/next-release
BUILD_DIR := $(THIS_DIR)/build/ref
PATCHED_DIR := $(BUILD_DIR)/patched/next-release
PATCHED_STAMP := $(PATCHED_DIR)/.from_upstream
PATCHES_DIR := $(THIS_DIR)/patches
OBJ_DIR := $(BUILD_DIR)/obj

# Select HQC instance: 1, 3, or 5
HQC_INSTANCE ?= 1
HQC_VARIANT := hqc-$(HQC_INSTANCE)

CC ?= cc
UNAME_S := $(shell uname -s)
HOST_ARCH := $(shell uname -m)

# Build shared lib.
LIB_EXT ?= so
LIB_NAME := libhqc-$(HQC_INSTANCE).$(LIB_EXT)
LIB_PATH := $(BUILD_DIR)/$(LIB_NAME)

CFLAGS ?= -O3 -std=c99 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS ?=
LDLIBS ?=

ifeq ($(UNAME_S),Darwin)
  SHARED_FLAG ?= -dynamiclib
  TARGET_ARCH ?= $(HOST_ARCH)
  ARCH_FLAGS ?= -arch $(TARGET_ARCH)
  EFFECTIVE_ARCH := $(TARGET_ARCH)
else
  SHARED_FLAG ?= -shared
  TARGET_ARCH ?= $(HOST_ARCH)
  ARCH_FLAGS ?=
  EFFECTIVE_ARCH := $(HOST_ARCH)
endif

COMMON_PATCHES := $(PATCHES_DIR)/decodeOneRMBlock.patch
ARM64_PATCH := $(PATCHES_DIR)/ref_arm64_no_immintrin.patch
PATCHES := $(COMMON_PATCHES) $(ARM64_PATCH)

PATCHED_PATCHES_STAMP := $(PATCHED_DIR)/.patches_applied_$(EFFECTIVE_ARCH)

COMMON_SRC := code.c crypto_memset.c fft.c kem.c symmetric.c
REF_SRC := vector.c reed_muller.c reed_solomon.c gf.c gf2x.c parsing.c hqc.c
FIPS_SRC := fips202.c

COMMON_OBJ := $(addprefix $(OBJ_DIR)/common/,$(COMMON_SRC:.c=.o))
REF_OBJ := $(addprefix $(OBJ_DIR)/ref/,$(REF_SRC:.c=.o))
FIPS_OBJ := $(addprefix $(OBJ_DIR)/lib/fips202/,$(FIPS_SRC:.c=.o))
OBJS := $(COMMON_OBJ) $(REF_OBJ) $(FIPS_OBJ)

INCLUDES := \
	-I$(PATCHED_DIR)/src/common \
	-I$(PATCHED_DIR)/src/common/$(HQC_VARIANT) \
	-I$(PATCHED_DIR)/src/ref \
	-I$(PATCHED_DIR)/src/ref/$(HQC_VARIANT) \
	-I$(PATCHED_DIR)/lib/fips202

.PHONY: all clean

all: $(PATCHED_PATCHES_STAMP)
	@$(MAKE) "$(LIB_PATH)"

$(PATCHED_STAMP):
	@mkdir -p "$(BUILD_DIR)/patched"
	@rsync -a --delete --exclude '.git' "$(UPSTREAM_DIR)/" "$(PATCHED_DIR)/"
	@touch "$@"

$(PATCHED_PATCHES_STAMP): $(PATCHED_STAMP) $(PATCHES)
	@set -e; \
	for p in $(PATCHES); do \
		echo "Applying patch $$(basename "$$p")"; \
		(cd "$(PATCHED_DIR)" && patch -p1 < "$$p"); \
	done
	@touch "$@"

$(LIB_PATH): $(OBJS)
	@mkdir -p "$(BUILD_DIR)"
	$(CC) $(ARCH_FLAGS) $(SHARED_FLAG) $(LDFLAGS) -o "$@" $(OBJS) $(LDLIBS)
	@echo "Built $@"

$(OBJ_DIR)/common/%.o: $(PATCHED_DIR)/src/common/%.c
	@mkdir -p "$(@D)"
	$(CC) $(ARCH_FLAGS) $(CFLAGS) $(INCLUDES) -c "$<" -o "$@"

$(OBJ_DIR)/ref/%.o: $(PATCHED_DIR)/src/ref/%.c
	@mkdir -p "$(@D)"
	$(CC) $(ARCH_FLAGS) $(CFLAGS) $(INCLUDES) -c "$<" -o "$@"

$(OBJ_DIR)/lib/fips202/%.o: $(PATCHED_DIR)/lib/fips202/%.c
	@mkdir -p "$(@D)"
	$(CC) $(ARCH_FLAGS) $(CFLAGS) $(INCLUDES) -c "$<" -o "$@"

clean:
	rm -rf "$(BUILD_DIR)"
