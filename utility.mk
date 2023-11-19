make := $(MAKE)
make_sub_dir_flag := -C

empty :=
space := $(empty) $(empty)

windows_os := Windows_NT

ifeq ($(OS),$(windows_os))
    console := powershell --comand
    # no equivalent for linux
    cmd := cmd /c
    copy := copy
    copy_flag := /Y
    rm := rmdir
    rm_file := del
    rm_flag := /Q /S
    # don't forget we will need to swap 
    ln := mklink
    # ln_flag := /d
    ln_flag := /j
    dylib_extention := dll
    exec_extension := exe
else
    console := sh
    copy := cp
    copy_flag := -f
    rm := rm
    rm_file := $(rm)
    rm_flag := -rf
    ln := ln
    ln_flag := -s -r -d
    dylib_extention := so
    exec_extension := x86_64s
endif

mkdir := mkdir

# function to use windows "back slash" path notation
windows_path = $(subst /,\,$(1))
canonicalize_path = $(if $(findstring $(windows_os),$(OS)),$(call windows_path,$(1)),$(1))

# force the revaluation of the rule even if other prerequesit files are not changed
.PHONY: .FORCE
.FORCE: