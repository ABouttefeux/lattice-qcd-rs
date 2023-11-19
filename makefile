include utility.mk 

name := lattice_qcd_rs
lib_name := lib$(name).rlib

source_sufix := rs
source_files := $(foreach sufix,$(source_sufix),$(wildcard *.$(sufix) */*.$(sufix) */*/*.$(sufix) */*/*/*.$(sufix) */*/*/*/*.$(sufix) */*/*/*/*/*.$(sufix)))

# cargo
cargo := cargo
cargo_build := build
cargo_test := test
cargo_doc := doc
cargo_clean := clean
# work space
cargo_all_flag := --all
cargo_crate_macro := -p $(name)-procedural_macro

# clippy
cargo_clippy := clippy
cargo_clippy_flag := -- -D warnings

rust_release_flag := --release
# doc
rust_doc_flag_private_item := --document-private-items
rust_doc_flag_no_dep := --no-deps
# target
rust_all_targets = --all-targets
rust_example_flag := --examples
# feature
rust_coverage_feature := --no-default-features --features="serde-serialize"

rust_stable := +stable
rust_nightly := +nightly

ifeq ($(OS),$(windows_os))
    powershell := powershell
    null := $$null
    call_with_var = $(powershell) $$env:$(1)=$(2); $(3); $$env:$(1)=$(null)
else
    call_with_var = $(1)=$(2) $(3)
endif

git_hooks := $(foreach file, $(notdir $(wildcard tools/git_hook/*)), .git/hooks/$(file))


.PHONY: all
all: clippy


.PHONY: build
build: target/debug/$(lib_name)


.PHONY: test
test: $(source_files)
	$(cargo) $(cargo_test) $(cargo_all_flag) $(rust_all_targets)


.PHONY: clippy
clippy: $(source_files)
	$(cargo) $(rust_nightly) $(cargo_clippy) $(cargo_all_flag) $(rust_all_targets) $(cargo_clippy_flag)


.PHONY: doc_check
doc_check: $(source_files)
	$(cargo) $(cargo_doc) $(cargo_all_flag) $(rust_doc_flag_private_item) $(rust_doc_flag_no_dep)


.PHONY: doc
doc: $(source_files)
	$(cargo) $(cargo_doc) $(cargo_all_flag) $(rust_doc_flag_private_item) $(rust_doc_flag_no_dep)


.PHONY: clean
clean: .FORCE
	$(cargo) $(cargo_clean)


.PHONY: setup
setup: $(git_hooks)


.git/hooks/%: tools/git_hook/%
	- $(copy) $(call canonicalize_path,$<) $(call canonicalize_path,$@)


target/release/$(lib_name): $(source_files)
	$(cargo) $(cargo_build) $(rust_release_flag)


target/debug/$(lib_name): $(source_files)
	$(cargo) $(cargo_build)
