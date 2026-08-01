# plio_probe.tcl — xsim pre-simulation hook for probing a PL→AIE AXIS link.
#
# Sourced by the generated tb.tcl via USER_PRE_SIM_SCRIPT (see the
# `if { [info exists ::env(USER_PRE_SIM_SCRIPT)] }` block), i.e. BEFORE `run all`.
#
# Parameterised by environment variables so one script serves both designs:
#   PROBE_CU    PL kernel instance   (default roi_crop_0; smoke test: stream_src_0)
#   PROBE_PORT  AXIS port base name  (default patch_out;  smoke test: out_r)
#   PROBE_VCD   output file          (default plio_probe.vcd)
#
# Naming convention (verified in vitis_design.protoinst for both designs):
#   inside the CU     <PORT>_TVALID / <PORT>_TREADY   (upper case)
#   VitisRegion edge  <PORT>_tvalid / <PORT>_tready   (lower case)
#
# NOTE: ai_engine_0.S00_AXIS is a SystemC/TLM socket ("AXIS_SOCKET":
# "S00_AXIS_tlm_axis_socket"), so it has NO RTL wires. The VitisRegion boundary
# port is the last RTL point before the TLM adapter, and its TREADY is driven BY
# that adapter — so it is exactly the AIE-side backpressure signal.
#
# Reading the result:
#   TVALID=1, TREADY=0 forever -> AIE side never accepts
#   TVALID=0 forever           -> the PL kernel never produces; look at the
#                                 ap_*_blocking_n signals to see WHY:
#     ap_ext_blocking_n = 0 -> stalled on external memory (m_axi / DDR read)
#     ap_str_blocking_n = 0 -> stalled on a stream (the AXIS write)
#     ap_int_blocking_n = 0 -> stalled on intra-kernel dataflow
#   both toggle, count < N     -> mid-stream stall

proc envOr {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default
}

set CU   [envOr PROBE_CU   roi_crop_0]
set PORT [envOr PROBE_PORT patch_out]
set VCD  [envOr PROBE_VCD  plio_probe.vcd]

set BASE   /tb/DUT/vitis_design_wrapper_i/vitis_design_i
set CUPATH $BASE/VitisRegion/$CU

puts "=== plio_probe: CU=$CU PORT=$PORT VCD=$VCD ==="

# label -> path, relative to $BASE. Anything that does not resolve is reported
# and skipped, so a renamed signal degrades to a smaller capture, never a crash.
set probes [list \
    [list clk            VitisRegion/ap_clk] \
    [list rstn           VitisRegion/ap_rst_n] \
    \
    [list cu_tvalid      VitisRegion/$CU/${PORT}_TVALID] \
    [list cu_tready      VitisRegion/$CU/${PORT}_TREADY] \
    [list cu_tdata       VitisRegion/$CU/${PORT}_TDATA] \
    [list cu_tlast       VitisRegion/$CU/${PORT}_TLAST] \
    \
    [list aie_tvalid     VitisRegion/${PORT}_tvalid] \
    [list aie_tready     VitisRegion/${PORT}_tready] \
    \
    [list ap_start       VitisRegion/$CU/inst/ap_start] \
    [list ap_done        VitisRegion/$CU/inst/ap_done] \
    [list ap_idle        VitisRegion/$CU/inst/ap_idle] \
    [list ap_ready       VitisRegion/$CU/inst/ap_ready] \
    \
    [list blk_int        VitisRegion/$CU/inst/ap_int_blocking_n] \
    [list blk_ext        VitisRegion/$CU/inst/ap_ext_blocking_n] \
    [list blk_str        VitisRegion/$CU/inst/ap_str_blocking_n] \
    \
    [list m_arvalid      VitisRegion/$CU/m_axi_gmem0_ARVALID] \
    [list m_arready      VitisRegion/$CU/m_axi_gmem0_ARREADY] \
    [list m_rvalid       VitisRegion/$CU/m_axi_gmem0_RVALID] \
    [list m_rready       VitisRegion/$CU/m_axi_gmem0_RREADY] \
    [list m_rlast        VitisRegion/$CU/m_axi_gmem0_RLAST] \
]

set found {}
puts "=== plio_probe: resolving named signals ==="
foreach p $probes {
    set label [lindex $p 0]
    set path  $BASE/[lindex $p 1]
    if {[get_objects -quiet $path] ne {}} {
        lappend found $path
        puts [format "  OK      %-12s %s" $label $path]
    } else {
        puts [format "  MISSING %-12s %s" $label $path]
    }
}

# ---------------------------------------------------------------------------
# Discover the kernel's pipelined sub-loops dynamically.
#
# Vitis names these grp_<kernel>_Pipeline_<LOOP>_fu_<N>, and the _fu_<N> suffix
# is assigned by HLS — it CHANGES between builds. Hardcoding it would silently
# stop matching after any rebuild, so sweep for it instead. For roi_crop these
# are the Stage A passes (PASS1_ROW_PASS1_COL, NORM_LOOP, PASS2_ROW_PASS2_COL);
# their ap_start/ap_done tell us which pass the kernel is sitting in.
# ---------------------------------------------------------------------------
puts "=== plio_probe: sweeping for pipelined sub-loops ==="
set subscopes {}
if {[catch {set subscopes [get_scopes -quiet $CUPATH/inst/*]} msg]} {
    puts "  get_scopes unavailable ($msg) — skipping sub-loop sweep"
}
foreach s $subscopes {
    set nm [file tail $s]
    if {![string match "grp_*" $nm]} { continue }
    puts "  loop: $nm"
    foreach sig {ap_start ap_done ap_idle ap_int_blocking_n ap_ext_blocking_n ap_str_blocking_n} {
        set path $s/$sig
        if {[get_objects -quiet $path] ne {}} {
            lappend found $path
            puts [format "    OK    %s" $path]
        }
    }
}

puts "=== plio_probe: logging [llength $found] signals to $VCD ==="

# open_vcd/log_vcd require the snapshot to be elaborated with `xelab --debug
# typical` (or all). The Vitis-generated elaborate.sh uses `--debug off`, which
# makes log_vcd fail with
#   [Simulator 45-10] The current simulation was compiled without trace information
# and, because simulate.sh runs xsim with `-onerror quit`, that error KILLS the
# whole emulation before the design ever boots. Hence the catch: a probe that
# cannot arm must not take the run down with it.
# Use `make debug_sim` (or `make smoke_debug_sim`) to re-elaborate with debug.
if {[llength $found] > 0} {
    if {[catch {
        open_vcd $VCD
        log_vcd $found
    } msg]} {
        puts "plio_probe: WARNING — VCD logging unavailable: $msg"
        puts "plio_probe: re-elaborate with 'make debug_sim', then re-run."
    }
    # Also push them into the wave database so the .wdb is usable in the GUI.
    foreach s $found { catch {log_wave -quiet $s} }
} else {
    puts "plio_probe: WARNING — nothing resolved; check PROBE_CU/PROBE_PORT."
}

# The VCD is written incrementally, so even if the emulation hangs and gets
# killed the file is truncated-but-parseable. close_vcd only runs on a clean exit.
puts "=== plio_probe: armed, handing back to tb.tcl ==="
