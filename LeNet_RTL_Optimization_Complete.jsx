import { useState, useCallback, useRef, useMemo } from "react";

// ═══════════════════════════════════════════════
// DESIGN SYSTEM
// ═══════════════════════════════════════════════
const C = {
  bg: "#060b18", s1: "#0c1225", s2: "#111a30", s3: "#182040",
  border: "#1c2744", borderH: "#2a3a5c",
  t1: "#e8ecf4", t2: "#a0aec0", t3: "#64748b",
  // Category colors
  c1: "#3b9eff", c2: "#a78bfa", c3: "#34d399", c4: "#fb923c",
  c5: "#f472b6", c6: "#fbbf24", c7: "#6ee7b7",
  red: "#ef4444", green: "#22c55e",
};
const mono = "'JetBrains Mono','Fira Code','SF Mono',monospace";
const sans = "'Inter','Segoe UI',system-ui,sans-serif";

// ═══════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════
const cats = [
  { id:1, t:"Parallelism", c:C.c1, ico:"⚡", d:"Do more independent computation in the same cycle.",
    items:[
      {n:"Spatial Parallelism",w:"Compute multiple spatial positions simultaneously by replicating convolution datapaths.",h:"Replicate sliding-window/MAC engines so several output pixels are produced per cycle. Each engine has its own window buffer and MAC array.",ex:"In Conv1 (28×28→24×24 output), instead of computing 1 output pixel per cycle (576 cycles), compute 4 adjacent pixels per cycle (144 cycles) using 4 parallel window engines.",rtl:"Instantiate N conv_pixel modules, each with its own 5×5 register window and 25 MACs. A shared line-buffer feeds all engines. Output valid signals are grouped.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↑↑",membw:"↑↑",simrt:"—",diff:"Medium",
        bef_lbl:"Sequential: 1 pixel/cycle",aft_lbl:"Parallel: 4 pixels/cycle",
        bef_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={20+i*110} y="10" width="90" height="30" rx="4" fill={i===0?c+"30":C.s1} stroke={i===0?c:C.border}/><text x={65+i*110} y="29" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Pixel {i}</text></g>)}<text x="240" y="60" textAnchor="middle" fill={C.t3} fontSize="9" fontFamily={mono}>4 cycles total</text></>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={20+i*110} y="10" width="90" height="30" rx="4" fill={c+"25"} stroke={c} strokeWidth="1.5"/><text x={65+i*110} y="29" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Pixel {i}</text></g>)}<text x="240" y="60" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono} fontWeight="bold">1 cycle — 4× speedup</text></>,
        cyc_svg:(c)=><><text x="10" y="16" fill={C.t3} fontSize="9" fontFamily={mono}>Before:</text>{[0,1,2,3].map(i=><rect key={i} x={80+i*55} y="5" width="50" height="18" rx="2" fill={i===0?c+"40":C.s2} stroke={C.border}/>)}<text x="10" y="46" fill={c} fontSize="9" fontFamily={mono}>After:</text><rect x="80" y="35" width="220" height="18" rx="2" fill={c+"30"} stroke={c}/><text x="190" y="48" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>All 4 pixels in 1 cycle</text></>,
      },
      {n:"Operator Parallelism",w:"Execute independent arithmetic operators concurrently instead of time-multiplexing a single unit.",h:"Instantiate multiple multipliers/MACs/adders. For a 5×5 convolution, use 25 multipliers simultaneously instead of reusing one 25 times.",ex:"Conv1 5×5 kernel: instantiate 25 multipliers forming all products in 1 cycle, then feed them into a balanced adder tree for reduction in ~5 levels.",rtl:"Replace single `result <= result + a[i]*b[i]` accumulation loop with 25 parallel `wire [15:0] prod[24:0]` and an adder tree module.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↑↑",membw:"↑",simrt:"—",diff:"Low",
        bef_lbl:"Single MAC reused 25×",aft_lbl:"25 parallel multipliers + tree",
        bef_svg:(c)=><><rect x="160" y="5" width="120" height="35" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>1 MAC × 25</text><path d="M220 40 Q220 55 220 45 Q230 55 220 60 Q210 55 220 45" stroke={C.t3} fill="none"/><text x="220" y="75" textAnchor="middle" fill={C.t3} fontSize="9" fontFamily={mono}>25 cycles</text></>,
        aft_svg:(c)=><>{[0,1,2,3,4].map(r=>[0,1,2,3,4].map(k=><rect key={`${r}${k}`} x={60+k*40} y={2+r*14} width="36" height="12" rx="2" fill={c+"18"} stroke={c} strokeWidth="0.5"/>))}<rect x="300" y="15" width="90" height="40" rx="4" fill={c+"25"} stroke={c}/><text x="345" y="39" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Σ Tree</text><text x="420" y="39" fill={c} fontSize="9" fontFamily={mono}>1 cyc!</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 25 cycles (sequential MAC)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: ~5 cycles (MUL‖ + log₂ reduce)</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>Speedup: ~5× for operator-bound paths</text></>,
      },
      {n:"Loop Unrolling",w:"Convert sequential loop iterations into replicated parallel hardware units.",h:"The RTL synthesis equivalent of software loop unrolling: each iteration becomes a dedicated hardware instance running concurrently.",ex:"A kernel loop iterating 25 times becomes 25 parallel MAC units. A filter loop iterating 6 times becomes 6 parallel filter engines.",rtl:"Replace `for(i=0;i<25;i++) acc<=acc+w[i]*x[i]` with generate-block instantiating 25 MAC units and a reduction tree.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↑↑",membw:"↑↑",simrt:"—",diff:"Low",
        bef_lbl:"Sequential loop: N iterations",aft_lbl:"N parallel hardware units",
        bef_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x="150" y={5+i*20} width="140" height="16" rx="3" fill={i===0?c+"20":C.s1} stroke={i===0?c:C.border}/><text x="220" y={16+i*20} textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>iter {i} (cycle {i+1})</text></g>)}</>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={40+i*110} y="10" width="95" height="28" rx="4" fill={c+"18"} stroke={c}/><text x={87+i*110} y="28" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>HW Unit {i}</text></g>)}<text x="240" y="55" textAnchor="middle" fill={c} fontSize="9" fontFamily={mono}>All execute in cycle 1</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: N cycles</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 1 cycle (fully unrolled) or N/k (partial)</text></>,
      },
      {n:"Filter Parallelism",w:"Compute multiple output feature maps concurrently by instantiating parallel filter engines.",h:"Instead of processing Conv1's 6 filters sequentially, instantiate 6 filter engines sharing the same input but using different weight sets.",ex:"Conv1: 6 parallel filter engines, each with 25 MACs, producing 6 output feature maps simultaneously. Input is broadcast; weights are per-filter.",rtl:"generate for(f=0;f<6;f++) begin: FILT conv_engine #(.FILT_ID(f)) u_conv(.clk,.in_pixel,.weights(w[f]),.out(fm[f])); end",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↑↑",membw:"↑",simrt:"—",diff:"Medium",
        bef_lbl:"6 filters sequential",aft_lbl:"6 filters parallel",
        bef_svg:(c)=><>{[0,1,2,3,4,5].map(i=><g key={i}><rect x={5+i*78} y="10" width="70" height="25" rx="3" fill={i===0?c+"20":C.s1} stroke={i===0?c:C.border}/><text x={40+i*78} y="26" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>F{i}</text></g>)}<text x="240" y="55" textAnchor="middle" fill={C.t3} fontSize="9" fontFamily={mono}>6 cycles (one at a time)</text></>,
        aft_svg:(c)=><>{[0,1,2,3,4,5].map(i=><g key={i}><rect x={5+i*78} y="10" width="70" height="25" rx="3" fill={c+"20"} stroke={c}/><text x={40+i*78} y="26" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>F{i}</text></g>)}<text x="240" y="55" textAnchor="middle" fill={c} fontSize="9" fontFamily={mono} fontWeight="bold">1 cycle — all 6 filters</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 6 × conv_time</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 1 × conv_time (6× speedup)</text></>,
      },
      {n:"Channel Parallelism",w:"Process multiple input channels concurrently and reduce their partial sums.",h:"For Conv2 with 6 input channels, compute all 6 channel contributions in parallel, then sum with a reduction tree.",ex:"Conv2: 6 input channels × 5×5 kernel = 150 MACs. Parallelize across channels: 6 engines of 25 MACs each, then reduce 6 partial sums.",rtl:"Each channel engine: 25 MACs + local reduction. A 6-input adder tree combines channel partial sums. Total: 150 multipliers + 2-level reduce.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↑↑",membw:"↑↑",simrt:"—",diff:"Medium",
        bef_lbl:"Channels processed sequentially",aft_lbl:"All channels in parallel + reduce",
        bef_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x="130" y={3+i*18} width="180" height="14" rx="3" fill={i===0?c+"20":C.s1} stroke={i===0?c:C.border}/><text x="220" y={13+i*18} textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>Ch{i}: 25 MACs → partial sum</text></g>)}</>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={10+i*100} y="3" width="90" height="22" rx="3" fill={c+"18"} stroke={c}/><text x={55+i*100} y="17" textAnchor="middle" fill={C.t1} fontSize="7" fontFamily={mono}>Ch{i}×W{i}</text><line x1={55+i*100} y1="25" x2={200} y2="40" stroke={c} strokeWidth="0.5"/></g>)}<rect x="150" y="40" width="100" height="20" rx="4" fill={c+"30"} stroke={c}/><text x="200" y="54" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>Σ Reduce</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 6 × 25 = 150 MAC cycles</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 25 MAC cycles + log₂(6) reduce</text></>,
      },
      {n:"Pixel Parallelism",w:"Compute several adjacent output pixels simultaneously using multiple window engines.",h:"Multiple sliding-window/MAC engines operate on adjacent spatial positions, sharing most input data through line buffers.",ex:"Process 4 adjacent Conv1 output pixels per cycle. Windows overlap by 4 columns, so ~80% of input data is shared between engines.",rtl:"4 parallel window_extract + mac_array modules. Line buffer feeds all 4. Output: 4 valid pixels per cycle.",lat:"↓",thr:"↑↑",fmax:"—",area:"↑",membw:"↑",simrt:"—",diff:"Medium",
        bef_lbl:"1 pixel per cycle",aft_lbl:"4 pixels per cycle",
        bef_svg:(c)=><><rect x="170" y="10" width="100" height="35" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="32" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>1 Window</text></>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={30+i*110} y="10" width="95" height="30" rx="4" fill={c+"18"} stroke={c}/><text x={77+i*110} y="29" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Win {i}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 576 cycles (24×24 output)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 144 cycles (4 pixels/cycle)</text></>,
      },
      {n:"Batch Parallelism",w:"Process multiple independent images concurrently by duplicating CNN engines.",h:"Replicate the entire inference pipeline (or selected stages) so N images are processed simultaneously.",ex:"Duplicate the full LeNet engine: 2 instances process 2 images in parallel. Throughput doubles; single-image latency unchanged.",rtl:"Instantiate 2 lenet_top modules with independent weight ROMs (or shared if dual-port). Independent input/output buses.",lat:"—",thr:"↑↑",fmax:"—",area:"↑↑↑",membw:"↑↑",simrt:"↑",diff:"Low",
        bef_lbl:"1 image at a time",aft_lbl:"2 images simultaneously",
        bef_svg:(c)=><><rect x="120" y="5" width="200" height="50" rx="6" fill={C.s1} stroke={C.border}/><text x="220" y="25" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>CNN Engine</text><text x="220" y="42" textAnchor="middle" fill={C.t3} fontSize="9" fontFamily={mono}>Image A</text></>,
        aft_svg:(c)=><>{[0,1].map(i=><g key={i}><rect x={30+i*230} y="5" width="200" height="50" rx="6" fill={c+"12"} stroke={c}/><text x={130+i*230} y="25" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Engine {i}</text><text x={130+i*230} y="42" textAnchor="middle" fill={c} fontSize="9" fontFamily={mono}>Image {String.fromCharCode(65+i)}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 1 image / T cycles</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 2 images / T cycles (2× throughput)</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>Note: single-image latency unchanged</text></>,
      },
    ]
  },
  { id:2, t:"Pipelining", c:C.c2, ico:"🔗", d:"Overlap operations across stages to increase throughput and/or clock frequency.",
    items:[
      {n:"Layer Pipelining",w:"Overlap major CNN layers so different images occupy different stages simultaneously.",h:"Insert pipeline registers between Conv, Pool, and FC stages. After pipeline fill, each stage processes a different image every cycle.",ex:"Image A in Conv2 while Image B in Conv1. After 5-stage fill, one image completes per stage-latency interval.",rtl:"Add pipeline registers (valid, data) between each major stage. Each stage operates independently with handshaking.",lat:"— (slightly ↑)",thr:"↑↑↑",fmax:"—",area:"↑",membw:"↑",simrt:"—",diff:"Medium",
        bef_lbl:"Stages execute one-at-a-time",aft_lbl:"Stages overlap across images",
        bef_svg:(c)=><>{["Conv1","Pool1","Conv2","FC"].map((s,i)=><g key={i}><rect x={10+i*115} y="10" width="100" height="28" rx="4" fill={i===0?c+"20":C.s1} stroke={i===0?c:C.border}/><text x={60+i*115} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}</>,
        aft_svg:(c)=>{const cols=[C.c1,C.c2,C.c3,C.c4];return<>{["T1","T2","T3","T4"].map((t,r)=><g key={r}><text x="5" y={18+r*18} fill={C.t3} fontSize="8" fontFamily={mono}>{t}</text>{Array.from({length:Math.min(r+1,4)},(_, ci)=>{const stage=ci;const img=r-ci;if(img<0||img>3)return null;return<rect key={ci} x={30+stage*110} y={5+r*18} width="100" height="14" rx="2" fill={cols[img]+"25"} stroke={cols[img]} strokeWidth="0.8"/>})}{Array.from({length:Math.min(r+1,4)},(_,ci)=>{const img=r-ci;if(img<0||img>3)return null;return<text key={`t${ci}`} x={80+ci*110} y={16+r*18} textAnchor="middle" fill={C.t1} fontSize="7" fontFamily={mono}>Img{img}</text>})}</g>)}</>},
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: Total = Σ all stage latencies per image</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: Throughput = 1 image / slowest_stage</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>First image same latency; steady-state much higher</text></>,
      },
      {n:"Operator Pipelining",w:"Split long arithmetic operations into registered stages to shorten the critical path.",h:"Insert pipeline registers between multiplier output, reduction-tree levels, bias addition, and activation. Each stage is shorter, allowing higher Fmax.",ex:"A 25-MAC + reduce + bias + ReLU path broken into 4 stages: multiply → reduce-L1 → reduce-L2+bias → ReLU.",rtl:"Add `always_ff` pipeline registers between each arithmetic stage. Propagate valid signals through the pipeline.",lat:"↑ (more stages)",thr:"↑↑",fmax:"↑↑",area:"↑",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"Long combinational path",aft_lbl:"Registered pipeline stages",
        bef_svg:(c)=><><rect x="40" y="12" width="380" height="30" rx="4" fill={C.s1} stroke={C.border}/><text x="230" y="31" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>MUL → ADD → ADD → ADD → BIAS → ReLU (long path!)</text></>,
        aft_svg:(c)=><>{["MUL","Σ L1","Σ L2+B","ReLU"].map((s,i)=><g key={i}><rect x={10+i*115} y="10" width="95" height="28" rx="4" fill={c+"18"} stroke={c}/><text x={57+i*115} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text>{i<3&&<rect x={107+i*115} y="13" width="6" height="22" rx="2" fill={c+"60"}/>}</g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: Fmax limited by longest comb path</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: Fmax ↑↑ (each stage is short)</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>Latency in cycles increases, but cycle time decreases</text></>,
      },
      {n:"Pipeline Registers",w:"Break long combinational paths by inserting registers between arithmetic stages.",h:"Registers between MUL, ADD, activation, and memory stages create pipeline boundaries. Each stage completes in one shorter clock cycle.",ex:"Place registers between: multiplication products → reduction tree → bias addition → ReLU → output register.",rtl:"Insert `always_ff @(posedge clk)` between each computational stage. Ensure valid/data propagation matches pipeline depth.",lat:"↑",thr:"↑",fmax:"↑↑",area:"↑",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"No intermediate registers",aft_lbl:"Registers between every stage",
        bef_svg:(c)=><>{["MUL","ADD","ACT","OUT"].map((s,i)=><g key={i}><rect x={20+i*110} y="10" width="90" height="28" rx="4" fill={C.s1} stroke={C.border}/><text x={65+i*110} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text>{i<3&&<line x1={110+i*110} y1="24" x2={130+i*110} y2="24" stroke={C.t3}/>}</g>)}</>,
        aft_svg:(c)=><>{["MUL","ADD","ACT","OUT"].map((s,i)=><g key={i}><rect x={20+i*110} y="10" width="90" height="28" rx="4" fill={c+"18"} stroke={c}/><text x={65+i*110} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text>{i<3&&<rect x={113+i*110} y="14" width="5" height="20" rx="2" fill={c+"60"}/>}</g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Each ▮ = pipeline register = new clock boundary</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>Critical path = longest single stage, not total</text></>,
      },
      {n:"Deep Pipelining",w:"Use many fine-grained pipeline stages to maximize clock frequency.",h:"Partition large datapaths into many short registered stages. Trade latency (more cycles) for frequency (shorter cycle time).",ex:"Split a 25-input reduction tree into 5 registered levels. Each level has ~5 adders. Fmax is determined by 1 adder delay, not 5.",rtl:"Multi-level generate blocks with pipeline registers at each tree level. Pipeline depth = ceil(log₂(N)).",lat:"↑",thr:"↑↑↑",fmax:"↑↑↑",area:"↑",membw:"—",simrt:"—",diff:"Medium",
        bef_lbl:"Shallow pipeline (few stages)",aft_lbl:"Deep pipeline (many stages)",
        bef_svg:(c)=><>{[0,1,2].map(i=><g key={i}><rect x={40+i*150} y="10" width="120" height="30" rx="5" fill={C.s1} stroke={C.border}/><text x={100+i*150} y="29" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Stage {i}</text></g>)}</>,
        aft_svg:(c)=><>{[0,1,2,3,4,5,6].map(i=><g key={i}><rect x={8+i*66} y="10" width="56" height="28" rx="3" fill={c+"15"} stroke={c}/><text x={36+i*66} y="28" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>S{i}</text>{i<6&&<rect x={66+i*66} y="15" width="4" height="18" rx="1" fill={c+"50"}/>}</g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 3 stages, long cycle time</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 7 stages, much shorter cycle time</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>Throughput = 1/T_stage (can be very high)</text></>,
      },
      {n:"Multiple Images in Flight",w:"Keep several images inside the pipeline simultaneously after pipeline fill.",h:"Track valid/control signals for each image in each stage. After fill, every stage is busy with a different image.",ex:"5-stage LeNet pipeline: after fill, images 0-4 occupy stages FC→Pool2→Conv2→Pool1→Conv1 simultaneously.",rtl:"Each stage has valid_in/valid_out. Pipeline control tracks image IDs. No stage is idle after fill.",lat:"—",thr:"↑↑↑",fmax:"—",area:"↑",membw:"↑",simrt:"—",diff:"Medium",
        bef_lbl:"Pipeline mostly empty",aft_lbl:"All stages occupied",
        bef_svg:(c)=><>{["Conv1","Pool1","Conv2","Pool2","FC"].map((s,i)=><g key={i}><rect x={5+i*92} y="10" width="84" height="28" rx="4" fill={i===0?C.s2:C.s1} stroke={i===0?C.border:C.border}/><text x={47+i*92} y="28" textAnchor="middle" fill={i===0?C.t1:C.t3} fontSize="8" fontFamily={mono}>{i===0?"Img0":""}</text></g>)}</>,
        aft_svg:(c)=>{const cols=[C.c1,C.c2,C.c3,C.c4,C.c5];return<>{["Conv1","Pool1","Conv2","Pool2","FC"].map((s,i)=><g key={i}><rect x={5+i*92} y="10" width="84" height="28" rx="4" fill={cols[i]+"20"} stroke={cols[i]}/><text x={47+i*92} y="22" textAnchor="middle" fill={C.t1} fontSize="7" fontFamily={mono}>{s}</text><text x={47+i*92} y="34" textAnchor="middle" fill={cols[i]} fontSize="8" fontFamily={mono}>Img{4-i}</text></g>)}</>},
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 1 image uses entire pipeline</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 5 images in 5 stages simultaneously</text></>,
      },
      {n:"Initiation Interval (II)",w:"The number of cycles between accepting successive new inputs. II=1 is ideal.",h:"Design pipeline so that a new input can enter every cycle (II=1). The pipeline accepts input continuously without stalls.",ex:"With II=1, the LeNet pipeline accepts a new image every cycle after fill. Steady-state throughput = 1 image/cycle.",rtl:"Ensure no stage takes >1 cycle or add buffering. All stages must have matching throughput (balanced pipeline).",lat:"—",thr:"↑↑↑",fmax:"—",area:"—",membw:"↑",simrt:"—",diff:"High",
        bef_lbl:"II = N (slow input rate)",aft_lbl:"II = 1 (max input rate)",
        bef_svg:(c)=><>{[0,1,2].map(i=><g key={i}><rect x={20+i*160} y="10" width="130" height="25" rx="4" fill={C.s1} stroke={C.border}/><text x={85+i*160} y="26" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Image {i}</text></g>)}<text x="240" y="55" textAnchor="middle" fill={C.t3} fontSize="9" fontFamily={mono}>Gap = 4 cycles between starts</text></>,
        aft_svg:(c)=><>{[0,1,2,3,4].map(i=><g key={i}><rect x={10+i*93} y="10" width="85" height="25" rx="4" fill={c+"18"} stroke={c}/><text x={52+i*93} y="26" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Img {i}</text></g>)}<text x="240" y="55" textAnchor="middle" fill={c} fontSize="9" fontFamily={mono}>II=1: new image every cycle</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>II = cycles between successive inputs</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>II=1 → maximum steady-state throughput</text></>,
      },
    ]
  },
  { id:3, t:"Memory & Data Movement", c:C.c3, ico:"💾", d:"Move less data and reuse it efficiently to keep compute units fed.",
    items:[
      {n:"Data Reuse",w:"Keep frequently accessed values in local registers/buffers instead of re-fetching from memory.",h:"Adjacent convolution windows share ~80% of their pixels. Store overlapping data locally so it doesn't need re-fetching.",ex:"Adjacent 5×5 windows in Conv1 share 20 of 25 pixels. Local register files hold the shared data, fetching only 5 new values per shift.",rtl:"Register array holds current window. Shift logic moves data; only new column/row loaded from line buffer each cycle.",lat:"↓",thr:"↑",fmax:"—",area:"↑",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"Re-fetch all 25 pixels each time",aft_lbl:"Reuse 20, fetch only 5 new",
        bef_svg:(c)=><><rect x="120" y="5" width="200" height="45" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="22" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Memory → 25 reads</text><text x="220" y="38" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>every pixel computation</text></>,
        aft_svg:(c)=><><rect x="40" y="5" width="160" height="40" rx="5" fill={c+"15"} stroke={c}/><text x="120" y="20" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Local Regs: 20 reused</text><text x="120" y="36" textAnchor="middle" fill={c} fontSize="8" fontFamily={mono}>80% reuse!</text><rect x="240" y="5" width="160" height="40" rx="5" fill={C.s2} stroke={C.border}/><text x="320" y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Mem: 5 new reads</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 25 reads/pixel × bandwidth cost</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 5 reads/pixel (5× bandwidth reduction)</text></>,
      },
      {n:"Streaming",w:"Send results directly from one stage to the next without writing to intermediate memory.",h:"Use valid/data handshake interfaces between stages. Data flows through without materializing full intermediate tensors.",ex:"Conv1 output pixels stream directly to Pool1 via valid/data signals. No need to store the full 24×24×6 feature map.",rtl:"Wire conv1_valid/conv1_data directly to pool1 input. Pool1 accumulates its 2×2 window from the stream.",lat:"↓",thr:"↑",fmax:"—",area:"↓",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"Write full tensor to memory",aft_lbl:"Stream pixel-by-pixel",
        bef_svg:(c)=><>{["Conv","MEM","Pool"].map((s,i)=><g key={i}><rect x={30+i*160} y="10" width="110" height="28" rx="4" fill={s==="MEM"?C.red+"15":C.s1} stroke={s==="MEM"?C.red:C.border}/><text x={85+i*160} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text>{i<2&&<line x1={140+i*160} y1="24" x2={190+i*160} y2="24" stroke={C.t3}/>}</g>)}</>,
        aft_svg:(c)=><>{["Conv","Pool"].map((s,i)=><g key={i}><rect x={60+i*200} y="10" width="120" height="28" rx="4" fill={c+"18"} stroke={c}/><text x={120+i*200} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}<line x1="180" y1="24" x2="260" y2="24" stroke={c} strokeWidth="2"/><text x="220" y="18" textAnchor="middle" fill={c} fontSize="8" fontFamily={mono}>valid+data</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: Conv finishes → write all → Pool reads all</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: Conv streams → Pool consumes immediately</text></>,
      },
      {n:"Line Buffers",w:"Store K image rows in shift registers to construct convolution windows on-the-fly.",h:"For a 5×5 kernel, maintain 5 row buffers. As pixels arrive, shift through buffers to construct the current window without re-reading from memory.",ex:"Conv1: 5 line buffers of width 28. Each stores one row. The 5×5 window is formed by reading 5 positions from each buffer.",rtl:"5 shift registers of width 28. New pixel enters row 0, shifts propagate. Window taps at fixed positions across all 5 rows.",lat:"↓",thr:"↑",fmax:"—",area:"↑",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"Re-read rows from memory",aft_lbl:"Rows buffered in shift registers",
        bef_svg:(c)=><><rect x="120" y="5" width="200" height="50" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="25" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Memory</text><text x="220" y="42" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>Random access for each window</text></>,
        aft_svg:(c)=><>{[0,1,2,3,4].map(i=><g key={i}><rect x="80" y={2+i*12} width="280" height="10" rx="2" fill={c} fillOpacity={0.08+i*0.04} stroke={c} strokeWidth="0.5"/><text x="70" y={10+i*12} textAnchor="end" fill={C.t3} fontSize="7" fontFamily={mono}>R{i}</text></g>)}<text x="400" y="35" fill={c} fontSize="9" fontFamily={mono}>→ Window</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: N×K memory reads per window</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 1 new pixel per cycle (shift register)</text></>,
      },
      {n:"Sliding Windows",w:"Reuse overlapping window data by shifting and inserting only new values.",h:"When moving to the next pixel position, shift the current window contents and load only the new column (or row).",ex:"Moving the 5×5 window one position right: shift all 25 registers left, load 5 new values from the rightmost column.",rtl:"5×5 register array with shift logic. Each cycle: shift columns left, load new_col[4:0] into rightmost position.",lat:"↓",thr:"↑",fmax:"—",area:"—",membw:"↓↓",simrt:"↓",diff:"Low",
        bef_lbl:"Load full 25 values per window",aft_lbl:"Shift + load 5 new values",
        bef_svg:(c)=><><rect x="130" y="5" width="180" height="45" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="22" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>25 loads per position</text><text x="220" y="38" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>wasteful!</text></>,
        aft_svg:(c)=><><rect x="40" y="5" width="150" height="45" rx="5" fill={c+"10"} stroke={c} strokeDasharray="4"/><text x="115" y="22" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Window @ t</text><text x="115" y="38" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>shift →</text><rect x="250" y="5" width="150" height="45" rx="5" fill={c+"18"} stroke={c}/><text x="325" y="22" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Window @ t+1</text><rect x="370" y="8" width="26" height="39" rx="2" fill={c+"40"} stroke={c}/><text x="383" y="31" textAnchor="middle" fill={C.t1} fontSize="7" fontFamily={mono} fontWeight="bold">NEW</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 25 reads per window position</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 5 reads per shift (5× reduction)</text></>,
      },
      {n:"Double Buffering",w:"Fill one buffer while the consumer reads from another, overlapping data movement with computation.",h:"Two buffers alternate roles: while Stage N writes results into Buffer A, Stage N+1 reads from Buffer B (filled previously).",ex:"While Conv2 writes feature maps to Buffer A, Pool2 processes feature maps from Buffer B (written by Conv2 in the previous block).",rtl:"Two SRAM/register blocks with a mux controlled by a phase signal. Phase toggles after each processing block completes.",lat:"↓",thr:"↑↑",fmax:"—",area:"↑",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"Wait for write, then read",aft_lbl:"Write and read simultaneously",
        bef_svg:(c)=><><rect x="80" y="5" width="280" height="25" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="22" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>WRITE → then → READ (sequential)</text></>,
        aft_svg:(c)=><><rect x="30" y="5" width="190" height="30" rx="4" fill={c+"15"} stroke={c}/><text x="125" y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Buf A: WRITE ✏️</text><rect x="240" y="5" width="190" height="30" rx="4" fill={c+"25"} stroke={c}/><text x="335" y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Buf B: READ 📖</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: T_write + T_read (serial)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: max(T_write, T_read) (overlapped)</text></>,
      },
      {n:"Ping-Pong Buffers",w:"Alternate two buffers between read/write roles after each processing block.",h:"Identical to double buffering but emphasizes the role-swapping pattern. Ping and Pong alternate every block.",ex:"After Conv1 fills Ping, Pool1 reads Ping while Conv1 fills Pong with the next block. Roles swap continuously.",rtl:"Two equal-sized buffers. `assign write_buf = phase ? buf_b : buf_a; assign read_buf = phase ? buf_a : buf_b;`",lat:"↓",thr:"↑↑",fmax:"—",area:"↑",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"Single buffer (serialize)",aft_lbl:"Ping-Pong (overlap)",
        bef_svg:(c)=><><rect x="140" y="10" width="160" height="35" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="32" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>1 Buffer</text></>,
        aft_svg:(c)=><><rect x="40" y="10" width="170" height="30" rx="5" fill={c+"20"} stroke={c}/><text x="125" y="29" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>PING (Write)</text><rect x="240" y="10" width="170" height="30" rx="5" fill={c+"10"} stroke={c}/><text x="325" y="29" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>PONG (Read)</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Roles swap after each block completion</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>Zero producer/consumer idle time</text></>,
      },
      {n:"Memory Banking",w:"Split memory into independently accessible banks so multiple values can be read in parallel.",h:"Map different channels, pixel groups, or weight sets to separate memory banks. Each bank has its own read port.",ex:"Conv2 weights: store each of 6 input-channel weight sets in a separate bank. All 6 can be read simultaneously.",rtl:"6 separate BRAM/register-file instances. Address generation maps channel index to bank select.",lat:"↓",thr:"↑↑",fmax:"—",area:"↑",membw:"↑↑↑",simrt:"—",diff:"Medium",
        bef_lbl:"Single-port memory (1 read/cycle)",aft_lbl:"Banked memory (N reads/cycle)",
        bef_svg:(c)=><><rect x="130" y="5" width="180" height="45" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="22" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>1 Memory Block</text><text x="220" y="38" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>1 read/cycle</text></>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={20+i*115} y="5" width="100" height="40" rx="4" fill={c+"15"} stroke={c}/><text x={70+i*115} y="22" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>Bank {i}</text><text x={70+i*115} y="36" textAnchor="middle" fill={c} fontSize="7" fontFamily={mono}>read port</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 4 reads = 4 cycles</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 4 reads = 1 cycle (4 banks)</text></>,
      },
      {n:"Reducing Memory Transfers",w:"Eliminate unnecessary intermediate reads/writes by fusing stages or streaming directly.",h:"Instead of writing Conv output to memory, reading it for ReLU, writing ReLU output, reading it for Pool — fuse all three.",ex:"Fuse Conv1→ReLU→Pool1: Conv output feeds directly to ReLU (sign check), then to Pool (max/avg), no intermediate storage.",rtl:"Single module with conv_out → relu_out → pool_out wires. No intermediate BRAM. `wire [15:0] conv_out, relu_out;`",lat:"↓↓",thr:"↑",fmax:"—",area:"↓",membw:"↓↓↓",simrt:"↓↓",diff:"Medium",
        bef_lbl:"Conv → MEM → ReLU → MEM → Pool → MEM",aft_lbl:"Conv → ReLU → Pool (fused)",
        bef_svg:(c)=><>{["Conv","MEM","ReLU","MEM","Pool","MEM"].map((s,i)=><g key={i}><rect x={2+i*78} y="10" width="70" height="25" rx="3" fill={s==="MEM"?C.red+"12":C.s1} stroke={s==="MEM"?C.red+"50":C.border}/><text x={37+i*78} y="26" textAnchor="middle" fill={s==="MEM"?C.red:C.t1} fontSize="8" fontFamily={mono}>{s}</text></g>)}</>,
        aft_svg:(c)=><><rect x="60" y="5" width="320" height="38" rx="6" fill={c+"10"} stroke={c} strokeWidth="1.5"/>{["Conv","ReLU","Pool"].map((s,i)=><g key={i}><rect x={80+i*100} y="12" width="80" height="24" rx="4" fill={c+"20"}/><text x={120+i*100} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 6 memory operations (3 writes + 3 reads)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 0 intermediate memory ops</text></>,
      },
    ]
  },
  { id:4, t:"Arithmetic & Computation", c:C.c4, ico:"🔢", d:"Make calculations shallower, smaller, or eliminate unnecessary work.",
    items:[
      {n:"Reduction Trees",w:"Replace serial accumulation with a balanced binary adder tree for O(log N) depth.",h:"25 products → balanced tree: 13 adders at level 1, 7 at level 2, 4, 2, 1. Total depth: 5 levels vs 24 serial additions.",ex:"Conv1: 25 products reduced in 5 levels instead of 24 sequential additions. Critical path: 5 adder delays.",rtl:"`assign l1[0]=p[0]+p[1]; ... assign l5=l4[0]+l4[1];` — generate block creates balanced tree.",lat:"↓↓",thr:"↑",fmax:"↑",area:"—",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"Serial chain: O(N) depth",aft_lbl:"Balanced tree: O(log N) depth",
        bef_svg:(c)=><><rect x="60" y="5" width="320" height="30" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>p0 + p1 + p2 + ... + p24 (24 serial adds)</text></>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><rect key={i} x={30+i*55} y="3" width="45" height="14" rx="2" fill={c+"15"} stroke={c} strokeWidth="0.5"/>)}{[0,1].map(i=><rect key={i} x={55+i*110} y="22" width="45" height="14" rx="2" fill={c+"25"} stroke={c} strokeWidth="0.5"/>)}<rect x="80" y="41" width="60" height="16" rx="3" fill={c+"35"} stroke={c}/><text x="110" y="53" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>SUM</text><text x="300" y="35" fill={c} fontSize="9" fontFamily={mono}>3 levels!</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 24 adder delays (serial)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 5 adder delays (⌈log₂25⌉)</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>~5× shorter critical path</text></>,
      },
      {n:"Comparator Trees",w:"Find argmax with parallel pairwise comparisons in O(log N) depth.",h:"Compare 10 class scores in a binary tree: 5 comparators → 3 → 2 → 1. Track winning index alongside value.",ex:"FC2 produces 10 scores. Comparator tree finds the maximum and its index in 4 levels instead of 9 sequential comparisons.",rtl:"Each comparator: `assign {win_val, win_idx} = (a > b) ? {a, idx_a} : {b, idx_b};`",lat:"↓",thr:"↑",fmax:"↑",area:"—",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"Sequential max search",aft_lbl:"Parallel comparator tree",
        bef_svg:(c)=><><rect x="100" y="5" width="240" height="30" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>if(s0>max) max=s0; if(s1>max)... ×10</text></>,
        aft_svg:(c)=><>{[0,1,2,3,4].map(i=><rect key={i} x={10+i*92} y="3" width="80" height="14" rx="2" fill={c+"15"} stroke={c} strokeWidth="0.5"/>)}{[0,1].map(i=><rect key={i} x={55+i*184} y="22" width="80" height="14" rx="2" fill={c+"25"} stroke={c}/>)}<rect x="140" y="41" width="80" height="16" rx="3" fill={c+"35"} stroke={c}/><text x="180" y="53" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>Winner</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 9 sequential comparisons</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: ⌈log₂10⌉ = 4 parallel levels</text></>,
      },
      {n:"Fixed-Point / Quantization",w:"Replace floating-point with compact integer/fixed-point arithmetic.",h:"Use INT8 for inputs/weights (8-bit multiply) with INT32 accumulators. 4× smaller multipliers vs FP32.",ex:"Conv weights: INT8 (8b×8b=16b products, 32b accumulator). Verify accuracy against golden model ±tolerance.",rtl:"`wire signed [7:0] w, x; wire signed [15:0] prod = w * x; reg signed [31:0] acc;`",lat:"↓",thr:"↑",fmax:"↑",area:"↓↓",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"FP32: 32-bit multiply",aft_lbl:"INT8: 8-bit multiply",
        bef_svg:(c)=><><rect x="100" y="5" width="240" height="35" rx="5" fill={C.red+"12"} stroke={C.red+"50"}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>FP32 × FP32 = big hardware</text></>,
        aft_svg:(c)=><><rect x="140" y="5" width="160" height="35" rx="5" fill={c+"20"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>INT8 × INT8 = tiny!</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>FP32 multiplier: ~1000 LUTs</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>INT8 multiplier: ~60 LUTs (16× smaller)</text></>,
      },
      {n:"Constant Folding",w:"Precompute all invariant expressions at compile/elaboration time.",h:"Scales, biases, normalization constants — anything that doesn't change per-inference — computed offline and hardcoded.",ex:"Quantization scale factors, bias terms, address offsets all resolved before synthesis. Zero runtime cost.",rtl:"Use `parameter` or `localparam` for all constants. `localparam SCALE = 16'd1234;`",lat:"↓",thr:"—",fmax:"—",area:"↓",membw:"—",simrt:"↓",diff:"Low",
        bef_lbl:"Runtime computation of constants",aft_lbl:"Precomputed at compile time",
        bef_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>scale*offset+bias computed at runtime</text></>,
        aft_svg:(c)=><><rect x="120" y="5" width="200" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>localparam = 0x3F (done!)</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: MUL + ADD at runtime</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 0 runtime ops (hardcoded)</text></>,
      },
      {n:"Constant Multipliers",w:"Replace generic multipliers with shift-add networks when one operand is a known constant.",h:"Multiplication by a constant can be decomposed into shifts and adds. x×5 = (x<<2)+x. Saves multiplier resources.",ex:"If a quantization scale is fixed at 5, replace the multiplier with `(x << 2) + x`.",rtl:"`assign result = (x << 2) + x; // x * 5 without a multiplier`",lat:"↓",thr:"—",fmax:"↑",area:"↓",membw:"—",simrt:"—",diff:"Low",
        bef_lbl:"Generic multiplier",aft_lbl:"Shift + add network",
        bef_svg:(c)=><><rect x="140" y="5" width="160" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>x × 5 (full MUL)</text></>,
        aft_svg:(c)=><><rect x="100" y="5" width="240" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>(x « 2) + x (shift+add)</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Multiplier: ~60 LUTs, longer path</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>Shift+Add: ~10 LUTs, shorter path</text></>,
      },
      {n:"Activation Optimization",w:"Implement ReLU as a single sign-check and MUX — near-zero hardware cost.",h:"ReLU: if MSB is 1 (negative), output 0; else pass through. One comparator + one MUX.",ex:"All Conv layers use ReLU: `assign relu_out = data[MSB] ? 0 : data;` — essentially free in hardware.",rtl:"`assign relu_out = (acc[31]) ? 32'd0 : acc; // ReLU = 1 MUX`",lat:"—",thr:"—",fmax:"—",area:"↓↓",membw:"—",simrt:"—",diff:"Very Low",
        bef_lbl:"General activation function",aft_lbl:"ReLU: 1 MUX",
        bef_svg:(c)=><><rect x="100" y="5" width="240" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>LUT-based activation (expensive)</text></>,
        aft_svg:(c)=><><rect x="130" y="5" width="180" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>MSB ? 0 : x (1 MUX!)</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>ReLU adds essentially zero latency</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>Can be fused into the pipeline stage</text></>,
      },
      {n:"Removing Unnecessary Ops",w:"Skip computations whose results aren't needed (e.g., softmax before argmax).",h:"If only the class index is needed, argmax of raw scores gives the same result as argmax of softmax(scores). Skip softmax entirely.",ex:"LeNet output: skip softmax, apply argmax directly to FC2 output. Saves an expensive exp/div computation.",rtl:"Remove softmax module entirely. Wire FC2 output directly to argmax comparator tree.",lat:"↓",thr:"↑",fmax:"—",area:"↓↓",membw:"—",simrt:"↓",diff:"Very Low",
        bef_lbl:"FC2 → Softmax → Argmax",aft_lbl:"FC2 → Argmax (skip softmax)",
        bef_svg:(c)=><>{["FC2","Softmax","Argmax"].map((s,i)=><g key={i}><rect x={30+i*155} y="10" width="130" height="28" rx="4" fill={s==="Softmax"?C.red+"12":C.s1} stroke={s==="Softmax"?C.red+"50":C.border}/><text x={95+i*155} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}</>,
        aft_svg:(c)=><>{["FC2","Argmax"].map((s,i)=><g key={i}><rect x={80+i*200} y="10" width="130" height="28" rx="4" fill={c+"18"} stroke={c}/><text x={145+i*200} y="28" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}<line x1="210" y1="24" x2="280" y2="24" stroke={c} strokeWidth="2"/></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: softmax needs exp, div (expensive)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: direct argmax (comparator tree only)</text></>,
      },
    ]
  },
  { id:5, t:"Architecture & Control", c:C.c5, ico:"🏗️", d:"Organize hardware so it doesn't wait unnecessarily.",
    items:[
      {n:"FSM Optimization",w:"Remove unnecessary states and idle cycles from the control FSM.",h:"Replace fixed WAIT states with done/valid signal-driven transitions. Eliminate dead cycles.",ex:"Remove arbitrary 2-cycle WAIT between LOAD and COMPUTE. Use load_done signal for immediate transition.",rtl:"Merge compatible states. Use `always_comb` for next-state logic driven by completion signals, not counters.",lat:"↓",thr:"↑",fmax:"—",area:"↓",membw:"—",simrt:"↓",diff:"Low",
        bef_lbl:"FSM with unnecessary WAITs",aft_lbl:"Optimized FSM",
        bef_svg:(c)=><>{["IDLE","LOAD","WAIT","WAIT","COMP","DONE"].map((s,i)=><g key={i}><circle cx={20+i*80} cy="22" r="16" fill={s==="WAIT"?C.red+"15":"none"} stroke={s==="WAIT"?C.red:C.border}/><text x={20+i*80} y="26" textAnchor="middle" fill={s==="WAIT"?C.red:C.t1} fontSize="7" fontFamily={mono}>{s}</text></g>)}</>,
        aft_svg:(c)=><>{["IDLE","LOAD","COMP","DONE"].map((s,i)=><g key={i}><circle cx={40+i*120} cy="22" r="18" fill={c+"15"} stroke={c}/><text x={40+i*120} y="26" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s}</text>{i<3&&<line x1={58+i*120} y1="22" x2={142+i*120} y2="22" stroke={c} strokeWidth="1"/>}</g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 6 states (2 unnecessary WAITs)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 4 states (signal-driven)</text></>,
      },
      {n:"Valid/Ready Handshaking",w:"Use valid/ready protocol to coordinate pipeline stages without fixed timing.",h:"Producer asserts valid when data is available. Consumer asserts ready when it can accept. Transfer occurs on valid∧ready.",ex:"Conv1 output → Pool1 input: Conv1 asserts conv1_valid, Pool1 asserts pool1_ready. Data transfers only when both are high.",rtl:"`assign transfer = valid_in & ready_out; always_ff @(posedge clk) if(transfer) data_reg <= data_in;`",lat:"—",thr:"↑",fmax:"—",area:"↑",membw:"—",simrt:"—",diff:"Medium",
        bef_lbl:"Fixed timing / counter-based",aft_lbl:"Valid/Ready handshake",
        bef_svg:(c)=><><rect x="60" y="5" width="320" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Counter-based: assume N cycles per stage</text></>,
        aft_svg:(c)=><><rect x="30" y="5" width="150" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="105" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Producer</text><rect x="260" y="5" width="150" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="335" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Consumer</text><text x="220" y="18" textAnchor="middle" fill={c} fontSize="8" fontFamily={mono}>valid→</text><text x="220" y="32" textAnchor="middle" fill={C.c3} fontSize="8" fontFamily={mono}>←ready</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: fixed delays may cause idle cycles</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: data flows as fast as both sides allow</text></>,
      },
      {n:"Layer Fusion",w:"Combine adjacent operations (Conv+ReLU+Pool) into a single datapath with no intermediate storage.",h:"Instead of separate Conv, ReLU, and Pool modules with memory between them, create one fused module.",ex:"Fused Conv1-ReLU-Pool1: convolution output → sign check (ReLU) → max/average (Pool) in one pipeline, no BRAM between them.",rtl:"Single module: conv result feeds wire to ReLU logic, ReLU output feeds wire to pooling accumulator.",lat:"↓↓",thr:"↑",fmax:"—",area:"↓",membw:"↓↓↓",simrt:"↓↓",diff:"Medium",
        bef_lbl:"Separate stages with memory between",aft_lbl:"Fused datapath",
        bef_svg:(c)=><>{["Conv","MEM","ReLU","MEM","Pool"].map((s,i)=><g key={i}><rect x={5+i*92} y="10" width="82" height="25" rx="3" fill={s==="MEM"?C.red+"12":C.s1} stroke={s==="MEM"?C.red+"40":C.border}/><text x={46+i*92} y="26" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s}</text></g>)}</>,
        aft_svg:(c)=><><rect x="60" y="3" width="320" height="38" rx="6" fill={c+"08"} stroke={c}/>{["Conv","ReLU","Pool"].map((s,i)=><g key={i}><rect x={80+i*100} y="10" width="80" height="24" rx="4" fill={c+"20"}/><text x={120+i*100} y="26" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 4 memory round-trips</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 0 intermediate memory ops</text></>,
      },
      {n:"Dataflow Architecture",w:"Choose the overall strategy for how data moves through the CNN hardware.",h:"Options range from sequential (reuse one engine) to fully parallel (dedicated hardware for every layer).",ex:"Streaming dataflow: each layer has dedicated hardware, data flows through via valid/ready. Best for low-latency LeNet.",rtl:"Architecture choice determines module hierarchy: single reusable engine vs generate-based per-layer instances.",lat:"varies",thr:"varies",fmax:"—",area:"varies",membw:"varies",simrt:"—",diff:"High",
        bef_lbl:"Sequential (1 engine, reuse)",aft_lbl:"Streaming (dedicated per layer)",
        bef_svg:(c)=><><rect x="120" y="5" width="200" height="40" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="20" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>1 Generic Engine</text><text x="220" y="36" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>reused for each layer</text></>,
        aft_svg:(c)=><>{["Conv1","Pool1","Conv2","Pool2","FC"].map((s,i)=><g key={i}><rect x={5+i*92} y="5" width="82" height="30" rx="4" fill={c+"15"} stroke={c}/><text x={46+i*92} y="24" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Sequential: min area, max latency</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>Streaming: max throughput, more area</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>Choose based on your target metric</text></>,
      },
      {n:"Hardware Specialization",w:"Build RTL specifically for LeNet's fixed dimensions rather than a generic CNN engine.",h:"Hard-code kernel size (5×5), channel counts (1/6/16), FC sizes (120/84/10). Eliminates generic control overhead.",ex:"No runtime configuration of kernel size, stride, or channel count. All dimensions are compile-time parameters.",rtl:"Use `parameter KERN=5, CH_IN=6, CH_OUT=16;` everywhere. No runtime dimension registers or nested loop counters.",lat:"↓",thr:"↑",fmax:"↑",area:"↓",membw:"—",simrt:"↓",diff:"Low",
        bef_lbl:"Generic CNN engine",aft_lbl:"LeNet-specific RTL",
        bef_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="5" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Generic: config regs, nested loops, muxes</text></>,
        aft_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="5" fill={c+"18"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>Fixed: 5×5, 6ch, 16ch, 120/84/10</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: runtime config adds muxes + counters</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: compile-time constants → simpler logic</text></>,
      },
      {n:"Precomputation",w:"Perform all invariant work (weight quantization, address maps, test vectors) offline before simulation.",h:"Generate quantized weight .hex files, golden reference outputs, memory initialization files — all before RTL simulation starts.",ex:"Python script: train → quantize → export weights.hex, golden_conv1.hex, golden_fc2.hex. RTL uses $readmemh.",rtl:"`initial $readmemh(\"weights_conv1.hex\", weight_rom);` — no runtime weight processing.",lat:"↓",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓↓",diff:"Low",
        bef_lbl:"Runtime weight processing",aft_lbl:"Precomputed .hex files",
        bef_svg:(c)=><><rect x="60" y="5" width="320" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>RTL computes scales, converts weights...</text></>,
        aft_svg:(c)=><>{["Train","Quantize","Export .hex","$readmemh"].map((s,i)=><g key={i}><rect x={10+i*115} y="5" width="100" height="30" rx="4" fill={i<3?C.s2:c+"18"} stroke={i<3?C.border:c}/><text x={60+i*115} y="24" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: runtime overhead for weight prep</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: zero runtime prep (all precomputed)</text></>,
      },
    ]
  },
  { id:6, t:"Model Optimization", c:C.c6, ico:"✂️", d:"Reduce the CNN workload itself. Use only if model changes are permitted.",
    items:[
      {n:"Pruning",w:"Remove low-magnitude weights. Hardware must exploit sparsity for actual benefit.",h:"Zero out small weights. If RTL can skip zero-weight MACs (gated multipliers or sparse indexing), computation decreases.",ex:"Prune ~30% of Conv1 weights. With sparsity-aware RTL, skip those MACs entirely.",rtl:"Weight ROM stores sparse format. MAC array checks `if(weight!=0)` before computing. Or use compressed sparse storage.",lat:"↓",thr:"↑",fmax:"—",area:"↑",membw:"↓",simrt:"—",diff:"High",
        bef_lbl:"Dense weights (all computed)",aft_lbl:"Sparse weights (skip zeros)",
        bef_svg:(c)=><>{Array.from({length:25},(_, i)=><rect key={i} x={60+(i%5)*40} y={5+Math.floor(i/5)*14} width="36" height="12" rx="2" fill={c+"15"} stroke={c} strokeWidth="0.5"/>)}</>,
        aft_svg:(c)=><>{Array.from({length:25},(_, i)=>{const pruned=(i%3===0);return<rect key={i} x={60+(i%5)*40} y={5+Math.floor(i/5)*14} width="36" height="12" rx="2" fill={pruned?"none":c+"15"} stroke={pruned?C.red:c} strokeWidth={pruned?1:0.5} strokeDasharray={pruned?"3":""}/>})}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 25 MACs per pixel (all weights)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: ~17 MACs per pixel (30% pruned)</text><text x="10" y="55" fill={C.t2} fontSize="8" fontFamily={mono}>Requires sparsity-aware hardware!</text></>,
      },
      {n:"Structured Pruning",w:"Remove entire filters or channels to directly shrink hardware dimensions.",h:"Remove complete filters from Conv1/Conv2. Hardware dimensions shrink proportionally — no sparse logic needed.",ex:"Conv1: remove 2 of 6 filters → only 4 filter engines needed. Hardware shrinks by 33%.",rtl:"Change parameter: `parameter NUM_FILTERS = 4;` instead of 6. Retrain model to recover accuracy.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↓↓",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"6 filters (full model)",aft_lbl:"4 filters (structured pruning)",
        bef_svg:(c)=><>{[0,1,2,3,4,5].map(i=><rect key={i} x={15+i*77} y="10" width="67" height="25" rx="3" fill={c+"15"} stroke={c}/>)}{[0,1,2,3,4,5].map(i=><text key={i} x={48+i*77} y="26" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>F{i}</text>)}</>,
        aft_svg:(c)=><>{[0,1,2,3].map(i=><g key={i}><rect x={50+i*100} y="10" width="80" height="25" rx="3" fill={c+"20"} stroke={c}/><text x={90+i*100} y="26" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>F{i}</text></g>)}<text x="240" y="55" textAnchor="middle" fill={c} fontSize="9" fontFamily={mono}>33% less hardware</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 6 filter engines</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 4 filter engines (direct HW reduction)</text></>,
      },
      {n:"Filter Reduction",w:"Use fewer convolution filters if accuracy permits after retraining.",h:"Reduce Conv1 from 6 to 4 filters, Conv2 from 16 to 8. Directly reduces multiply count and memory.",ex:"Conv2: 16→8 filters = 50% less convolution work. Retrain and verify accuracy > threshold.",rtl:"Change generate parameters. Reduce weight ROM sizes. Update all downstream dimension parameters.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↓↓",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"Full filter count",aft_lbl:"Reduced filter count",
        bef_svg:(c)=><><text x="220" y="25" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>Conv2: 16 filters</text></>,
        aft_svg:(c)=><><text x="220" y="25" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>Conv2: 8 filters (50% reduction)</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 16 × 6 × 25 = 2400 MACs/pixel</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 8 × 6 × 25 = 1200 MACs/pixel</text></>,
      },
      {n:"Neuron Reduction",w:"Reduce FC layer sizes to decrease multiply count.",h:"FC1: 120→64 neurons, FC2: 84→32. Fewer MAC operations in the fully-connected stages.",ex:"FC1 with 64 neurons instead of 120: 400×64=25,600 MACs instead of 400×120=48,000 MACs.",rtl:"Change FC weight ROM dimensions and MAC array sizes. `parameter FC1_NEURONS = 64;`",lat:"↓",thr:"↑",fmax:"—",area:"↓",membw:"↓",simrt:"—",diff:"Medium",
        bef_lbl:"FC1: 120, FC2: 84",aft_lbl:"FC1: 64, FC2: 32",
        bef_svg:(c)=><><text x="220" y="25" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>FC1:120 → FC2:84 → Out:10</text></>,
        aft_svg:(c)=><><text x="220" y="25" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>FC1:64 → FC2:32 → Out:10</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 48000 + 8400 = 56400 FC MACs</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 25600 + 2560 = 28160 FC MACs</text></>,
      },
      {n:"Model Quantization",w:"Convert floating-point weights/activations to compact integer formats (INT8, INT4).",h:"Quantize trained FP32 model to INT8. 4× smaller weights, 4× smaller multipliers, potentially faster.",ex:"All weights: FP32→INT8. Activations: FP32→INT8 with INT32 accumulator. Verify accuracy against golden model.",rtl:"Replace FP32 data types with `logic signed [7:0]`. Accumulator: `logic signed [31:0]`.",lat:"↓",thr:"↑",fmax:"↑",area:"↓↓",membw:"↓↓",simrt:"↓",diff:"Medium",
        bef_lbl:"FP32 model",aft_lbl:"INT8 model",
        bef_svg:(c)=><><rect x="100" y="5" width="240" height="30" rx="4" fill={C.red+"12"} stroke={C.red+"40"}/><text x="220" y="24" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>32-bit float per weight</text></>,
        aft_svg:(c)=><><rect x="140" y="5" width="160" height="30" rx="4" fill={c+"20"} stroke={c}/><text x="220" y="24" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>8-bit int per weight</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>FP32: 4 bytes/weight, big multiplier</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>INT8: 1 byte/weight, tiny multiplier</text></>,
      },
      {n:"Removing Layers/Ops",w:"Remove entire layers or operations proven unnecessary for the target accuracy.",h:"Only remove operations after validating the modified model. Each removal must be verified against the golden reference.",ex:"If a pooling layer can be replaced by strided convolution, or if an FC layer can be removed, validate first.",rtl:"Remove the module instantiation and update all downstream connections and dimension parameters.",lat:"↓↓",thr:"↑↑",fmax:"—",area:"↓↓",membw:"↓↓",simrt:"↓",diff:"High",
        bef_lbl:"Full network",aft_lbl:"Simplified network",
        bef_svg:(c)=><>{["C1","P1","C2","P2","FC1","FC2"].map((s,i)=><g key={i}><rect x={10+i*77} y="10" width="65" height="22" rx="3" fill={C.s1} stroke={C.border}/><text x={42+i*77} y="25" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s}</text></g>)}</>,
        aft_svg:(c)=><>{["C1","P1","C2","P2","FC2"].map((s,i)=><g key={i}><rect x={30+i*90} y="10" width="75" height="22" rx="3" fill={c+"18"} stroke={c}/><text x={67+i*90} y="25" textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s}</text></g>)}<text x="430" y="25" fill={c} fontSize="9" fontFamily={mono}>-1 layer</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 6 layers, full computation</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 5 layers (validated accuracy)</text></>,
      },
    ]
  },
  { id:7, t:"Simulation Optimization", c:C.c7, ico:"🖥️", d:"Reduce simulator wall-clock time without changing RTL behavior.",
    items:[
      {n:"Waveform Optimization",w:"Disable or limit VCD/FST waveform dumping during benchmark runs.",h:"Full waveform dumping can slow simulation 10-50×. Dump only during debug; disable for performance measurements.",ex:"Use `ifdef DEBUG $dumpfile(\"wave.vcd\"); $dumpvars(0, tb); `endif — controlled by compile flag.",rtl:"Conditional compilation: `+define+DEBUG` enables dumping. Benchmark runs omit the flag.",lat:"—",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓↓↓",diff:"Very Low",
        bef_lbl:"Full VCD dump (slow!)",aft_lbl:"No dump during benchmark",
        bef_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="4" fill={C.red+"12"} stroke={C.red+"40"}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>$dumpvars(0, tb) — dumps everything 🐢</text></>,
        aft_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>No waveform dump 🚀 (10-50× faster)</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 30 min simulation (with dump)</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 1 min simulation (no dump)</text></>,
      },
      {n:"Testbench Optimization",w:"Minimize display statements, file I/O, and per-cycle checking overhead.",h:"Replace per-activation $display with summary statistics. Report only mismatch count and max error at end.",ex:"Replace `$display(\"act[%d]=%d\",i,act)` with end-of-test `$display(\"mismatches=%d max_err=%d\",cnt,maxe)`.",rtl:"Use counters for mismatch tracking. Only display summary. Use `$fwrite` to file if detailed log needed.",lat:"—",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓↓",diff:"Very Low",
        bef_lbl:"Print every activation",aft_lbl:"Print summary only",
        bef_svg:(c)=><><rect x="60" y="5" width="320" height="35" rx="4" fill={C.red+"12"} stroke={C.red+"40"}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>$display × 1000s of activations</text></>,
        aft_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="4" fill={c+"18"} stroke={c}/><text x="220" y="27" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>$display: mismatches=0, maxErr=2</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: I/O bottleneck from excessive printing</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: minimal I/O, much faster execution</text></>,
      },
      {n:"Input/Weight Pre-generation",w:"Generate all test inputs, weights, and golden outputs offline before simulation.",h:"Python script creates .hex files for inputs, weights, quantization params, and expected outputs. RTL uses $readmemh.",ex:"generate_test_data.py → inputs.hex, weights_conv1.hex, golden_conv1.hex, golden_fc2.hex, golden_argmax.hex",rtl:"`initial begin $readmemh(\"inputs.hex\", input_mem); $readmemh(\"golden.hex\", golden_mem); end`",lat:"—",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓↓",diff:"Low",
        bef_lbl:"Runtime data generation",aft_lbl:"Pre-generated .hex files",
        bef_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Testbench generates data at runtime</text></>,
        aft_svg:(c)=><>{["inputs.hex","weights.hex","golden.hex"].map((s,i)=><g key={i}><rect x={20+i*155} y="5" width="135" height="30" rx="4" fill={c+"15"} stroke={c}/><text x={87+i*155} y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s}</text></g>)}</>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: runtime prep overhead</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: instant $readmemh load</text></>,
      },
      {n:"Golden-Model Comparison",w:"Automate per-layer RTL vs reference comparison with mismatch counting.",h:"At each layer checkpoint, compare RTL output with golden reference. Report mismatch count and maximum error.",ex:"Compare conv1_out vs golden_conv1 pixel-by-pixel. Report: `Conv1: 0 mismatches, max_err=0`. Then conv2, fc1, etc.",rtl:"`if(rtl_out !== golden[addr]) begin mismatch_cnt++; if(diff>max_err) max_err=diff; end`",lat:"—",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓",diff:"Low",
        bef_lbl:"Manual waveform inspection",aft_lbl:"Automated comparison",
        bef_svg:(c)=><><rect x="80" y="5" width="280" height="35" rx="4" fill={C.s1} stroke={C.border}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Manually inspect waveforms 👀</text></>,
        aft_svg:(c)=><><rect x="60" y="5" width="150" height="30" rx="4" fill={c+"15"} stroke={c}/><text x="135" y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>RTL Output</text><text x="245" y="24" textAnchor="middle" fill={c} fontSize="14" fontFamily={mono}>≟</text><rect x="270" y="5" width="150" height="30" rx="4" fill={C.c6+"15"} stroke={C.c6}/><text x="345" y="24" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Golden Ref</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: hours of manual checking</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: automated, instant pass/fail</text></>,
      },
      {n:"Signal Reduction",w:"Expose only essential debug/checkpoint signals instead of thousands of internals.",h:"Fewer observed signals = fewer simulator events = faster simulation. Only expose layer outputs and control signals.",ex:"Expose: conv1_done, conv1_out, pool1_done, pool1_out, etc. Hide: internal MAC values, FSM sub-states, counters.",rtl:"Use `(* keep *)` only on checkpoint signals. Remove unnecessary `output` ports from sub-modules.",lat:"—",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓↓",diff:"Very Low",
        bef_lbl:"5000+ observed signals",aft_lbl:"~20 checkpoint signals",
        bef_svg:(c)=><><text x="220" y="25" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>5000+ internal signals monitored</text></>,
        aft_svg:(c)=><><text x="220" y="25" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono}>~20 checkpoint signals only</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Fewer events = less simulator work</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>Can be 2-5× sim speedup</text></>,
      },
      {n:"Regression Optimization",w:"Use a testing pyramid: fast unit tests for iteration, full regression for milestones.",h:"Don't run full dataset regression on every change. Use layer-level unit tests for fast iteration.",ex:"Daily: unit-test Conv1 with 5 inputs. Weekly: integration test full pipeline. Milestone: full 10K test set.",rtl:"Separate testbenches: tb_conv1.sv, tb_pool1.sv, tb_lenet_full.sv. Makefile targets for each level.",lat:"—",thr:"—",fmax:"—",area:"—",membw:"—",simrt:"↓↓↓",diff:"Low",
        bef_lbl:"Full regression every time",aft_lbl:"Test pyramid strategy",
        bef_svg:(c)=><><rect x="100" y="5" width="240" height="35" rx="4" fill={C.red+"12"} stroke={C.red+"40"}/><text x="220" y="27" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>10K images × full pipeline (hours!)</text></>,
        aft_svg:(c)=><><text x="220" y="15" textAnchor="middle" fill={C.t3} fontSize="8" fontFamily={mono}>Full (milestone)</text><text x="220" y="28" textAnchor="middle" fill={C.t2} fontSize="9" fontFamily={mono}>Integration (weekly)</text><text x="220" y="41" textAnchor="middle" fill={c} fontSize="10" fontFamily={mono} fontWeight="bold">Unit Tests (per-change) ⚡</text></>,
        cyc_svg:(c)=><><text x="10" y="15" fill={C.t3} fontSize="9" fontFamily={mono}>Before: 2 hours per change</text><text x="10" y="35" fill={c} fontSize="9" fontFamily={mono}>After: 30 seconds per change (unit test)</text></>,
      },
    ]
  },
];

const allItems = cats.flatMap(cat => cat.items.map(item => ({...item, cat})));

// ═══════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════
const css = `
* { box-sizing:border-box; margin:0; padding:0; }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
.fade { animation: fadeIn 0.25s ease; }
.btn { background:none; border:1px solid ${C.border}; color:${C.t2}; padding:5px 12px; border-radius:6px; cursor:pointer; font-family:${mono}; font-size:12px; transition:all 0.15s; }
.btn:hover { border-color:${C.borderH}; color:${C.t1}; }
.card { background:${C.s1}; border:1px solid ${C.border}; border-radius:10px; padding:14px; transition:all 0.15s; cursor:pointer; text-align:left; width:100%; }
.card:hover { border-color:${C.borderH}; background:${C.s2}; }
.section { background:${C.s1}; border:1px solid ${C.border}; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
.tag { font-size:10px; padding:2px 7px; border-radius:4px; font-family:${mono}; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:${C.bg}; }
::-webkit-scrollbar-thumb { background:${C.border}; border-radius:3px; }
`;

// ═══════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════

function Nav({onNav, current}) {
  const tabs = [
    {id:"cover",l:"Home"},{id:"arch",l:"Architecture"},{id:"conv",l:"Convolution"},{id:"map",l:"Mind Map"},
    {id:"roadmap",l:"Roadmap"},{id:"matrix",l:"Matrix"},{id:"checklist",l:"Checklist"}
  ];
  return (
    <div style={{display:"flex",gap:4,flexWrap:"wrap",marginBottom:16,borderBottom:`1px solid ${C.border}`,paddingBottom:10}}>
      {tabs.map(t=>(
        <button key={t.id} onClick={()=>onNav(t.id)} className="btn" style={{
          borderColor:current===t.id?C.c1:C.border,
          color:current===t.id?C.c1:C.t3,
          fontSize:11,padding:"4px 10px",
        }}>{t.l}</button>
      ))}
    </div>
  );
}

function CoverPage({onNav}) {
  return <div className="fade" style={{textAlign:"center",padding:"40px 0"}}>
    <div style={{fontSize:11,color:C.t3,fontFamily:mono,letterSpacing:3,marginBottom:8}}>INTERACTIVE TECHNICAL REFERENCE</div>
    <h1 style={{fontSize:28,color:C.t1,fontFamily:sans,fontWeight:800,lineHeight:1.2}}>LeNet RTL Performance<br/>Optimization</h1>
    <p style={{color:C.t2,fontSize:13,margin:"16px auto 24px",maxWidth:460,lineHeight:1.6}}>A visual engineering guide for converting a LeNet golden model to optimized RTL. Covers 47 techniques across 7 categories with before/after diagrams, timing analysis, and RTL implementation views.</p>
    <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8,maxWidth:420,margin:"0 auto 24px"}}>
      {[{l:"Latency",d:"Cycles per inference",c:C.c1},{l:"Throughput",d:"Images/sec steady state",c:C.c2},{l:"Sim Runtime",d:"Wall-clock sim time",c:C.c3}].map((m,i)=>(
        <div key={i} style={{background:C.s1,border:`1px solid ${C.border}`,borderTop:`2px solid ${m.c}`,borderRadius:8,padding:"8px 6px",textAlign:"center"}}>
          <div style={{fontSize:12,color:m.c,fontWeight:700,fontFamily:mono}}>{m.l}</div>
          <div style={{fontSize:10,color:C.t3,marginTop:2}}>{m.d}</div>
        </div>
      ))}
    </div>
    <button onClick={()=>onNav("arch")} className="btn" style={{borderColor:C.c1,color:C.c1,padding:"8px 20px",fontSize:13}}>Start Exploring →</button>
  </div>;
}

function ArchPage() {
  const layers = [
    {n:"Input",dim:"32×32×1",ops:"-",color:C.t3},
    {n:"Conv1",dim:"28×28×6",ops:"5×5×1×6 = 150 MACs/pixel\n× 784 pixels = 117,600",color:C.c1},
    {n:"Pool1",dim:"14×14×6",ops:"2×2 max/avg\n× 1,176 = 1,176 compare",color:C.c2},
    {n:"Conv2",dim:"10×10×16",ops:"5×5×6×16 = 2,400 MACs/pixel\n× 100 pixels = 240,000",color:C.c3},
    {n:"Pool2",dim:"5×5×16",ops:"2×2 max/avg\n× 400 = 400 compare",color:C.c4},
    {n:"FC1",dim:"120",ops:"400×120 = 48,000 MACs",color:C.c5},
    {n:"FC2",dim:"84",ops:"120×84 = 10,080 MACs",color:C.c6},
    {n:"Output",dim:"10",ops:"Argmax of 10 scores",color:C.c7},
  ];
  return <div className="fade">
    <h2 style={{fontSize:20,color:C.t1,marginBottom:4}}>LeNet Architecture</h2>
    <p style={{color:C.t2,fontSize:12,marginBottom:16}}>Standard LeNet-5 with dimensions, operation counts, and applicable optimizations per layer.</p>
    <div style={{display:"grid",gap:6}}>
      {layers.map((l,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:10,background:C.s1,border:`1px solid ${C.border}`,borderLeft:`3px solid ${l.color}`,borderRadius:8,padding:"10px 12px"}}>
          <div style={{width:56,textAlign:"center"}}>
            <div style={{fontSize:13,color:l.color,fontWeight:700,fontFamily:mono}}>{l.n}</div>
          </div>
          <div style={{flex:1}}>
            <div style={{fontSize:11,color:C.t1,fontFamily:mono}}>{l.dim}</div>
            <div style={{fontSize:10,color:C.t3,fontFamily:mono,whiteSpace:"pre-line",lineHeight:1.4}}>{l.ops}</div>
          </div>
          {i>0&&i<7&&<div style={{fontSize:16}}>→</div>}
        </div>
      ))}
    </div>
    <div className="section" style={{marginTop:12}}>
      <div style={{fontSize:11,color:C.c4,fontFamily:mono,marginBottom:4}}>TOTAL OPERATIONS</div>
      <div style={{fontSize:12,color:C.t1}}>Conv: ~357,600 MACs • FC: ~58,080 MACs • Total: ~415,680 MACs</div>
      <div style={{fontSize:11,color:C.t3,marginTop:4}}>Conv layers dominate computation (86%). Optimize convolution first.</div>
    </div>
  </div>;
}

function ConvPage() {
  return <div className="fade">
    <h2 style={{fontSize:20,color:C.t1,marginBottom:4}}>Convolution → RTL Mapping</h2>
    <p style={{color:C.t2,fontSize:12,marginBottom:16}}>How a 5×5 convolution maps to hardware, and how each optimization modifies it.</p>
    <div className="section">
      <div style={{fontSize:11,color:C.c1,fontFamily:mono,marginBottom:6}}>BASELINE PIPELINE</div>
      <svg viewBox="0 0 480 180" style={{width:"100%",maxWidth:480}}>
        <rect width="480" height="180" rx="8" fill={C.bg}/>
        {[
          {l:"Feature Map",y:10,w:100},{l:"5×5 Window",y:40,w:100},{l:"25 Multipliers",y:70,w:120},
          {l:"Reduction Tree",y:100,w:120},{l:"+ Bias",y:130,w:80},{l:"ReLU → Output",y:155,w:110}
        ].map((s,i)=><g key={i}>
          <rect x={190-s.w/2} y={s.y} width={s.w} height="22" rx="4" fill={C.c1+"15"} stroke={C.c1} strokeWidth="0.8"/>
          <text x="240" y={s.y+15} textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>{s.l}</text>
          {i<5&&<line x1="240" y1={s.y+22} x2="240" y2={s.y+30} stroke={C.c1} strokeWidth="0.8"/>}
        </g>)}
        <text x="380" y="80" fill={C.t3} fontSize="9" fontFamily={mono}>× per pixel</text>
        <text x="380" y="95" fill={C.t3} fontSize="9" fontFamily={mono}>× per filter</text>
        <text x="380" y="110" fill={C.t3} fontSize="9" fontFamily={mono}>× per channel</text>
      </svg>
    </div>
    <div className="section">
      <div style={{fontSize:11,color:C.c3,fontFamily:mono,marginBottom:6}}>OPTIMIZATION STACK (cumulative improvements)</div>
      {["Operator Parallel → 25 MACs in 1 cycle","Reduction Tree → O(log₂25) = 5 levels","Pipeline Regs → higher Fmax","Line Buffers → no repeated reads","Sliding Window → 80% data reuse","Streaming → direct to next stage","Layer Fusion → Conv+ReLU+Pool fused"].map((s,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"4px 0",borderBottom:i<6?`1px solid ${C.border}`:"none"}}>
          <span style={{fontSize:10,color:cats[Math.min(i,6)].c,fontFamily:mono,fontWeight:700,width:16,textAlign:"right"}}>{i+1}</span>
          <span style={{fontSize:11,color:C.t1}}>{s}</span>
        </div>
      ))}
    </div>
  </div>;
}

function MindMapPage({onCat, onItem}) {
  return <div className="fade">
    <h2 style={{fontSize:20,color:C.t1,marginBottom:4}}>Master Mind Map</h2>
    <p style={{color:C.t2,fontSize:12,marginBottom:16}}>Click any category or technique to explore its details.</p>
    {cats.map(cat=>(
      <div key={cat.id} style={{marginBottom:10}}>
        <button className="card" onClick={()=>onCat(cat)} style={{borderLeft:`3px solid ${cat.c}`}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <span style={{fontSize:18}}>{cat.ico}</span>
            <div style={{flex:1}}>
              <div style={{display:"flex",alignItems:"center",gap:6}}>
                <span style={{fontSize:11,color:cat.c,fontFamily:mono,fontWeight:700}}>{cat.id}.</span>
                <span style={{fontSize:14,color:C.t1,fontWeight:600}}>{cat.t}</span>
                <span style={{fontSize:10,color:C.t3,fontFamily:mono,marginLeft:"auto"}}>{cat.items.length}</span>
              </div>
              <div style={{fontSize:11,color:C.t3,marginTop:2}}>{cat.d}</div>
            </div>
            <span style={{color:cat.c,fontSize:18}}>›</span>
          </div>
          <div style={{display:"flex",gap:4,flexWrap:"wrap",marginTop:8}}>
            {cat.items.map((item,j)=>(
              <span key={j} className="tag" style={{color:cat.c,background:cat.c+"10",border:`1px solid ${cat.c}22`}}
                onClick={(e)=>{e.stopPropagation();onItem(cat,item);}}>
                {item.n.length > 20 ? item.n.split(" ").slice(0,2).join(" ") : item.n}
              </span>
            ))}
          </div>
        </button>
      </div>
    ))}
  </div>;
}

function CategoryPage({cat, onItem, onBack}) {
  return <div className="fade">
    <button className="btn" onClick={onBack} style={{marginBottom:14}}>← Mind Map</button>
    <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
      <span style={{fontSize:24}}>{cat.ico}</span>
      <h2 style={{fontSize:22,color:C.t1}}>{cat.t}</h2>
    </div>
    <p style={{color:C.t2,fontSize:13,marginBottom:16}}>{cat.d}</p>
    <div style={{display:"grid",gap:8}}>
      {cat.items.map((item,i)=>(
        <button key={i} className="card" onClick={()=>onItem(item)} style={{borderLeft:`3px solid ${cat.c}`}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <span style={{width:26,height:26,borderRadius:7,background:cat.c+"12",border:`1px solid ${cat.c}30`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,color:cat.c,fontWeight:700,fontFamily:mono,flexShrink:0}}>{i+1}</span>
            <div style={{flex:1,minWidth:0}}>
              <div style={{fontSize:13,color:C.t1,fontWeight:600}}>{item.n}</div>
              <div style={{fontSize:11,color:C.t3,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{item.w}</div>
            </div>
            {/* Performance indicators */}
            <div style={{display:"flex",gap:4,flexShrink:0}}>
              {item.lat.includes("↓")&&<span className="tag" style={{color:C.green,background:C.green+"10"}}>Lat{item.lat}</span>}
              {item.thr.includes("↑")&&<span className="tag" style={{color:C.c2,background:C.c2+"10"}}>Thr{item.thr}</span>}
            </div>
            <span style={{color:cat.c,fontSize:16}}>›</span>
          </div>
        </button>
      ))}
    </div>
  </div>;
}

function DetailPage({cat, item, onBack, onBackMap}) {
  const c = cat.c;
  const metrics = [
    {l:"Latency",v:item.lat},{l:"Throughput",v:item.thr},{l:"Fmax",v:item.fmax},
    {l:"Area",v:item.area},{l:"Mem BW",v:item.membw},{l:"Sim Runtime",v:item.simrt}
  ];
  const mColor = (v) => v.includes("↑↑")?"#22c55e":v.includes("↑")?"#86efac":v.includes("↓↓")?"#ef4444":v.includes("↓")?"#fca5a5":C.t3;
  const mColorInv = (l,v) => {
    // For latency/area/membw/simrt, ↓ is good. For throughput/fmax, ↑ is good
    const goodDown = ["Latency","Area","Mem BW","Sim Runtime"].includes(l);
    if(v==="—") return C.t3;
    if(goodDown) return v.includes("↓")?"#22c55e":"#ef4444";
    return v.includes("↑")?"#22c55e":"#ef4444";
  };
  
  return <div className="fade">
    <div style={{display:"flex",gap:6,marginBottom:14,flexWrap:"wrap"}}>
      <button className="btn" onClick={onBackMap}>← Mind Map</button>
      <button className="btn" onClick={onBack} style={{borderColor:c+"40",color:c}}>← {cat.t}</button>
    </div>
    <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:16}}>
      <div style={{width:40,height:40,borderRadius:10,background:c+"15",border:`2px solid ${c}`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:18}}>{cat.ico}</div>
      <div>
        <div style={{fontSize:11,color:c,fontFamily:mono,letterSpacing:1}}>{cat.t.toUpperCase()}</div>
        <h2 style={{fontSize:20,color:C.t1,fontWeight:700}}>{item.n}</h2>
      </div>
    </div>

    {/* A. What is it */}
    <div className="section" style={{borderLeft:`3px solid ${c}`}}>
      <div style={{fontSize:11,color:c,fontFamily:mono,marginBottom:4}}>A. WHAT IS IT?</div>
      <div style={{fontSize:13,color:C.t1,lineHeight:1.6}}>{item.w}</div>
      <div style={{fontSize:12,color:C.t2,marginTop:6,lineHeight:1.5}}>{item.h}</div>
    </div>

    {/* B. Before vs After */}
    <div className="section" style={{borderLeft:`3px solid ${c}`}}>
      <div style={{fontSize:11,color:c,fontFamily:mono,marginBottom:8}}>B. BEFORE vs AFTER</div>
      <div style={{marginBottom:8}}>
        <div style={{fontSize:10,color:C.t3,fontFamily:mono,marginBottom:4}}>BEFORE: {item.bef_lbl}</div>
        <svg viewBox="0 0 480 80" style={{width:"100%",maxWidth:480}}>
          <rect width="480" height="80" rx="6" fill={C.bg}/>
          {item.bef_svg(c)}
        </svg>
      </div>
      <div>
        <div style={{fontSize:10,color:c,fontFamily:mono,marginBottom:4}}>AFTER: {item.aft_lbl}</div>
        <svg viewBox="0 0 480 70" style={{width:"100%",maxWidth:480}}>
          <rect width="480" height="70" rx="6" fill={C.bg}/>
          {item.aft_svg(c)}
        </svg>
      </div>
    </div>

    {/* C. Timing / Cycle Diagram */}
    <div className="section" style={{borderLeft:`3px solid ${c}`}}>
      <div style={{fontSize:11,color:c,fontFamily:mono,marginBottom:6}}>C. TIMING / CYCLE IMPACT</div>
      <svg viewBox="0 0 480 65" style={{width:"100%",maxWidth:480}}>
        <rect width="480" height="65" rx="6" fill={C.bg}/>
        {item.cyc_svg(c)}
      </svg>
    </div>

    {/* D. LeNet Example */}
    <div className="section" style={{borderLeft:`3px solid ${c}`}}>
      <div style={{fontSize:11,color:c,fontFamily:mono,marginBottom:4}}>D. LeNet-SPECIFIC EXAMPLE</div>
      <div style={{fontSize:12,color:C.t1,lineHeight:1.6}}>{item.ex}</div>
    </div>

    {/* E. RTL Implementation */}
    <div className="section" style={{borderLeft:`3px solid ${c}`}}>
      <div style={{fontSize:11,color:c,fontFamily:mono,marginBottom:4}}>E. RTL IMPLEMENTATION</div>
      <div style={{fontSize:11,color:C.t2,lineHeight:1.5,fontFamily:mono,background:C.bg,padding:10,borderRadius:6,whiteSpace:"pre-wrap",wordBreak:"break-word"}}>{item.rtl}</div>
    </div>

    {/* F. Performance Effect */}
    <div className="section" style={{borderLeft:`3px solid ${c}`}}>
      <div style={{fontSize:11,color:c,fontFamily:mono,marginBottom:8}}>F. PERFORMANCE EFFECT</div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:4}}>
        {metrics.map((m,i)=>(
          <div key={i} style={{background:C.bg,borderRadius:6,padding:"6px 8px",textAlign:"center"}}>
            <div style={{fontSize:9,color:C.t3,fontFamily:mono}}>{m.l}</div>
            <div style={{fontSize:16,color:mColorInv(m.l,m.v),fontWeight:700,fontFamily:mono}}>{m.v}</div>
          </div>
        ))}
      </div>
      <div style={{fontSize:10,color:C.t3,marginTop:6,fontFamily:mono}}>Difficulty: {item.diff}</div>
    </div>
  </div>;
}

function RoadmapPage() {
  return <div className="fade">
    <h2 style={{fontSize:20,color:C.t1,marginBottom:4}}>Optimization Roadmap</h2>
    <p style={{color:C.t2,fontSize:12,marginBottom:16}}>Follow the bottleneck. Measure, optimize, re-measure.</p>
    <svg viewBox="0 0 480 320" style={{width:"100%",maxWidth:480}}>
      <rect width="480" height="320" rx="8" fill={C.bg}/>
      {/* Flow */}
      <rect x="160" y="10" width="160" height="28" rx="6" fill={C.c1+"20"} stroke={C.c1}/><text x="240" y="28" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>Find Bottleneck</text>
      <line x1="200" y1="38" x2="120" y2="58" stroke={C.t3}/><line x1="280" y1="38" x2="360" y2="58" stroke={C.t3}/>
      <rect x="40" y="58" width="160" height="24" rx="5" fill={C.c4+"15"} stroke={C.c4}/><text x="120" y="74" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Compute Bound?</text>
      <rect x="280" y="58" width="160" height="24" rx="5" fill={C.c3+"15"} stroke={C.c3}/><text x="360" y="74" textAnchor="middle" fill={C.t1} fontSize="9" fontFamily={mono}>Memory Bound?</text>
      
      {[{l:"→ Parallelism",y:92,c:C.c1},{l:"→ Pipelining",y:116,c:C.c2},{l:"→ Arithmetic Opt",y:140,c:C.c4}].map((s,i)=>
        <g key={i}><rect x="50" y={s.y} width="140" height="20" rx="4" fill={s.c+"12"} stroke={s.c} strokeWidth="0.5"/><text x="120" y={s.y+14} textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s.l}</text></g>
      )}
      {[{l:"→ Data Reuse",y:92,c:C.c3},{l:"→ Banking",y:116,c:C.c3},{l:"→ Streaming",y:140,c:C.c3}].map((s,i)=>
        <g key={i}><rect x="290" y={s.y} width="140" height="20" rx="4" fill={s.c+"12"} stroke={s.c} strokeWidth="0.5"/><text x="360" y={s.y+14} textAnchor="middle" fill={C.t1} fontSize="8" fontFamily={mono}>{s.l}</text></g>
      )}

      <line x1="120" y1="160" x2="240" y2="180" stroke={C.t3}/><line x1="360" y1="160" x2="240" y2="180" stroke={C.t3}/>
      <rect x="140" y="180" width="200" height="28" rx="6" fill={C.c5+"20"} stroke={C.c5}/><text x="240" y="198" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>Measure Again</text>
      <line x1="240" y1="208" x2="240" y2="225" stroke={C.t3}/>
      <rect x="140" y="225" width="200" height="28" rx="6" fill={C.c6+"20"} stroke={C.c6}/><text x="240" y="243" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>Keep or Reject?</text>
      <line x1="240" y1="253" x2="240" y2="270" stroke={C.t3}/>
      <rect x="140" y="270" width="200" height="28" rx="6" fill={C.c7+"20"} stroke={C.c7}/><text x="240" y="288" textAnchor="middle" fill={C.t1} fontSize="10" fontFamily={mono}>Next Bottleneck ↻</text>
      <path d="M340 284 Q420 284 420 198 Q420 140 350 140" stroke={C.t3} fill="none" strokeDasharray="4"/>
    </svg>
    <div className="section" style={{marginTop:10}}>
      <div style={{fontSize:11,color:C.c1,fontFamily:mono,marginBottom:6}}>PRIORITY ORDER</div>
      {cats.map((cat,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"3px 0"}}>
          <span style={{fontSize:11,color:cat.c,fontWeight:700,fontFamily:mono,width:16,textAlign:"right"}}>{i+1}</span>
          <span style={{fontSize:12,color:C.t1}}>{cat.t}</span>
          <span style={{fontSize:10,color:C.t3,fontFamily:mono,marginLeft:"auto"}}>{cat.d.split(".")[0]}</span>
        </div>
      ))}
    </div>
  </div>;
}

function MatrixPage() {
  return <div className="fade">
    <h2 style={{fontSize:20,color:C.t1,marginBottom:4}}>Technique Comparison Matrix</h2>
    <p style={{color:C.t2,fontSize:12,marginBottom:10}}>All 47 techniques with performance impact indicators.</p>
    <div style={{fontSize:10,color:C.t3,fontFamily:mono,marginBottom:8}}>
      <span style={{color:C.green}}>↑/↑↑</span>=improves <span style={{color:C.red}}>↓/↓↓</span>=increases <span>—</span>=unchanged • For Latency/Area/MemBW/SimRT: ↓ is good. For Throughput/Fmax: ↑ is good.
    </div>
    <div style={{overflowX:"auto"}}>
      <table style={{width:"100%",borderCollapse:"collapse",fontSize:10,fontFamily:mono}}>
        <thead>
          <tr style={{background:C.s2}}>
            {["Technique","Cat","Lat","Thr","Fmax","Area","BW","Sim","Diff"].map((h,i)=>(
              <th key={i} style={{padding:"6px 4px",textAlign:"left",color:C.t3,borderBottom:`1px solid ${C.border}`,fontSize:9,whiteSpace:"nowrap"}}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {allItems.map((item,i)=>{
            const goodDown = (label, v) => {
              if(v==="—") return C.t3;
              const isGoodDown = ["Lat","Area","BW","Sim"].includes(label);
              if(isGoodDown) return v.includes("↓")?C.green:C.red;
              return v.includes("↑")?C.green:C.red;
            };
            return <tr key={i} style={{background:i%2===0?C.s1:"transparent",borderBottom:`1px solid ${C.border}22`}}>
              <td style={{padding:"4px",color:C.t1,maxWidth:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{item.n}</td>
              <td style={{padding:"4px",color:item.cat.c}}>{item.cat.id}</td>
              {[{l:"Lat",v:item.lat},{l:"Thr",v:item.thr},{l:"Fmax",v:item.fmax},{l:"Area",v:item.area},{l:"BW",v:item.membw},{l:"Sim",v:item.simrt}].map((m,j)=>(
                <td key={j} style={{padding:"4px",color:goodDown(m.l,m.v),textAlign:"center"}}>{m.v}</td>
              ))}
              <td style={{padding:"4px",color:C.t3}}>{item.diff}</td>
            </tr>;
          })}
        </tbody>
      </table>
    </div>
  </div>;
}

function ChecklistPage() {
  const [checks, setChecks] = useState({});
  const items = [
    "Establish golden-model baseline",
    "Freeze numerical format (INT8/FP32/etc.)",
    "Verify every RTL layer independently",
    "Measure baseline latency (cycles)",
    "Measure baseline throughput (images/sec)",
    "Identify critical path (synthesis report)",
    "Identify memory bottleneck (stall analysis)",
    "Apply parallelism (MACs, filters, channels)",
    "Apply pipelining (operator, layer, deep)",
    "Apply data reuse (line buffers, sliding window)",
    "Optimize reduction trees (balanced adder tree)",
    "Optimize control/FSM (remove idle states)",
    "Evaluate layer fusion (Conv+ReLU+Pool)",
    "Evaluate quantization / model optimization",
    "Disable unnecessary waveform dumping",
    "Run regression tests",
    "Compare accuracy vs golden model",
    "Compare performance vs baseline",
    "Keep only optimizations that improve target metric",
  ];
  return <div className="fade">
    <h2 style={{fontSize:20,color:C.t1,marginBottom:4}}>Optimization Checklist</h2>
    <p style={{color:C.t2,fontSize:12,marginBottom:14}}>Track your progress through the optimization flow.</p>
    <div style={{display:"grid",gap:4}}>
      {items.map((item,i)=>(
        <button key={i} onClick={()=>setChecks(p=>({...p,[i]:!p[i]}))} style={{
          display:"flex",alignItems:"center",gap:10,
          background:checks[i]?C.green+"08":C.s1,
          border:`1px solid ${checks[i]?C.green+"30":C.border}`,
          borderRadius:8,padding:"8px 12px",cursor:"pointer",textAlign:"left",transition:"all 0.15s",
        }}>
          <div style={{width:20,height:20,borderRadius:5,border:`2px solid ${checks[i]?C.green:C.border}`,background:checks[i]?C.green:C.bg,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,color:"white",flexShrink:0}}>
            {checks[i]?"✓":""}
          </div>
          <span style={{fontSize:12,color:checks[i]?C.green:C.t1,textDecoration:checks[i]?"line-through":"none"}}>{item}</span>
        </button>
      ))}
    </div>
    <div style={{textAlign:"center",marginTop:14,fontSize:12,color:C.t3}}>
      {Object.values(checks).filter(Boolean).length}/{items.length} complete
    </div>
  </div>;
}

// ═══════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState("cover");
  const [selCat, setSelCat] = useState(null);
  const [selItem, setSelItem] = useState(null);
  const ref = useRef(null);
  const scroll = () => ref.current?.scrollTo(0,0);

  const nav = (p) => { setPage(p); scroll(); };
  const goCat = (cat) => { setSelCat(cat); setPage("category"); scroll(); };
  const goItem = (cat, item) => { setSelCat(cat); setSelItem(item); setPage("detail"); scroll(); };

  return (
    <div ref={ref} style={{background:C.bg,color:C.t1,fontFamily:sans,minHeight:"100vh",padding:"14px 14px",maxWidth:620,margin:"0 auto",overflowY:"auto"}}>
      <style>{css}</style>
      <Nav onNav={nav} current={page}/>
      
      {page==="cover"&&<CoverPage onNav={nav}/>}
      {page==="arch"&&<ArchPage/>}
      {page==="conv"&&<ConvPage/>}
      {page==="map"&&<MindMapPage onCat={goCat} onItem={goItem}/>}
      {page==="category"&&selCat&&<CategoryPage cat={selCat} onItem={(item)=>goItem(selCat,item)} onBack={()=>nav("map")}/>}
      {page==="detail"&&selCat&&selItem&&<DetailPage cat={selCat} item={selItem} onBack={()=>{setPage("category");scroll();}} onBackMap={()=>nav("map")}/>}
      {page==="roadmap"&&<RoadmapPage/>}
      {page==="matrix"&&<MatrixPage/>}
      {page==="checklist"&&<ChecklistPage/>}
    </div>
  );
}
