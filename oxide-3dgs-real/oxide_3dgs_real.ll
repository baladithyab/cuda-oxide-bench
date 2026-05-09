; ModuleID = 'builtin.module'
source_filename = "oxide_3dgs_real"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare float @__nv_expf(float)

define void @rasterize_2dgs(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, ptr %v14, i64 %v15, ptr %v16, i64 %v17, i32 %v18, i32 %v19, i32 %v20, ptr %v21, i64 %v22, ptr %v23, i64 %v24, ptr %v25, i64 %v26) {
entry:
  %v27 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v1, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v3, 1
  %v31 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v32 = insertvalue { ptr, i64 } %v31, i64 %v5, 1
  %v33 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v34 = insertvalue { ptr, i64 } %v33, i64 %v7, 1
  %v35 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v36 = insertvalue { ptr, i64 } %v35, i64 %v9, 1
  %v37 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v38 = insertvalue { ptr, i64 } %v37, i64 %v11, 1
  %v39 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v40 = insertvalue { ptr, i64 } %v39, i64 %v13, 1
  %v41 = insertvalue { ptr, i64 } undef, ptr %v14, 0
  %v42 = insertvalue { ptr, i64 } %v41, i64 %v15, 1
  %v43 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v44 = insertvalue { ptr, i64 } %v43, i64 %v17, 1
  %v45 = insertvalue { ptr, i64 } undef, ptr %v21, 0
  %v46 = insertvalue { ptr, i64 } %v45, i64 %v22, 1
  %v47 = insertvalue { ptr, i64 } undef, ptr %v23, 0
  %v48 = insertvalue { ptr, i64 } %v47, i64 %v24, 1
  %v49 = insertvalue { ptr, i64 } undef, ptr %v25, 0
  %v50 = insertvalue { ptr, i64 } %v49, i64 %v26, 1
  br label %bb0
bb0:
  %v51 = phi { ptr, i64 } [ %v28, %entry ]
  %v52 = phi { ptr, i64 } [ %v30, %entry ]
  %v53 = phi { ptr, i64 } [ %v32, %entry ]
  %v54 = phi { ptr, i64 } [ %v34, %entry ]
  %v55 = phi { ptr, i64 } [ %v36, %entry ]
  %v56 = phi { ptr, i64 } [ %v38, %entry ]
  %v57 = phi { ptr, i64 } [ %v40, %entry ]
  %v58 = phi { ptr, i64 } [ %v42, %entry ]
  %v59 = phi { ptr, i64 } [ %v44, %entry ]
  %v60 = phi i32 [ %v18, %entry ]
  %v61 = phi i32 [ %v19, %entry ]
  %v62 = phi i32 [ %v20, %entry ]
  %v63 = phi { ptr, i64 } [ %v46, %entry ]
  %v64 = phi { ptr, i64 } [ %v48, %entry ]
  %v65 = phi { ptr, i64 } [ %v50, %entry ]
  %v66 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb1
bb1:
  %v67 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v68 = mul i32 %v66, %v67
  %v69 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb3
bb3:
  %v70 = add i32 %v68, %v69
  %v71 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v72 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  br label %bb5
bb5:
  %v73 = mul i32 %v71, %v72
  %v74 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb6
bb6:
  %v75 = add i32 %v73, %v74
  %v76 = icmp uge i32 %v70, %v61
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb7, label %bb8
bb7:
  %v78 = icmp uge i32 %v75, %v62
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb9, label %bb8
bb8:
  br label %bb32
bb9:
  %v80 = uitofp i32 %v70 to float
  %v81 = uitofp i32 %v75 to float
  %v82 = mul i32 %v75, %v61
  %v83 = add i32 %v82, %v70
  %v84 = zext i32 %v83 to i64
  %v85 = zext i32 %v60 to i64
  br label %bb10
bb10:
  %v86 = phi float [ 0.0, %bb9 ], [ %v180, %bb30 ]
  %v87 = phi float [ 0.0, %bb9 ], [ %v181, %bb30 ]
  %v88 = phi float [ 0.0, %bb9 ], [ %v182, %bb30 ]
  %v89 = phi float [ 1.0, %bb9 ], [ %v183, %bb30 ]
  %v90 = phi i64 [ 0, %bb9 ], [ %v184, %bb30 ]
  %v91 = icmp ult i64 %v90, %v85
  %v92 = xor i1 %v91, 1
  br i1 %v92, label %bb31, label %bb11
bb11:
  %v93 = extractvalue { ptr, i64 } %v51, 1
  %v94 = icmp ult i64 %v90, %v93
  br i1 %v94, label %bb12, label %bb33
bb12:
  %v95 = extractvalue { ptr, i64 } %v51, 0
  %v96 = getelementptr inbounds float, ptr %v95, i64 %v90
  %v97 = load float, ptr %v96
  %v98 = fsub float %v80, %v97
  %v99 = extractvalue { ptr, i64 } %v52, 1
  %v100 = icmp ult i64 %v90, %v99
  br i1 %v100, label %bb13, label %bb34
bb13:
  %v101 = extractvalue { ptr, i64 } %v52, 0
  %v102 = getelementptr inbounds float, ptr %v101, i64 %v90
  %v103 = load float, ptr %v102
  %v104 = fsub float %v81, %v103
  %v105 = extractvalue { ptr, i64 } %v53, 1
  %v106 = icmp ult i64 %v90, %v105
  br i1 %v106, label %bb14, label %bb35
bb14:
  %v107 = extractvalue { ptr, i64 } %v53, 0
  %v108 = getelementptr inbounds float, ptr %v107, i64 %v90
  %v109 = load float, ptr %v108
  %v110 = fmul float %v109, %v98
  %v111 = fmul float %v110, %v98
  %v112 = extractvalue { ptr, i64 } %v54, 1
  %v113 = icmp ult i64 %v90, %v112
  br i1 %v113, label %bb15, label %bb36
bb15:
  %v114 = extractvalue { ptr, i64 } %v54, 0
  %v115 = getelementptr inbounds float, ptr %v114, i64 %v90
  %v116 = load float, ptr %v115
  %v117 = fmul float 2.0, %v116
  %v118 = fmul float %v117, %v98
  %v119 = fmul float %v118, %v104
  %v120 = fadd float %v111, %v119
  %v121 = extractvalue { ptr, i64 } %v55, 1
  %v122 = icmp ult i64 %v90, %v121
  br i1 %v122, label %bb16, label %bb37
bb16:
  %v123 = extractvalue { ptr, i64 } %v55, 0
  %v124 = getelementptr inbounds float, ptr %v123, i64 %v90
  %v125 = load float, ptr %v124
  %v126 = fmul float %v125, %v104
  %v127 = fmul float %v126, %v104
  %v128 = fadd float %v120, %v127
  %v129 = fmul float -0.5, %v128
  %v130 = fcmp ole float %v129, 0.0
  %v131 = xor i1 %v130, 1
  br i1 %v131, label %bb30, label %bb17
bb17:
  %v132 = extractvalue { ptr, i64 } %v56, 1
  %v133 = icmp ult i64 %v90, %v132
  br i1 %v133, label %bb18, label %bb38
bb18:
  %v134 = extractvalue { ptr, i64 } %v56, 0
  %v135 = getelementptr inbounds float, ptr %v134, i64 %v90
  %v136 = load float, ptr %v135
  %v137 = call float @__nv_expf(float %v129)
  br label %bb19
bb19:
  %v138 = fmul float %v136, %v137
  %v139 = fcmp oge float %v138, 0.003921568859368563
  %v140 = xor i1 %v139, 1
  br i1 %v140, label %bb29, label %bb20
bb20:
  %v141 = fcmp ogt float %v138, 0.9900000095367432
  %v142 = xor i1 %v141, 1
  br i1 %v142, label %bb22, label %bb21
bb21:
  br label %bb23
bb22:
  br label %bb23
bb23:
  %v143 = phi float [ 0.9900000095367432, %bb21 ], [ %v138, %bb22 ]
  %v144 = fmul float %v143, %v89
  %v145 = extractvalue { ptr, i64 } %v57, 1
  %v146 = icmp ult i64 %v90, %v145
  br i1 %v146, label %bb24, label %bb39
bb24:
  %v147 = extractvalue { ptr, i64 } %v57, 0
  %v148 = getelementptr inbounds float, ptr %v147, i64 %v90
  %v149 = load float, ptr %v148
  %v150 = fmul float %v144, %v149
  %v151 = fadd float %v86, %v150
  %v152 = extractvalue { ptr, i64 } %v58, 1
  %v153 = icmp ult i64 %v90, %v152
  br i1 %v153, label %bb25, label %bb40
bb25:
  %v154 = extractvalue { ptr, i64 } %v58, 0
  %v155 = getelementptr inbounds float, ptr %v154, i64 %v90
  %v156 = load float, ptr %v155
  %v157 = fmul float %v144, %v156
  %v158 = fadd float %v87, %v157
  %v159 = extractvalue { ptr, i64 } %v59, 1
  %v160 = icmp ult i64 %v90, %v159
  br i1 %v160, label %bb26, label %bb41
bb26:
  %v161 = extractvalue { ptr, i64 } %v59, 0
  %v162 = getelementptr inbounds float, ptr %v161, i64 %v90
  %v163 = load float, ptr %v162
  %v164 = fmul float %v144, %v163
  %v165 = fadd float %v88, %v164
  %v166 = fsub float 1.0, %v143
  %v167 = fmul float %v89, %v166
  %v168 = fcmp olt float %v167, 0.00009999999747378752
  %v169 = xor i1 %v168, 1
  br i1 %v169, label %bb28, label %bb27
bb27:
  %v170 = extractvalue { ptr, i64 } %v63, 0
  %v171 = getelementptr inbounds float, ptr %v170, i64 %v84
  store float %v151, ptr %v171
  %v172 = extractvalue { ptr, i64 } %v64, 0
  %v173 = getelementptr inbounds float, ptr %v172, i64 %v84
  store float %v158, ptr %v173
  %v174 = extractvalue { ptr, i64 } %v65, 0
  %v175 = getelementptr inbounds float, ptr %v174, i64 %v84
  store float %v165, ptr %v175
  br label %bb32
bb28:
  br label %bb29
bb29:
  %v176 = phi float [ %v86, %bb19 ], [ %v151, %bb28 ]
  %v177 = phi float [ %v87, %bb19 ], [ %v158, %bb28 ]
  %v178 = phi float [ %v88, %bb19 ], [ %v165, %bb28 ]
  %v179 = phi float [ %v89, %bb19 ], [ %v167, %bb28 ]
  br label %bb30
bb30:
  %v180 = phi float [ %v86, %bb16 ], [ %v176, %bb29 ]
  %v181 = phi float [ %v87, %bb16 ], [ %v177, %bb29 ]
  %v182 = phi float [ %v88, %bb16 ], [ %v178, %bb29 ]
  %v183 = phi float [ %v89, %bb16 ], [ %v179, %bb29 ]
  %v184 = add i64 %v90, 1
  br label %bb10
bb31:
  %v185 = extractvalue { ptr, i64 } %v63, 0
  %v186 = getelementptr inbounds float, ptr %v185, i64 %v84
  store float %v86, ptr %v186
  %v187 = extractvalue { ptr, i64 } %v64, 0
  %v188 = getelementptr inbounds float, ptr %v187, i64 %v84
  store float %v87, ptr %v188
  %v189 = extractvalue { ptr, i64 } %v65, 0
  %v190 = getelementptr inbounds float, ptr %v189, i64 %v84
  store float %v88, ptr %v190
  br label %bb32
bb32:
  ret void
bb33:
  unreachable
bb34:
  unreachable
bb35:
  unreachable
bb36:
  unreachable
bb37:
  unreachable
bb38:
  unreachable
bb39:
  unreachable
bb40:
  unreachable
bb41:
  unreachable
}


@llvm.used = appending global [1 x ptr] [ptr @rasterize_2dgs], section "llvm.metadata"

!0 = !{ptr @rasterize_2dgs, !"kernel", i32 1}
!nvvm.annotations = !{!0}

!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 2}
