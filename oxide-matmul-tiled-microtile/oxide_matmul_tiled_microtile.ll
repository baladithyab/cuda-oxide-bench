; ModuleID = 'builtin.module'
source_filename = "oxide_matmul_tiled_microtile"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_3 = addrspace(3) global [1024 x float] zeroinitializer, align 4
@__shared_mem_2 = addrspace(3) global [1024 x float] zeroinitializer, align 4
@__shared_mem_1 = addrspace(3) global [1024 x float] zeroinitializer, align 4
@__shared_mem_0 = addrspace(3) global [1024 x float] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare void @llvm.nvvm.barrier0() #0

define void @matmul_tiled_4x4_safe(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb2
bb2:
  %v20 = zext i32 %v19 to i64
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v22 = zext i32 %v21 to i64
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v16 to i64
  %v26 = mul i64 %v24, 64
  %v27 = mul i64 %v20, 4
  %v28 = add i64 %v26, %v27
  %v29 = mul i64 %v22, 64
  %v30 = mul i64 %v18, 4
  %v31 = add i64 %v29, %v30
  %v32 = mul i64 %v20, 16
  %v33 = add i64 %v32, %v18
  %v34 = udiv i64 %v25, 16
  %v35 = extractvalue { ptr, i64 } %v13, 0
  %v36 = extractvalue { ptr, i64 } %v14, 0
  br label %bb5
bb5:
  %v37 = phi float [ 0.0, %bb4 ], [ %v90, %bb27 ]
  %v38 = phi float [ 0.0, %bb4 ], [ %v91, %bb27 ]
  %v39 = phi float [ 0.0, %bb4 ], [ %v92, %bb27 ]
  %v40 = phi float [ 0.0, %bb4 ], [ %v93, %bb27 ]
  %v41 = phi float [ 0.0, %bb4 ], [ %v94, %bb27 ]
  %v42 = phi float [ 0.0, %bb4 ], [ %v95, %bb27 ]
  %v43 = phi float [ 0.0, %bb4 ], [ %v96, %bb27 ]
  %v44 = phi float [ 0.0, %bb4 ], [ %v97, %bb27 ]
  %v45 = phi float [ 0.0, %bb4 ], [ %v98, %bb27 ]
  %v46 = phi float [ 0.0, %bb4 ], [ %v99, %bb27 ]
  %v47 = phi float [ 0.0, %bb4 ], [ %v100, %bb27 ]
  %v48 = phi float [ 0.0, %bb4 ], [ %v101, %bb27 ]
  %v49 = phi float [ 0.0, %bb4 ], [ %v102, %bb27 ]
  %v50 = phi float [ 0.0, %bb4 ], [ %v103, %bb27 ]
  %v51 = phi float [ 0.0, %bb4 ], [ %v104, %bb27 ]
  %v52 = phi float [ 0.0, %bb4 ], [ %v105, %bb27 ]
  %v53 = phi i64 [ 0, %bb4 ], [ %v191, %bb27 ]
  %v54 = icmp ult i64 %v53, %v34
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb28, label %bb6
bb6:
  %v56 = mul i64 %v53, 16
  br label %bb7
bb7:
  %v57 = phi i64 [ 0, %bb6 ], [ %v72, %bb9 ]
  %v58 = icmp ult i64 %v57, 4
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb10, label %bb8
bb8:
  %v60 = mul i64 %v57, 256
  %v61 = add i64 %v33, %v60
  %v62 = udiv i64 %v61, 16
  %v63 = and i64 %v61, 15
  %v64 = add i64 %v26, %v62
  %v65 = add i64 %v56, %v63
  %v66 = mul i64 %v64, %v25
  %v67 = add i64 %v66, %v65
  %v68 = getelementptr inbounds float, ptr %v35, i64 %v67
  %v69 = load float, ptr %v68
  %v71 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v61
  br label %bb9
bb9:
  store float %v69, ptr addrspace(3) %v71
  %v72 = add i64 %v57, 1
  br label %bb7
bb10:
  br label %bb11
bb11:
  %v73 = phi i64 [ 0, %bb10 ], [ %v88, %bb13 ]
  %v74 = icmp ult i64 %v73, 4
  %v75 = xor i1 %v74, 1
  br i1 %v75, label %bb14, label %bb12
bb12:
  %v76 = mul i64 %v73, 256
  %v77 = add i64 %v33, %v76
  %v78 = udiv i64 %v77, 64
  %v79 = and i64 %v77, 63
  %v80 = add i64 %v56, %v78
  %v81 = add i64 %v29, %v79
  %v82 = mul i64 %v80, %v25
  %v83 = add i64 %v82, %v81
  %v84 = getelementptr inbounds float, ptr %v36, i64 %v83
  %v85 = load float, ptr %v84
  %v87 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v77
  br label %bb13
bb13:
  store float %v85, ptr addrspace(3) %v87
  %v88 = add i64 %v73, 1
  br label %bb11
bb14:
  call void @llvm.nvvm.barrier0() #0
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v90 = phi float [ %v37, %bb15 ], [ %v158, %bb25 ]
  %v91 = phi float [ %v38, %bb15 ], [ %v160, %bb25 ]
  %v92 = phi float [ %v39, %bb15 ], [ %v162, %bb25 ]
  %v93 = phi float [ %v40, %bb15 ], [ %v164, %bb25 ]
  %v94 = phi float [ %v41, %bb15 ], [ %v166, %bb25 ]
  %v95 = phi float [ %v42, %bb15 ], [ %v168, %bb25 ]
  %v96 = phi float [ %v43, %bb15 ], [ %v170, %bb25 ]
  %v97 = phi float [ %v44, %bb15 ], [ %v172, %bb25 ]
  %v98 = phi float [ %v45, %bb15 ], [ %v174, %bb25 ]
  %v99 = phi float [ %v46, %bb15 ], [ %v176, %bb25 ]
  %v100 = phi float [ %v47, %bb15 ], [ %v178, %bb25 ]
  %v101 = phi float [ %v48, %bb15 ], [ %v180, %bb25 ]
  %v102 = phi float [ %v49, %bb15 ], [ %v182, %bb25 ]
  %v103 = phi float [ %v50, %bb15 ], [ %v184, %bb25 ]
  %v104 = phi float [ %v51, %bb15 ], [ %v186, %bb25 ]
  %v105 = phi float [ %v52, %bb15 ], [ %v188, %bb25 ]
  %v106 = phi i64 [ 0, %bb15 ], [ %v189, %bb25 ]
  %v107 = icmp ult i64 %v106, 16
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb26, label %bb17
bb17:
  %v110 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v111 = mul i64 %v27, 16
  %v112 = add i64 %v111, %v106
  %v113 = getelementptr inbounds float, ptr addrspace(3) %v110, i64 %v112
  br label %bb18
bb18:
  %v114 = load float, ptr addrspace(3) %v113
  %v115 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v116 = add i64 %v27, 1
  %v117 = mul i64 %v116, 16
  %v118 = add i64 %v117, %v106
  %v119 = getelementptr inbounds float, ptr addrspace(3) %v115, i64 %v118
  br label %bb19
bb19:
  %v120 = load float, ptr addrspace(3) %v119
  %v121 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v122 = add i64 %v27, 2
  %v123 = mul i64 %v122, 16
  %v124 = add i64 %v123, %v106
  %v125 = getelementptr inbounds float, ptr addrspace(3) %v121, i64 %v124
  br label %bb20
bb20:
  %v126 = load float, ptr addrspace(3) %v125
  %v127 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v128 = add i64 %v27, 3
  %v129 = mul i64 %v128, 16
  %v130 = add i64 %v129, %v106
  %v131 = getelementptr inbounds float, ptr addrspace(3) %v127, i64 %v130
  br label %bb21
bb21:
  %v132 = load float, ptr addrspace(3) %v131
  %v134 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v135 = mul i64 %v106, 64
  %v136 = add i64 %v135, %v30
  %v137 = getelementptr inbounds float, ptr addrspace(3) %v134, i64 %v136
  br label %bb22
bb22:
  %v138 = load float, ptr addrspace(3) %v137
  %v139 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v140 = mul i64 %v106, 64
  %v141 = add i64 %v140, %v30
  %v142 = add i64 %v141, 1
  %v143 = getelementptr inbounds float, ptr addrspace(3) %v139, i64 %v142
  br label %bb23
bb23:
  %v144 = load float, ptr addrspace(3) %v143
  %v145 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v146 = mul i64 %v106, 64
  %v147 = add i64 %v146, %v30
  %v148 = add i64 %v147, 2
  %v149 = getelementptr inbounds float, ptr addrspace(3) %v145, i64 %v148
  br label %bb24
bb24:
  %v150 = load float, ptr addrspace(3) %v149
  %v151 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v152 = mul i64 %v106, 64
  %v153 = add i64 %v152, %v30
  %v154 = add i64 %v153, 3
  %v155 = getelementptr inbounds float, ptr addrspace(3) %v151, i64 %v154
  br label %bb25
bb25:
  %v156 = load float, ptr addrspace(3) %v155
  %v157 = fmul float %v114, %v138
  %v158 = fadd float %v90, %v157
  %v159 = fmul float %v114, %v144
  %v160 = fadd float %v91, %v159
  %v161 = fmul float %v114, %v150
  %v162 = fadd float %v92, %v161
  %v163 = fmul float %v114, %v156
  %v164 = fadd float %v93, %v163
  %v165 = fmul float %v120, %v138
  %v166 = fadd float %v94, %v165
  %v167 = fmul float %v120, %v144
  %v168 = fadd float %v95, %v167
  %v169 = fmul float %v120, %v150
  %v170 = fadd float %v96, %v169
  %v171 = fmul float %v120, %v156
  %v172 = fadd float %v97, %v171
  %v173 = fmul float %v126, %v138
  %v174 = fadd float %v98, %v173
  %v175 = fmul float %v126, %v144
  %v176 = fadd float %v99, %v175
  %v177 = fmul float %v126, %v150
  %v178 = fadd float %v100, %v177
  %v179 = fmul float %v126, %v156
  %v180 = fadd float %v101, %v179
  %v181 = fmul float %v132, %v138
  %v182 = fadd float %v102, %v181
  %v183 = fmul float %v132, %v144
  %v184 = fadd float %v103, %v183
  %v185 = fmul float %v132, %v150
  %v186 = fadd float %v104, %v185
  %v187 = fmul float %v132, %v156
  %v188 = fadd float %v105, %v187
  %v189 = add i64 %v106, 1
  br label %bb16
bb26:
  call void @llvm.nvvm.barrier0() #0
  br label %bb27
bb27:
  %v191 = add i64 %v53, 1
  br label %bb5
bb28:
  %v192 = extractvalue { ptr, i64 } %v15, 0
  %v193 = mul i64 %v28, %v25
  %v194 = add i64 %v193, %v31
  %v195 = getelementptr inbounds float, ptr %v192, i64 %v194
  store float %v37, ptr %v195
  %v196 = add i64 %v194, 1
  %v197 = getelementptr inbounds float, ptr %v192, i64 %v196
  store float %v38, ptr %v197
  %v198 = add i64 %v194, 2
  %v199 = getelementptr inbounds float, ptr %v192, i64 %v198
  store float %v39, ptr %v199
  %v200 = add i64 %v194, 3
  %v201 = getelementptr inbounds float, ptr %v192, i64 %v200
  store float %v40, ptr %v201
  %v202 = add i64 %v28, 1
  %v203 = mul i64 %v202, %v25
  %v204 = add i64 %v203, %v31
  %v205 = getelementptr inbounds float, ptr %v192, i64 %v204
  store float %v41, ptr %v205
  %v206 = add i64 %v204, 1
  %v207 = getelementptr inbounds float, ptr %v192, i64 %v206
  store float %v42, ptr %v207
  %v208 = add i64 %v204, 2
  %v209 = getelementptr inbounds float, ptr %v192, i64 %v208
  store float %v43, ptr %v209
  %v210 = add i64 %v204, 3
  %v211 = getelementptr inbounds float, ptr %v192, i64 %v210
  store float %v44, ptr %v211
  %v212 = add i64 %v28, 2
  %v213 = mul i64 %v212, %v25
  %v214 = add i64 %v213, %v31
  %v215 = getelementptr inbounds float, ptr %v192, i64 %v214
  store float %v45, ptr %v215
  %v216 = add i64 %v214, 1
  %v217 = getelementptr inbounds float, ptr %v192, i64 %v216
  store float %v46, ptr %v217
  %v218 = add i64 %v214, 2
  %v219 = getelementptr inbounds float, ptr %v192, i64 %v218
  store float %v47, ptr %v219
  %v220 = add i64 %v214, 3
  %v221 = getelementptr inbounds float, ptr %v192, i64 %v220
  store float %v48, ptr %v221
  %v222 = add i64 %v28, 3
  %v223 = mul i64 %v222, %v25
  %v224 = add i64 %v223, %v31
  %v225 = getelementptr inbounds float, ptr %v192, i64 %v224
  store float %v49, ptr %v225
  %v226 = add i64 %v224, 1
  %v227 = getelementptr inbounds float, ptr %v192, i64 %v226
  store float %v50, ptr %v227
  %v228 = add i64 %v224, 2
  %v229 = getelementptr inbounds float, ptr %v192, i64 %v228
  store float %v51, ptr %v229
  %v230 = add i64 %v224, 3
  %v231 = getelementptr inbounds float, ptr %v192, i64 %v230
  store float %v52, ptr %v231
  ret void
}

declare float @__nv_fmaf(float, float, float)

define void @matmul_tiled_4x4(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb2
bb2:
  %v20 = zext i32 %v19 to i64
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v22 = zext i32 %v21 to i64
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v16 to i64
  %v26 = mul i64 %v24, 64
  %v27 = mul i64 %v20, 4
  %v28 = add i64 %v26, %v27
  %v29 = mul i64 %v22, 64
  %v30 = mul i64 %v18, 4
  %v31 = add i64 %v29, %v30
  %v32 = mul i64 %v20, 16
  %v33 = add i64 %v32, %v18
  %v34 = udiv i64 %v25, 16
  %v35 = extractvalue { ptr, i64 } %v13, 0
  %v36 = extractvalue { ptr, i64 } %v14, 0
  br label %bb5
bb5:
  %v37 = phi float [ 0.0, %bb4 ], [ %v90, %bb43 ]
  %v38 = phi float [ 0.0, %bb4 ], [ %v91, %bb43 ]
  %v39 = phi float [ 0.0, %bb4 ], [ %v92, %bb43 ]
  %v40 = phi float [ 0.0, %bb4 ], [ %v93, %bb43 ]
  %v41 = phi float [ 0.0, %bb4 ], [ %v94, %bb43 ]
  %v42 = phi float [ 0.0, %bb4 ], [ %v95, %bb43 ]
  %v43 = phi float [ 0.0, %bb4 ], [ %v96, %bb43 ]
  %v44 = phi float [ 0.0, %bb4 ], [ %v97, %bb43 ]
  %v45 = phi float [ 0.0, %bb4 ], [ %v98, %bb43 ]
  %v46 = phi float [ 0.0, %bb4 ], [ %v99, %bb43 ]
  %v47 = phi float [ 0.0, %bb4 ], [ %v100, %bb43 ]
  %v48 = phi float [ 0.0, %bb4 ], [ %v101, %bb43 ]
  %v49 = phi float [ 0.0, %bb4 ], [ %v102, %bb43 ]
  %v50 = phi float [ 0.0, %bb4 ], [ %v103, %bb43 ]
  %v51 = phi float [ 0.0, %bb4 ], [ %v104, %bb43 ]
  %v52 = phi float [ 0.0, %bb4 ], [ %v105, %bb43 ]
  %v53 = phi i64 [ 0, %bb4 ], [ %v175, %bb43 ]
  %v54 = icmp ult i64 %v53, %v34
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb44, label %bb6
bb6:
  %v56 = mul i64 %v53, 16
  br label %bb7
bb7:
  %v57 = phi i64 [ 0, %bb6 ], [ %v72, %bb9 ]
  %v58 = icmp ult i64 %v57, 4
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb10, label %bb8
bb8:
  %v60 = mul i64 %v57, 256
  %v61 = add i64 %v33, %v60
  %v62 = udiv i64 %v61, 16
  %v63 = and i64 %v61, 15
  %v64 = add i64 %v26, %v62
  %v65 = add i64 %v56, %v63
  %v66 = mul i64 %v64, %v25
  %v67 = add i64 %v66, %v65
  %v68 = getelementptr inbounds float, ptr %v35, i64 %v67
  %v69 = load float, ptr %v68
  %v71 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_2, i64 %v61
  br label %bb9
bb9:
  store float %v69, ptr addrspace(3) %v71
  %v72 = add i64 %v57, 1
  br label %bb7
bb10:
  br label %bb11
bb11:
  %v73 = phi i64 [ 0, %bb10 ], [ %v88, %bb13 ]
  %v74 = icmp ult i64 %v73, 4
  %v75 = xor i1 %v74, 1
  br i1 %v75, label %bb14, label %bb12
bb12:
  %v76 = mul i64 %v73, 256
  %v77 = add i64 %v33, %v76
  %v78 = udiv i64 %v77, 64
  %v79 = and i64 %v77, 63
  %v80 = add i64 %v56, %v78
  %v81 = add i64 %v29, %v79
  %v82 = mul i64 %v80, %v25
  %v83 = add i64 %v82, %v81
  %v84 = getelementptr inbounds float, ptr %v36, i64 %v83
  %v85 = load float, ptr %v84
  %v87 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_3, i64 %v77
  br label %bb13
bb13:
  store float %v85, ptr addrspace(3) %v87
  %v88 = add i64 %v73, 1
  br label %bb11
bb14:
  call void @llvm.nvvm.barrier0() #0
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v90 = phi float [ %v37, %bb15 ], [ %v157, %bb41 ]
  %v91 = phi float [ %v38, %bb15 ], [ %v158, %bb41 ]
  %v92 = phi float [ %v39, %bb15 ], [ %v159, %bb41 ]
  %v93 = phi float [ %v40, %bb15 ], [ %v160, %bb41 ]
  %v94 = phi float [ %v41, %bb15 ], [ %v161, %bb41 ]
  %v95 = phi float [ %v42, %bb15 ], [ %v162, %bb41 ]
  %v96 = phi float [ %v43, %bb15 ], [ %v163, %bb41 ]
  %v97 = phi float [ %v44, %bb15 ], [ %v164, %bb41 ]
  %v98 = phi float [ %v45, %bb15 ], [ %v165, %bb41 ]
  %v99 = phi float [ %v46, %bb15 ], [ %v166, %bb41 ]
  %v100 = phi float [ %v47, %bb15 ], [ %v167, %bb41 ]
  %v101 = phi float [ %v48, %bb15 ], [ %v168, %bb41 ]
  %v102 = phi float [ %v49, %bb15 ], [ %v169, %bb41 ]
  %v103 = phi float [ %v50, %bb15 ], [ %v170, %bb41 ]
  %v104 = phi float [ %v51, %bb15 ], [ %v171, %bb41 ]
  %v105 = phi float [ %v52, %bb15 ], [ %v172, %bb41 ]
  %v106 = phi i64 [ 0, %bb15 ], [ %v173, %bb41 ]
  %v107 = icmp ult i64 %v106, 16
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb42, label %bb17
bb17:
  %v110 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v111 = mul i64 %v27, 16
  %v112 = add i64 %v111, %v106
  %v113 = getelementptr inbounds float, ptr addrspace(3) %v110, i64 %v112
  br label %bb18
bb18:
  %v114 = load float, ptr addrspace(3) %v113
  %v115 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v116 = add i64 %v27, 1
  %v117 = mul i64 %v116, 16
  %v118 = add i64 %v117, %v106
  %v119 = getelementptr inbounds float, ptr addrspace(3) %v115, i64 %v118
  br label %bb19
bb19:
  %v120 = load float, ptr addrspace(3) %v119
  %v121 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v122 = add i64 %v27, 2
  %v123 = mul i64 %v122, 16
  %v124 = add i64 %v123, %v106
  %v125 = getelementptr inbounds float, ptr addrspace(3) %v121, i64 %v124
  br label %bb20
bb20:
  %v126 = load float, ptr addrspace(3) %v125
  %v127 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v128 = add i64 %v27, 3
  %v129 = mul i64 %v128, 16
  %v130 = add i64 %v129, %v106
  %v131 = getelementptr inbounds float, ptr addrspace(3) %v127, i64 %v130
  br label %bb21
bb21:
  %v132 = load float, ptr addrspace(3) %v131
  %v134 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v135 = mul i64 %v106, 64
  %v136 = add i64 %v135, %v30
  %v137 = getelementptr inbounds float, ptr addrspace(3) %v134, i64 %v136
  br label %bb22
bb22:
  %v138 = load float, ptr addrspace(3) %v137
  %v139 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v140 = mul i64 %v106, 64
  %v141 = add i64 %v140, %v30
  %v142 = add i64 %v141, 1
  %v143 = getelementptr inbounds float, ptr addrspace(3) %v139, i64 %v142
  br label %bb23
bb23:
  %v144 = load float, ptr addrspace(3) %v143
  %v145 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v146 = mul i64 %v106, 64
  %v147 = add i64 %v146, %v30
  %v148 = add i64 %v147, 2
  %v149 = getelementptr inbounds float, ptr addrspace(3) %v145, i64 %v148
  br label %bb24
bb24:
  %v150 = load float, ptr addrspace(3) %v149
  %v151 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v152 = mul i64 %v106, 64
  %v153 = add i64 %v152, %v30
  %v154 = add i64 %v153, 3
  %v155 = getelementptr inbounds float, ptr addrspace(3) %v151, i64 %v154
  br label %bb25
bb25:
  %v156 = load float, ptr addrspace(3) %v155
  %v157 = call float @__nv_fmaf(float %v114, float %v138, float %v90)
  br label %bb26
bb26:
  %v158 = call float @__nv_fmaf(float %v114, float %v144, float %v91)
  br label %bb27
bb27:
  %v159 = call float @__nv_fmaf(float %v114, float %v150, float %v92)
  br label %bb28
bb28:
  %v160 = call float @__nv_fmaf(float %v114, float %v156, float %v93)
  br label %bb29
bb29:
  %v161 = call float @__nv_fmaf(float %v120, float %v138, float %v94)
  br label %bb30
bb30:
  %v162 = call float @__nv_fmaf(float %v120, float %v144, float %v95)
  br label %bb31
bb31:
  %v163 = call float @__nv_fmaf(float %v120, float %v150, float %v96)
  br label %bb32
bb32:
  %v164 = call float @__nv_fmaf(float %v120, float %v156, float %v97)
  br label %bb33
bb33:
  %v165 = call float @__nv_fmaf(float %v126, float %v138, float %v98)
  br label %bb34
bb34:
  %v166 = call float @__nv_fmaf(float %v126, float %v144, float %v99)
  br label %bb35
bb35:
  %v167 = call float @__nv_fmaf(float %v126, float %v150, float %v100)
  br label %bb36
bb36:
  %v168 = call float @__nv_fmaf(float %v126, float %v156, float %v101)
  br label %bb37
bb37:
  %v169 = call float @__nv_fmaf(float %v132, float %v138, float %v102)
  br label %bb38
bb38:
  %v170 = call float @__nv_fmaf(float %v132, float %v144, float %v103)
  br label %bb39
bb39:
  %v171 = call float @__nv_fmaf(float %v132, float %v150, float %v104)
  br label %bb40
bb40:
  %v172 = call float @__nv_fmaf(float %v132, float %v156, float %v105)
  br label %bb41
bb41:
  %v173 = add i64 %v106, 1
  br label %bb16
bb42:
  call void @llvm.nvvm.barrier0() #0
  br label %bb43
bb43:
  %v175 = add i64 %v53, 1
  br label %bb5
bb44:
  %v176 = extractvalue { ptr, i64 } %v15, 0
  %v177 = mul i64 %v28, %v25
  %v178 = add i64 %v177, %v31
  %v179 = getelementptr inbounds float, ptr %v176, i64 %v178
  store float %v37, ptr %v179
  %v180 = add i64 %v178, 1
  %v181 = getelementptr inbounds float, ptr %v176, i64 %v180
  store float %v38, ptr %v181
  %v182 = add i64 %v178, 2
  %v183 = getelementptr inbounds float, ptr %v176, i64 %v182
  store float %v39, ptr %v183
  %v184 = add i64 %v178, 3
  %v185 = getelementptr inbounds float, ptr %v176, i64 %v184
  store float %v40, ptr %v185
  %v186 = add i64 %v28, 1
  %v187 = mul i64 %v186, %v25
  %v188 = add i64 %v187, %v31
  %v189 = getelementptr inbounds float, ptr %v176, i64 %v188
  store float %v41, ptr %v189
  %v190 = add i64 %v188, 1
  %v191 = getelementptr inbounds float, ptr %v176, i64 %v190
  store float %v42, ptr %v191
  %v192 = add i64 %v188, 2
  %v193 = getelementptr inbounds float, ptr %v176, i64 %v192
  store float %v43, ptr %v193
  %v194 = add i64 %v188, 3
  %v195 = getelementptr inbounds float, ptr %v176, i64 %v194
  store float %v44, ptr %v195
  %v196 = add i64 %v28, 2
  %v197 = mul i64 %v196, %v25
  %v198 = add i64 %v197, %v31
  %v199 = getelementptr inbounds float, ptr %v176, i64 %v198
  store float %v45, ptr %v199
  %v200 = add i64 %v198, 1
  %v201 = getelementptr inbounds float, ptr %v176, i64 %v200
  store float %v46, ptr %v201
  %v202 = add i64 %v198, 2
  %v203 = getelementptr inbounds float, ptr %v176, i64 %v202
  store float %v47, ptr %v203
  %v204 = add i64 %v198, 3
  %v205 = getelementptr inbounds float, ptr %v176, i64 %v204
  store float %v48, ptr %v205
  %v206 = add i64 %v28, 3
  %v207 = mul i64 %v206, %v25
  %v208 = add i64 %v207, %v31
  %v209 = getelementptr inbounds float, ptr %v176, i64 %v208
  store float %v49, ptr %v209
  %v210 = add i64 %v208, 1
  %v211 = getelementptr inbounds float, ptr %v176, i64 %v210
  store float %v50, ptr %v211
  %v212 = add i64 %v208, 2
  %v213 = getelementptr inbounds float, ptr %v176, i64 %v212
  store float %v51, ptr %v213
  %v214 = add i64 %v208, 3
  %v215 = getelementptr inbounds float, ptr %v176, i64 %v214
  store float %v52, ptr %v215
  ret void
}


@llvm.used = appending global [2 x ptr] [ptr @matmul_tiled_4x4_safe, ptr @matmul_tiled_4x4], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{ptr @matmul_tiled_4x4_safe, !"kernel", i32 1}
!1 = !{ptr @matmul_tiled_4x4, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1}

!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 2}
