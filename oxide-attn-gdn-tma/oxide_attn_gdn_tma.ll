; ModuleID = 'builtin.module'
source_filename = "oxide_attn_gdn_tma"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_15 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_14 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_13 = addrspace(3) global [64 x float] zeroinitializer, align 4
@__shared_mem_12 = addrspace(3) global [2048 x float] zeroinitializer, align 128
@__shared_mem_11 = addrspace(3) global [64 x float] zeroinitializer, align 4
@__shared_mem_10 = addrspace(3) global [64 x float] zeroinitializer, align 4
@__shared_mem_9 = addrspace(3) global [2 x float] zeroinitializer, align 4
@__shared_mem_8 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_7 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_6 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_5 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_4 = addrspace(3) global [8192 x float] zeroinitializer, align 128
@__shared_mem_3 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_2 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_1 = addrspace(3) global [2 x float] zeroinitializer, align 4
@__shared_mem_0 = addrspace(3) global [1 x i64] zeroinitializer, align 8
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3), i32) #0
declare void @llvm.nvvm.barrier0() #0
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7), ptr addrspace(3), ptr, i32, i32, i16, i64, i1, i1, i32) #0
declare i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3)) #0
declare float @__nv_fmaf(float, float, float)

define void @gdn_decode_dk256_tma(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, ptr %v11, i64 %v12, ptr %v13, i64 %v14, i32 %v15, i32 %v16) {
entry:
  %v17 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v1, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v3, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v5, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v7, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v9, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v11, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v12, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v14, 1
  br label %bb0
bb0:
  %v31 = phi { ptr, i64 } [ %v18, %entry ]
  %v32 = phi { ptr, i64 } [ %v20, %entry ]
  %v33 = phi { ptr, i64 } [ %v22, %entry ]
  %v34 = phi { ptr, i64 } [ %v24, %entry ]
  %v35 = phi { ptr, i64 } [ %v26, %entry ]
  %v36 = phi ptr [ %v10, %entry ]
  %v37 = phi { ptr, i64 } [ %v28, %entry ]
  %v38 = phi { ptr, i64 } [ %v30, %entry ]
  %v39 = phi i32 [ %v15, %entry ]
  %v40 = phi i32 [ %v16, %entry ]
  %v41 = alloca [32 x float]
  %v42 = alloca [32 x float]
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v44 = zext i32 %v43 to i64
  %v45 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v47 = zext i32 %v46 to i64
  %v48 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v49 = zext i32 %v48 to i64
  %v50 = icmp eq i64 %v44, 0
  %v51 = icmp eq i64 %v44, 0
  br i1 %v51, label %bb5, label %bb7
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_0, i32 %v45) #0
  br label %bb6
bb6:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb7
bb7:
  call void @llvm.nvvm.barrier0() #0
  br label %bb8
bb8:
  %v55 = xor i1 %v50, 1
  br i1 %v55, label %bb12, label %bb9
bb9:
  %v56 = extractvalue { ptr, i64 } %v34, 1
  %v57 = icmp ult i64 %v47, %v56
  br i1 %v57, label %bb10, label %bb97
bb10:
  %v58 = extractvalue { ptr, i64 } %v34, 0
  %v59 = getelementptr inbounds float, ptr %v58, i64 %v47
  %v60 = load float, ptr %v59
  %v62 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 0
  br label %bb11
bb11:
  store float %v60, ptr addrspace(3) %v62
  br label %bb12
bb12:
  %v63 = icmp eq i64 %v44, 1
  br i1 %v63, label %bb13, label %bb16
bb13:
  %v64 = extractvalue { ptr, i64 } %v35, 1
  %v65 = icmp ult i64 %v47, %v64
  br i1 %v65, label %bb14, label %bb98
bb14:
  %v66 = extractvalue { ptr, i64 } %v35, 0
  %v67 = getelementptr inbounds float, ptr %v66, i64 %v47
  %v68 = load float, ptr %v67
  %v70 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 1
  br label %bb15
bb15:
  store float %v68, ptr addrspace(3) %v70
  br label %bb16
bb16:
  %v71 = mul i64 %v47, 256
  %v72 = add i64 %v71, %v44
  %v73 = extractvalue { ptr, i64 } %v31, 1
  %v74 = icmp ult i64 %v72, %v73
  br i1 %v74, label %bb17, label %bb99
bb17:
  %v75 = extractvalue { ptr, i64 } %v31, 0
  %v76 = getelementptr inbounds float, ptr %v75, i64 %v72
  %v77 = load float, ptr %v76
  %v78 = extractvalue { ptr, i64 } %v32, 1
  %v79 = icmp ult i64 %v72, %v78
  br i1 %v79, label %bb18, label %bb100
bb18:
  %v80 = extractvalue { ptr, i64 } %v32, 0
  %v81 = getelementptr inbounds float, ptr %v80, i64 %v72
  %v82 = load float, ptr %v81
  %v84 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_2, i64 %v44
  br label %bb19
bb19:
  store float %v82, ptr addrspace(3) %v84
  %v86 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_3, i64 %v44
  br label %bb20
bb20:
  store float %v77, ptr addrspace(3) %v86
  %v87 = xor i1 %v50, 1
  br i1 %v87, label %bb23, label %bb21
bb21:
  %v89 = addrspacecast ptr addrspace(3) @__shared_mem_4 to ptr
  %v90 = mul i64 %v49, 32
  %v91 = trunc i64 %v90 to i32
  %v92 = trunc i64 %v71 to i32
  %v94 = addrspacecast ptr %v89 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v94, ptr addrspace(3) @__shared_mem_0, ptr %v36, i32 %v91, i32 %v92, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb22
bb22:
  br label %bb23
bb23:
  %v96 = xor i1 %v50, 1
  br i1 %v96, label %bb26, label %bb24
bb24:
  %v98 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v99 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v98, i32 32768) #0
  br label %bb25
bb25:
  br label %bb28
bb26:
  %v101 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v102 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v101) #0
  br label %bb27
bb27:
  br label %bb28
bb28:
  %v103 = phi i64 [ %v99, %bb25 ], [ %v102, %bb27 ], [ %v103, %bb31 ]
  %v105 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v106 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.shared.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,l,~{memory}"(ptr addrspace(3) %v105, i64 %v103) #0
  %v107 = trunc i32 %v106 to i1
  br label %bb29
bb29:
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb31, label %bb30
bb30:
  call void @llvm.nvvm.barrier0() #0
  br label %bb32
bb31:
  br label %bb28
bb32:
  %v111 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v112 = getelementptr inbounds float, ptr addrspace(3) %v111, i64 0
  br label %bb33
bb33:
  %v113 = load float, ptr addrspace(3) %v112
  %v114 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v115 = getelementptr inbounds float, ptr addrspace(3) %v114, i64 1
  br label %bb34
bb34:
  %v116 = load float, ptr addrspace(3) %v115
  %v117 = insertvalue [32 x float] undef, float 0.0, 0
  %v118 = insertvalue [32 x float] %v117, float 0.0, 1
  %v119 = insertvalue [32 x float] %v118, float 0.0, 2
  %v120 = insertvalue [32 x float] %v119, float 0.0, 3
  %v121 = insertvalue [32 x float] %v120, float 0.0, 4
  %v122 = insertvalue [32 x float] %v121, float 0.0, 5
  %v123 = insertvalue [32 x float] %v122, float 0.0, 6
  %v124 = insertvalue [32 x float] %v123, float 0.0, 7
  %v125 = insertvalue [32 x float] %v124, float 0.0, 8
  %v126 = insertvalue [32 x float] %v125, float 0.0, 9
  %v127 = insertvalue [32 x float] %v126, float 0.0, 10
  %v128 = insertvalue [32 x float] %v127, float 0.0, 11
  %v129 = insertvalue [32 x float] %v128, float 0.0, 12
  %v130 = insertvalue [32 x float] %v129, float 0.0, 13
  %v131 = insertvalue [32 x float] %v130, float 0.0, 14
  %v132 = insertvalue [32 x float] %v131, float 0.0, 15
  %v133 = insertvalue [32 x float] %v132, float 0.0, 16
  %v134 = insertvalue [32 x float] %v133, float 0.0, 17
  %v135 = insertvalue [32 x float] %v134, float 0.0, 18
  %v136 = insertvalue [32 x float] %v135, float 0.0, 19
  %v137 = insertvalue [32 x float] %v136, float 0.0, 20
  %v138 = insertvalue [32 x float] %v137, float 0.0, 21
  %v139 = insertvalue [32 x float] %v138, float 0.0, 22
  %v140 = insertvalue [32 x float] %v139, float 0.0, 23
  %v141 = insertvalue [32 x float] %v140, float 0.0, 24
  %v142 = insertvalue [32 x float] %v141, float 0.0, 25
  %v143 = insertvalue [32 x float] %v142, float 0.0, 26
  %v144 = insertvalue [32 x float] %v143, float 0.0, 27
  %v145 = insertvalue [32 x float] %v144, float 0.0, 28
  %v146 = insertvalue [32 x float] %v145, float 0.0, 29
  %v147 = insertvalue [32 x float] %v146, float 0.0, 30
  %v148 = insertvalue [32 x float] %v147, float 0.0, 31
  store [32 x float] %v148, ptr %v41
  br label %bb35
bb35:
  %v149 = phi i64 [ 0, %bb34 ], [ %v161, %bb38 ]
  %v150 = icmp ult i64 %v149, 32
  %v151 = xor i1 %v150, 1
  br i1 %v151, label %bb39, label %bb36
bb36:
  %v153 = bitcast ptr addrspace(3) @__shared_mem_4 to ptr addrspace(3)
  %v154 = mul i64 %v44, 32
  %v155 = add i64 %v154, %v149
  %v156 = getelementptr inbounds float, ptr addrspace(3) %v153, i64 %v155
  br label %bb37
bb37:
  %v157 = load float, ptr addrspace(3) %v156
  %v158 = icmp ult i64 %v149, 32
  br i1 %v158, label %bb38, label %bb101
bb38:
  %v159 = fmul float %v157, %v113
  %v160 = getelementptr inbounds [32 x float], ptr %v41, i32 0, i64 %v149
  store float %v159, ptr %v160
  %v161 = add i64 %v149, 1
  br label %bb35
bb39:
  br label %bb40
bb40:
  %v162 = phi i64 [ 0, %bb39 ], [ %v197, %bb59 ]
  %v163 = icmp ult i64 %v162, 32
  %v164 = xor i1 %v163, 1
  br i1 %v164, label %bb60, label %bb41
bb41:
  %v165 = icmp ult i64 %v162, 32
  br i1 %v165, label %bb42, label %bb102
bb42:
  %v166 = getelementptr inbounds [32 x float], ptr %v41, i32 0, i64 %v162
  %v167 = load float, ptr %v166
  %v169 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_5, i64 %v44
  br label %bb43
bb43:
  %v170 = fmul float %v82, %v167
  store float %v170, ptr addrspace(3) %v169
  call void @llvm.nvvm.barrier0() #0
  br label %bb44
bb44:
  br label %bb45
bb45:
  %v172 = phi i64 [ 128, %bb44 ], [ %v189, %bb53 ]
  %v173 = icmp ugt i64 %v172, 0
  %v174 = xor i1 %v173, 1
  br i1 %v174, label %bb54, label %bb46
bb46:
  %v175 = icmp ult i64 %v44, %v172
  %v176 = xor i1 %v175, 1
  br i1 %v176, label %bb51, label %bb47
bb47:
  %v177 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v178 = getelementptr inbounds float, ptr addrspace(3) %v177, i64 %v44
  br label %bb48
bb48:
  %v179 = load float, ptr addrspace(3) %v178
  %v180 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v181 = add i64 %v44, %v172
  %v182 = getelementptr inbounds float, ptr addrspace(3) %v180, i64 %v181
  br label %bb49
bb49:
  %v183 = load float, ptr addrspace(3) %v182
  %v184 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_5, i64 %v44
  br label %bb50
bb50:
  %v185 = fadd float %v179, %v183
  store float %v185, ptr addrspace(3) %v184
  br label %bb52
bb51:
  br label %bb52
bb52:
  call void @llvm.nvvm.barrier0() #0
  br label %bb53
bb53:
  %v187 = zext i32 1 to i64
  %v188 = and i64 %v187, 63
  %v189 = lshr i64 %v172, %v188
  br label %bb45
bb54:
  %v190 = xor i1 %v50, 1
  br i1 %v190, label %bb58, label %bb55
bb55:
  %v191 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v192 = getelementptr inbounds float, ptr addrspace(3) %v191, i64 0
  br label %bb56
bb56:
  %v193 = load float, ptr addrspace(3) %v192
  %v195 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_6, i64 %v162
  br label %bb57
bb57:
  store float %v193, ptr addrspace(3) %v195
  br label %bb58
bb58:
  call void @llvm.nvvm.barrier0() #0
  br label %bb59
bb59:
  %v197 = add i64 %v162, 1
  br label %bb40
bb60:
  %v198 = mul i64 %v49, 32
  %v199 = insertvalue [32 x float] undef, float 0.0, 0
  %v200 = insertvalue [32 x float] %v199, float 0.0, 1
  %v201 = insertvalue [32 x float] %v200, float 0.0, 2
  %v202 = insertvalue [32 x float] %v201, float 0.0, 3
  %v203 = insertvalue [32 x float] %v202, float 0.0, 4
  %v204 = insertvalue [32 x float] %v203, float 0.0, 5
  %v205 = insertvalue [32 x float] %v204, float 0.0, 6
  %v206 = insertvalue [32 x float] %v205, float 0.0, 7
  %v207 = insertvalue [32 x float] %v206, float 0.0, 8
  %v208 = insertvalue [32 x float] %v207, float 0.0, 9
  %v209 = insertvalue [32 x float] %v208, float 0.0, 10
  %v210 = insertvalue [32 x float] %v209, float 0.0, 11
  %v211 = insertvalue [32 x float] %v210, float 0.0, 12
  %v212 = insertvalue [32 x float] %v211, float 0.0, 13
  %v213 = insertvalue [32 x float] %v212, float 0.0, 14
  %v214 = insertvalue [32 x float] %v213, float 0.0, 15
  %v215 = insertvalue [32 x float] %v214, float 0.0, 16
  %v216 = insertvalue [32 x float] %v215, float 0.0, 17
  %v217 = insertvalue [32 x float] %v216, float 0.0, 18
  %v218 = insertvalue [32 x float] %v217, float 0.0, 19
  %v219 = insertvalue [32 x float] %v218, float 0.0, 20
  %v220 = insertvalue [32 x float] %v219, float 0.0, 21
  %v221 = insertvalue [32 x float] %v220, float 0.0, 22
  %v222 = insertvalue [32 x float] %v221, float 0.0, 23
  %v223 = insertvalue [32 x float] %v222, float 0.0, 24
  %v224 = insertvalue [32 x float] %v223, float 0.0, 25
  %v225 = insertvalue [32 x float] %v224, float 0.0, 26
  %v226 = insertvalue [32 x float] %v225, float 0.0, 27
  %v227 = insertvalue [32 x float] %v226, float 0.0, 28
  %v228 = insertvalue [32 x float] %v227, float 0.0, 29
  %v229 = insertvalue [32 x float] %v228, float 0.0, 30
  %v230 = insertvalue [32 x float] %v229, float 0.0, 31
  store [32 x float] %v230, ptr %v42
  %v231 = fmul float %v116, %v82
  br label %bb61
bb61:
  %v232 = phi i64 [ 0, %bb60 ], [ %v253, %bb67 ]
  %v233 = icmp ult i64 %v232, 32
  %v234 = xor i1 %v233, 1
  br i1 %v234, label %bb68, label %bb62
bb62:
  %v235 = add i64 %v71, %v198
  %v236 = add i64 %v235, %v232
  %v237 = extractvalue { ptr, i64 } %v33, 1
  %v238 = icmp ult i64 %v236, %v237
  br i1 %v238, label %bb63, label %bb103
bb63:
  %v239 = extractvalue { ptr, i64 } %v33, 0
  %v240 = getelementptr inbounds float, ptr %v239, i64 %v236
  %v241 = load float, ptr %v240
  %v243 = bitcast ptr addrspace(3) @__shared_mem_6 to ptr addrspace(3)
  %v244 = getelementptr inbounds float, ptr addrspace(3) %v243, i64 %v232
  br label %bb64
bb64:
  %v245 = load float, ptr addrspace(3) %v244
  %v246 = fsub float %v241, %v245
  %v247 = icmp ult i64 %v232, 32
  br i1 %v247, label %bb65, label %bb104
bb65:
  %v248 = getelementptr inbounds [32 x float], ptr %v41, i32 0, i64 %v232
  %v249 = load float, ptr %v248
  %v250 = call float @__nv_fmaf(float %v231, float %v246, float %v249)
  br label %bb66
bb66:
  %v251 = icmp ult i64 %v232, 32
  br i1 %v251, label %bb67, label %bb105
bb67:
  %v252 = getelementptr inbounds [32 x float], ptr %v42, i32 0, i64 %v232
  store float %v250, ptr %v252
  %v253 = add i64 %v232, 1
  br label %bb61
bb68:
  br label %bb69
bb69:
  %v254 = phi i64 [ 0, %bb68 ], [ %v289, %bb88 ]
  %v255 = icmp ult i64 %v254, 32
  %v256 = xor i1 %v255, 1
  br i1 %v256, label %bb89, label %bb70
bb70:
  %v257 = icmp ult i64 %v254, 32
  br i1 %v257, label %bb71, label %bb106
bb71:
  %v258 = getelementptr inbounds [32 x float], ptr %v42, i32 0, i64 %v254
  %v259 = load float, ptr %v258
  %v261 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_5, i64 %v44
  br label %bb72
bb72:
  %v262 = fmul float %v77, %v259
  store float %v262, ptr addrspace(3) %v261
  call void @llvm.nvvm.barrier0() #0
  br label %bb73
bb73:
  br label %bb74
bb74:
  %v264 = phi i64 [ 128, %bb73 ], [ %v281, %bb82 ]
  %v265 = icmp ugt i64 %v264, 0
  %v266 = xor i1 %v265, 1
  br i1 %v266, label %bb83, label %bb75
bb75:
  %v267 = icmp ult i64 %v44, %v264
  %v268 = xor i1 %v267, 1
  br i1 %v268, label %bb80, label %bb76
bb76:
  %v269 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v270 = getelementptr inbounds float, ptr addrspace(3) %v269, i64 %v44
  br label %bb77
bb77:
  %v271 = load float, ptr addrspace(3) %v270
  %v272 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v273 = add i64 %v44, %v264
  %v274 = getelementptr inbounds float, ptr addrspace(3) %v272, i64 %v273
  br label %bb78
bb78:
  %v275 = load float, ptr addrspace(3) %v274
  %v276 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_5, i64 %v44
  br label %bb79
bb79:
  %v277 = fadd float %v271, %v275
  store float %v277, ptr addrspace(3) %v276
  br label %bb81
bb80:
  br label %bb81
bb81:
  call void @llvm.nvvm.barrier0() #0
  br label %bb82
bb82:
  %v279 = zext i32 1 to i64
  %v280 = and i64 %v279, 63
  %v281 = lshr i64 %v264, %v280
  br label %bb74
bb83:
  %v282 = xor i1 %v50, 1
  br i1 %v282, label %bb87, label %bb84
bb84:
  %v283 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v284 = getelementptr inbounds float, ptr addrspace(3) %v283, i64 0
  br label %bb85
bb85:
  %v285 = load float, ptr addrspace(3) %v284
  %v287 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_7, i64 %v254
  br label %bb86
bb86:
  store float %v285, ptr addrspace(3) %v287
  br label %bb87
bb87:
  call void @llvm.nvvm.barrier0() #0
  br label %bb88
bb88:
  %v289 = add i64 %v254, 1
  br label %bb69
bb89:
  %v290 = mul i64 %v72, 256
  %v291 = extractvalue { ptr, i64 } %v37, 0
  br label %bb90
bb90:
  %v292 = phi i64 [ 0, %bb89 ], [ %v301, %bb92 ]
  %v293 = icmp ult i64 %v292, 32
  %v294 = xor i1 %v293, 1
  br i1 %v294, label %bb93, label %bb91
bb91:
  %v295 = icmp ult i64 %v292, 32
  br i1 %v295, label %bb92, label %bb107
bb92:
  %v296 = getelementptr inbounds [32 x float], ptr %v42, i32 0, i64 %v292
  %v297 = load float, ptr %v296
  %v298 = add i64 %v290, %v198
  %v299 = add i64 %v298, %v292
  %v300 = getelementptr inbounds float, ptr %v291, i64 %v299
  store float %v297, ptr %v300
  %v301 = add i64 %v292, 1
  br label %bb90
bb93:
  %v302 = icmp ult i64 %v44, 32
  %v303 = xor i1 %v302, 1
  br i1 %v303, label %bb96, label %bb94
bb94:
  %v304 = extractvalue { ptr, i64 } %v38, 0
  %v306 = bitcast ptr addrspace(3) @__shared_mem_7 to ptr addrspace(3)
  %v307 = getelementptr inbounds float, ptr addrspace(3) %v306, i64 %v44
  br label %bb95
bb95:
  %v308 = load float, ptr addrspace(3) %v307
  %v309 = add i64 %v71, %v198
  %v310 = add i64 %v309, %v44
  %v311 = getelementptr inbounds float, ptr %v304, i64 %v310
  store float %v308, ptr %v311
  br label %bb96
bb96:
  ret void
bb97:
  unreachable
bb98:
  unreachable
bb99:
  unreachable
bb100:
  unreachable
bb101:
  unreachable
bb102:
  unreachable
bb103:
  unreachable
bb104:
  unreachable
bb105:
  unreachable
bb106:
  unreachable
bb107:
  unreachable
}

define void @gdn_decode_dk64_tma(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, ptr %v11, i64 %v12, ptr %v13, i64 %v14, i32 %v15, i32 %v16) {
entry:
  %v17 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v1, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v3, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v5, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v7, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v9, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v11, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v12, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v14, 1
  br label %bb0
bb0:
  %v31 = phi { ptr, i64 } [ %v18, %entry ]
  %v32 = phi { ptr, i64 } [ %v20, %entry ]
  %v33 = phi { ptr, i64 } [ %v22, %entry ]
  %v34 = phi { ptr, i64 } [ %v24, %entry ]
  %v35 = phi { ptr, i64 } [ %v26, %entry ]
  %v36 = phi ptr [ %v10, %entry ]
  %v37 = phi { ptr, i64 } [ %v28, %entry ]
  %v38 = phi { ptr, i64 } [ %v30, %entry ]
  %v39 = phi i32 [ %v15, %entry ]
  %v40 = phi i32 [ %v16, %entry ]
  %v41 = alloca [32 x float]
  %v42 = alloca [32 x float]
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v44 = zext i32 %v43 to i64
  %v45 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v47 = zext i32 %v46 to i64
  %v48 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v49 = zext i32 %v48 to i64
  %v50 = icmp eq i64 %v44, 0
  %v51 = icmp eq i64 %v44, 0
  br i1 %v51, label %bb5, label %bb7
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_8, i32 %v45) #0
  br label %bb6
bb6:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb7
bb7:
  call void @llvm.nvvm.barrier0() #0
  br label %bb8
bb8:
  %v55 = xor i1 %v50, 1
  br i1 %v55, label %bb12, label %bb9
bb9:
  %v56 = extractvalue { ptr, i64 } %v34, 1
  %v57 = icmp ult i64 %v47, %v56
  br i1 %v57, label %bb10, label %bb97
bb10:
  %v58 = extractvalue { ptr, i64 } %v34, 0
  %v59 = getelementptr inbounds float, ptr %v58, i64 %v47
  %v60 = load float, ptr %v59
  %v62 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_9, i64 0
  br label %bb11
bb11:
  store float %v60, ptr addrspace(3) %v62
  br label %bb12
bb12:
  %v63 = icmp eq i64 %v44, 1
  br i1 %v63, label %bb13, label %bb16
bb13:
  %v64 = extractvalue { ptr, i64 } %v35, 1
  %v65 = icmp ult i64 %v47, %v64
  br i1 %v65, label %bb14, label %bb98
bb14:
  %v66 = extractvalue { ptr, i64 } %v35, 0
  %v67 = getelementptr inbounds float, ptr %v66, i64 %v47
  %v68 = load float, ptr %v67
  %v70 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_9, i64 1
  br label %bb15
bb15:
  store float %v68, ptr addrspace(3) %v70
  br label %bb16
bb16:
  %v71 = mul i64 %v47, 64
  %v72 = add i64 %v71, %v44
  %v73 = extractvalue { ptr, i64 } %v31, 1
  %v74 = icmp ult i64 %v72, %v73
  br i1 %v74, label %bb17, label %bb99
bb17:
  %v75 = extractvalue { ptr, i64 } %v31, 0
  %v76 = getelementptr inbounds float, ptr %v75, i64 %v72
  %v77 = load float, ptr %v76
  %v78 = extractvalue { ptr, i64 } %v32, 1
  %v79 = icmp ult i64 %v72, %v78
  br i1 %v79, label %bb18, label %bb100
bb18:
  %v80 = extractvalue { ptr, i64 } %v32, 0
  %v81 = getelementptr inbounds float, ptr %v80, i64 %v72
  %v82 = load float, ptr %v81
  %v84 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_10, i64 %v44
  br label %bb19
bb19:
  store float %v82, ptr addrspace(3) %v84
  %v86 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_11, i64 %v44
  br label %bb20
bb20:
  store float %v77, ptr addrspace(3) %v86
  %v87 = xor i1 %v50, 1
  br i1 %v87, label %bb23, label %bb21
bb21:
  %v89 = addrspacecast ptr addrspace(3) @__shared_mem_12 to ptr
  %v90 = mul i64 %v49, 32
  %v91 = trunc i64 %v90 to i32
  %v92 = trunc i64 %v71 to i32
  %v94 = addrspacecast ptr %v89 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v94, ptr addrspace(3) @__shared_mem_8, ptr %v36, i32 %v91, i32 %v92, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb22
bb22:
  br label %bb23
bb23:
  %v96 = xor i1 %v50, 1
  br i1 %v96, label %bb26, label %bb24
bb24:
  %v98 = bitcast ptr addrspace(3) @__shared_mem_8 to ptr addrspace(3)
  %v99 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v98, i32 8192) #0
  br label %bb25
bb25:
  br label %bb28
bb26:
  %v101 = bitcast ptr addrspace(3) @__shared_mem_8 to ptr addrspace(3)
  %v102 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v101) #0
  br label %bb27
bb27:
  br label %bb28
bb28:
  %v103 = phi i64 [ %v99, %bb25 ], [ %v102, %bb27 ], [ %v103, %bb31 ]
  %v105 = bitcast ptr addrspace(3) @__shared_mem_8 to ptr addrspace(3)
  %v106 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.shared.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,l,~{memory}"(ptr addrspace(3) %v105, i64 %v103) #0
  %v107 = trunc i32 %v106 to i1
  br label %bb29
bb29:
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb31, label %bb30
bb30:
  call void @llvm.nvvm.barrier0() #0
  br label %bb32
bb31:
  br label %bb28
bb32:
  %v111 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v112 = getelementptr inbounds float, ptr addrspace(3) %v111, i64 0
  br label %bb33
bb33:
  %v113 = load float, ptr addrspace(3) %v112
  %v114 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v115 = getelementptr inbounds float, ptr addrspace(3) %v114, i64 1
  br label %bb34
bb34:
  %v116 = load float, ptr addrspace(3) %v115
  %v117 = insertvalue [32 x float] undef, float 0.0, 0
  %v118 = insertvalue [32 x float] %v117, float 0.0, 1
  %v119 = insertvalue [32 x float] %v118, float 0.0, 2
  %v120 = insertvalue [32 x float] %v119, float 0.0, 3
  %v121 = insertvalue [32 x float] %v120, float 0.0, 4
  %v122 = insertvalue [32 x float] %v121, float 0.0, 5
  %v123 = insertvalue [32 x float] %v122, float 0.0, 6
  %v124 = insertvalue [32 x float] %v123, float 0.0, 7
  %v125 = insertvalue [32 x float] %v124, float 0.0, 8
  %v126 = insertvalue [32 x float] %v125, float 0.0, 9
  %v127 = insertvalue [32 x float] %v126, float 0.0, 10
  %v128 = insertvalue [32 x float] %v127, float 0.0, 11
  %v129 = insertvalue [32 x float] %v128, float 0.0, 12
  %v130 = insertvalue [32 x float] %v129, float 0.0, 13
  %v131 = insertvalue [32 x float] %v130, float 0.0, 14
  %v132 = insertvalue [32 x float] %v131, float 0.0, 15
  %v133 = insertvalue [32 x float] %v132, float 0.0, 16
  %v134 = insertvalue [32 x float] %v133, float 0.0, 17
  %v135 = insertvalue [32 x float] %v134, float 0.0, 18
  %v136 = insertvalue [32 x float] %v135, float 0.0, 19
  %v137 = insertvalue [32 x float] %v136, float 0.0, 20
  %v138 = insertvalue [32 x float] %v137, float 0.0, 21
  %v139 = insertvalue [32 x float] %v138, float 0.0, 22
  %v140 = insertvalue [32 x float] %v139, float 0.0, 23
  %v141 = insertvalue [32 x float] %v140, float 0.0, 24
  %v142 = insertvalue [32 x float] %v141, float 0.0, 25
  %v143 = insertvalue [32 x float] %v142, float 0.0, 26
  %v144 = insertvalue [32 x float] %v143, float 0.0, 27
  %v145 = insertvalue [32 x float] %v144, float 0.0, 28
  %v146 = insertvalue [32 x float] %v145, float 0.0, 29
  %v147 = insertvalue [32 x float] %v146, float 0.0, 30
  %v148 = insertvalue [32 x float] %v147, float 0.0, 31
  store [32 x float] %v148, ptr %v41
  br label %bb35
bb35:
  %v149 = phi i64 [ 0, %bb34 ], [ %v161, %bb38 ]
  %v150 = icmp ult i64 %v149, 32
  %v151 = xor i1 %v150, 1
  br i1 %v151, label %bb39, label %bb36
bb36:
  %v153 = bitcast ptr addrspace(3) @__shared_mem_12 to ptr addrspace(3)
  %v154 = mul i64 %v44, 32
  %v155 = add i64 %v154, %v149
  %v156 = getelementptr inbounds float, ptr addrspace(3) %v153, i64 %v155
  br label %bb37
bb37:
  %v157 = load float, ptr addrspace(3) %v156
  %v158 = icmp ult i64 %v149, 32
  br i1 %v158, label %bb38, label %bb101
bb38:
  %v159 = fmul float %v157, %v113
  %v160 = getelementptr inbounds [32 x float], ptr %v41, i32 0, i64 %v149
  store float %v159, ptr %v160
  %v161 = add i64 %v149, 1
  br label %bb35
bb39:
  br label %bb40
bb40:
  %v162 = phi i64 [ 0, %bb39 ], [ %v197, %bb59 ]
  %v163 = icmp ult i64 %v162, 32
  %v164 = xor i1 %v163, 1
  br i1 %v164, label %bb60, label %bb41
bb41:
  %v165 = icmp ult i64 %v162, 32
  br i1 %v165, label %bb42, label %bb102
bb42:
  %v166 = getelementptr inbounds [32 x float], ptr %v41, i32 0, i64 %v162
  %v167 = load float, ptr %v166
  %v169 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_13, i64 %v44
  br label %bb43
bb43:
  %v170 = fmul float %v82, %v167
  store float %v170, ptr addrspace(3) %v169
  call void @llvm.nvvm.barrier0() #0
  br label %bb44
bb44:
  br label %bb45
bb45:
  %v172 = phi i64 [ 32, %bb44 ], [ %v189, %bb53 ]
  %v173 = icmp ugt i64 %v172, 0
  %v174 = xor i1 %v173, 1
  br i1 %v174, label %bb54, label %bb46
bb46:
  %v175 = icmp ult i64 %v44, %v172
  %v176 = xor i1 %v175, 1
  br i1 %v176, label %bb51, label %bb47
bb47:
  %v177 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v178 = getelementptr inbounds float, ptr addrspace(3) %v177, i64 %v44
  br label %bb48
bb48:
  %v179 = load float, ptr addrspace(3) %v178
  %v180 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v181 = add i64 %v44, %v172
  %v182 = getelementptr inbounds float, ptr addrspace(3) %v180, i64 %v181
  br label %bb49
bb49:
  %v183 = load float, ptr addrspace(3) %v182
  %v184 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_13, i64 %v44
  br label %bb50
bb50:
  %v185 = fadd float %v179, %v183
  store float %v185, ptr addrspace(3) %v184
  br label %bb52
bb51:
  br label %bb52
bb52:
  call void @llvm.nvvm.barrier0() #0
  br label %bb53
bb53:
  %v187 = zext i32 1 to i64
  %v188 = and i64 %v187, 63
  %v189 = lshr i64 %v172, %v188
  br label %bb45
bb54:
  %v190 = xor i1 %v50, 1
  br i1 %v190, label %bb58, label %bb55
bb55:
  %v191 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v192 = getelementptr inbounds float, ptr addrspace(3) %v191, i64 0
  br label %bb56
bb56:
  %v193 = load float, ptr addrspace(3) %v192
  %v195 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_14, i64 %v162
  br label %bb57
bb57:
  store float %v193, ptr addrspace(3) %v195
  br label %bb58
bb58:
  call void @llvm.nvvm.barrier0() #0
  br label %bb59
bb59:
  %v197 = add i64 %v162, 1
  br label %bb40
bb60:
  %v198 = mul i64 %v49, 32
  %v199 = insertvalue [32 x float] undef, float 0.0, 0
  %v200 = insertvalue [32 x float] %v199, float 0.0, 1
  %v201 = insertvalue [32 x float] %v200, float 0.0, 2
  %v202 = insertvalue [32 x float] %v201, float 0.0, 3
  %v203 = insertvalue [32 x float] %v202, float 0.0, 4
  %v204 = insertvalue [32 x float] %v203, float 0.0, 5
  %v205 = insertvalue [32 x float] %v204, float 0.0, 6
  %v206 = insertvalue [32 x float] %v205, float 0.0, 7
  %v207 = insertvalue [32 x float] %v206, float 0.0, 8
  %v208 = insertvalue [32 x float] %v207, float 0.0, 9
  %v209 = insertvalue [32 x float] %v208, float 0.0, 10
  %v210 = insertvalue [32 x float] %v209, float 0.0, 11
  %v211 = insertvalue [32 x float] %v210, float 0.0, 12
  %v212 = insertvalue [32 x float] %v211, float 0.0, 13
  %v213 = insertvalue [32 x float] %v212, float 0.0, 14
  %v214 = insertvalue [32 x float] %v213, float 0.0, 15
  %v215 = insertvalue [32 x float] %v214, float 0.0, 16
  %v216 = insertvalue [32 x float] %v215, float 0.0, 17
  %v217 = insertvalue [32 x float] %v216, float 0.0, 18
  %v218 = insertvalue [32 x float] %v217, float 0.0, 19
  %v219 = insertvalue [32 x float] %v218, float 0.0, 20
  %v220 = insertvalue [32 x float] %v219, float 0.0, 21
  %v221 = insertvalue [32 x float] %v220, float 0.0, 22
  %v222 = insertvalue [32 x float] %v221, float 0.0, 23
  %v223 = insertvalue [32 x float] %v222, float 0.0, 24
  %v224 = insertvalue [32 x float] %v223, float 0.0, 25
  %v225 = insertvalue [32 x float] %v224, float 0.0, 26
  %v226 = insertvalue [32 x float] %v225, float 0.0, 27
  %v227 = insertvalue [32 x float] %v226, float 0.0, 28
  %v228 = insertvalue [32 x float] %v227, float 0.0, 29
  %v229 = insertvalue [32 x float] %v228, float 0.0, 30
  %v230 = insertvalue [32 x float] %v229, float 0.0, 31
  store [32 x float] %v230, ptr %v42
  %v231 = fmul float %v116, %v82
  br label %bb61
bb61:
  %v232 = phi i64 [ 0, %bb60 ], [ %v253, %bb67 ]
  %v233 = icmp ult i64 %v232, 32
  %v234 = xor i1 %v233, 1
  br i1 %v234, label %bb68, label %bb62
bb62:
  %v235 = add i64 %v71, %v198
  %v236 = add i64 %v235, %v232
  %v237 = extractvalue { ptr, i64 } %v33, 1
  %v238 = icmp ult i64 %v236, %v237
  br i1 %v238, label %bb63, label %bb103
bb63:
  %v239 = extractvalue { ptr, i64 } %v33, 0
  %v240 = getelementptr inbounds float, ptr %v239, i64 %v236
  %v241 = load float, ptr %v240
  %v243 = bitcast ptr addrspace(3) @__shared_mem_14 to ptr addrspace(3)
  %v244 = getelementptr inbounds float, ptr addrspace(3) %v243, i64 %v232
  br label %bb64
bb64:
  %v245 = load float, ptr addrspace(3) %v244
  %v246 = fsub float %v241, %v245
  %v247 = icmp ult i64 %v232, 32
  br i1 %v247, label %bb65, label %bb104
bb65:
  %v248 = getelementptr inbounds [32 x float], ptr %v41, i32 0, i64 %v232
  %v249 = load float, ptr %v248
  %v250 = call float @__nv_fmaf(float %v231, float %v246, float %v249)
  br label %bb66
bb66:
  %v251 = icmp ult i64 %v232, 32
  br i1 %v251, label %bb67, label %bb105
bb67:
  %v252 = getelementptr inbounds [32 x float], ptr %v42, i32 0, i64 %v232
  store float %v250, ptr %v252
  %v253 = add i64 %v232, 1
  br label %bb61
bb68:
  br label %bb69
bb69:
  %v254 = phi i64 [ 0, %bb68 ], [ %v289, %bb88 ]
  %v255 = icmp ult i64 %v254, 32
  %v256 = xor i1 %v255, 1
  br i1 %v256, label %bb89, label %bb70
bb70:
  %v257 = icmp ult i64 %v254, 32
  br i1 %v257, label %bb71, label %bb106
bb71:
  %v258 = getelementptr inbounds [32 x float], ptr %v42, i32 0, i64 %v254
  %v259 = load float, ptr %v258
  %v261 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_13, i64 %v44
  br label %bb72
bb72:
  %v262 = fmul float %v77, %v259
  store float %v262, ptr addrspace(3) %v261
  call void @llvm.nvvm.barrier0() #0
  br label %bb73
bb73:
  br label %bb74
bb74:
  %v264 = phi i64 [ 32, %bb73 ], [ %v281, %bb82 ]
  %v265 = icmp ugt i64 %v264, 0
  %v266 = xor i1 %v265, 1
  br i1 %v266, label %bb83, label %bb75
bb75:
  %v267 = icmp ult i64 %v44, %v264
  %v268 = xor i1 %v267, 1
  br i1 %v268, label %bb80, label %bb76
bb76:
  %v269 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v270 = getelementptr inbounds float, ptr addrspace(3) %v269, i64 %v44
  br label %bb77
bb77:
  %v271 = load float, ptr addrspace(3) %v270
  %v272 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v273 = add i64 %v44, %v264
  %v274 = getelementptr inbounds float, ptr addrspace(3) %v272, i64 %v273
  br label %bb78
bb78:
  %v275 = load float, ptr addrspace(3) %v274
  %v276 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_13, i64 %v44
  br label %bb79
bb79:
  %v277 = fadd float %v271, %v275
  store float %v277, ptr addrspace(3) %v276
  br label %bb81
bb80:
  br label %bb81
bb81:
  call void @llvm.nvvm.barrier0() #0
  br label %bb82
bb82:
  %v279 = zext i32 1 to i64
  %v280 = and i64 %v279, 63
  %v281 = lshr i64 %v264, %v280
  br label %bb74
bb83:
  %v282 = xor i1 %v50, 1
  br i1 %v282, label %bb87, label %bb84
bb84:
  %v283 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v284 = getelementptr inbounds float, ptr addrspace(3) %v283, i64 0
  br label %bb85
bb85:
  %v285 = load float, ptr addrspace(3) %v284
  %v287 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_15, i64 %v254
  br label %bb86
bb86:
  store float %v285, ptr addrspace(3) %v287
  br label %bb87
bb87:
  call void @llvm.nvvm.barrier0() #0
  br label %bb88
bb88:
  %v289 = add i64 %v254, 1
  br label %bb69
bb89:
  %v290 = mul i64 %v72, 64
  %v291 = extractvalue { ptr, i64 } %v37, 0
  br label %bb90
bb90:
  %v292 = phi i64 [ 0, %bb89 ], [ %v301, %bb92 ]
  %v293 = icmp ult i64 %v292, 32
  %v294 = xor i1 %v293, 1
  br i1 %v294, label %bb93, label %bb91
bb91:
  %v295 = icmp ult i64 %v292, 32
  br i1 %v295, label %bb92, label %bb107
bb92:
  %v296 = getelementptr inbounds [32 x float], ptr %v42, i32 0, i64 %v292
  %v297 = load float, ptr %v296
  %v298 = add i64 %v290, %v198
  %v299 = add i64 %v298, %v292
  %v300 = getelementptr inbounds float, ptr %v291, i64 %v299
  store float %v297, ptr %v300
  %v301 = add i64 %v292, 1
  br label %bb90
bb93:
  %v302 = icmp ult i64 %v44, 32
  %v303 = xor i1 %v302, 1
  br i1 %v303, label %bb96, label %bb94
bb94:
  %v304 = extractvalue { ptr, i64 } %v38, 0
  %v306 = bitcast ptr addrspace(3) @__shared_mem_15 to ptr addrspace(3)
  %v307 = getelementptr inbounds float, ptr addrspace(3) %v306, i64 %v44
  br label %bb95
bb95:
  %v308 = load float, ptr addrspace(3) %v307
  %v309 = add i64 %v71, %v198
  %v310 = add i64 %v309, %v44
  %v311 = getelementptr inbounds float, ptr %v304, i64 %v310
  store float %v308, ptr %v311
  br label %bb96
bb96:
  ret void
bb97:
  unreachable
bb98:
  unreachable
bb99:
  unreachable
bb100:
  unreachable
bb101:
  unreachable
bb102:
  unreachable
bb103:
  unreachable
bb104:
  unreachable
bb105:
  unreachable
bb106:
  unreachable
bb107:
  unreachable
}


@llvm.used = appending global [2 x ptr] [ptr @gdn_decode_dk256_tma, ptr @gdn_decode_dk64_tma], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{ptr @gdn_decode_dk256_tma, !"kernel", i32 1}
!1 = !{ptr @gdn_decode_dk64_tma, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1}

!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 2}
