; ModuleID = 'builtin.module'
source_filename = "oxide_matmul_tiled"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_3 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_2 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_1 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_0 = addrspace(3) global [256 x float] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare void @llvm.nvvm.barrier0() #0

define ptx_kernel void @matmul_tiled(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
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
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb2
bb2:
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v21 = mul i32 %v20, 16
  %v22 = add i32 %v21, %v18
  %v23 = mul i32 %v19, 16
  %v24 = add i32 %v23, %v17
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v18 to i64
  %v27 = zext i32 %v16 to i64
  %v28 = zext i32 %v22 to i64
  %v29 = zext i32 %v24 to i64
  %v30 = mul i64 %v26, 16
  %v31 = add i64 %v30, %v25
  %v32 = udiv i64 %v27, 16
  br label %bb5
bb5:
  %v33 = phi float [ 0.0, %bb4 ], [ %v70, %bb25 ]
  %v34 = phi i64 [ 0, %bb4 ], [ %v87, %bb25 ]
  %v35 = icmp ult i64 %v34, %v32
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb26, label %bb6
bb6:
  %v37 = mul i64 %v34, 16
  %v38 = add i64 %v37, %v25
  %v39 = mul i64 %v34, 16
  %v40 = add i64 %v39, %v26
  %v41 = icmp ult i64 %v28, %v27
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb10, label %bb7
bb7:
  %v43 = icmp ult i64 %v38, %v27
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb10, label %bb8
bb8:
  %v45 = mul i64 %v28, %v27
  %v46 = add i64 %v45, %v38
  %v47 = extractvalue { ptr, i64 } %v13, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb9, label %bb30
bb9:
  %v49 = extractvalue { ptr, i64 } %v13, 0
  %v50 = getelementptr inbounds float, ptr %v49, i64 %v46
  %v51 = load float, ptr %v50
  br label %bb11
bb10:
  br label %bb11
bb11:
  %v52 = phi float [ %v51, %bb9 ], [ 0.0, %bb10 ]
  %v53 = icmp ult i64 %v40, %v27
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb15, label %bb12
bb12:
  %v55 = icmp ult i64 %v29, %v27
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb15, label %bb13
bb13:
  %v57 = mul i64 %v40, %v27
  %v58 = add i64 %v57, %v29
  %v59 = extractvalue { ptr, i64 } %v14, 1
  %v60 = icmp ult i64 %v58, %v59
  br i1 %v60, label %bb14, label %bb31
bb14:
  %v61 = extractvalue { ptr, i64 } %v14, 0
  %v62 = getelementptr inbounds float, ptr %v61, i64 %v58
  %v63 = load float, ptr %v62
  br label %bb16
bb15:
  br label %bb16
bb16:
  %v64 = phi float [ %v63, %bb14 ], [ 0.0, %bb15 ]
  %v66 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v31
  br label %bb17
bb17:
  store float %v52, ptr addrspace(3) %v66
  %v68 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v31
  br label %bb18
bb18:
  store float %v64, ptr addrspace(3) %v68
  call void @llvm.nvvm.barrier0() #0
  br label %bb19
bb19:
  br label %bb20
bb20:
  %v70 = phi float [ %v33, %bb19 ], [ %v84, %bb23 ]
  %v71 = phi i64 [ 0, %bb19 ], [ %v85, %bb23 ]
  %v72 = icmp ult i64 %v71, 16
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb24, label %bb21
bb21:
  %v74 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v75 = add i64 %v30, %v71
  %v76 = getelementptr inbounds float, ptr addrspace(3) %v74, i64 %v75
  br label %bb22
bb22:
  %v77 = load float, ptr addrspace(3) %v76
  %v78 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v79 = mul i64 %v71, 16
  %v80 = add i64 %v79, %v25
  %v81 = getelementptr inbounds float, ptr addrspace(3) %v78, i64 %v80
  br label %bb23
bb23:
  %v82 = load float, ptr addrspace(3) %v81
  %v83 = fmul float %v77, %v82
  %v84 = fadd float %v70, %v83
  %v85 = add i64 %v71, 1
  br label %bb20
bb24:
  call void @llvm.nvvm.barrier0() #0
  br label %bb25
bb25:
  %v87 = add i64 %v34, 1
  br label %bb5
bb26:
  %v88 = icmp ult i64 %v28, %v27
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb29, label %bb27
bb27:
  %v90 = icmp ult i64 %v29, %v27
  %v91 = xor i1 %v90, 1
  br i1 %v91, label %bb29, label %bb28
bb28:
  %v92 = extractvalue { ptr, i64 } %v15, 0
  %v93 = mul i64 %v28, %v27
  %v94 = add i64 %v93, %v29
  %v95 = getelementptr inbounds float, ptr %v92, i64 %v94
  store float %v33, ptr %v95
  br label %bb29
bb29:
  ret void
bb30:
  unreachable
bb31:
  unreachable
}

define ptx_kernel void @matmul_tiled_unchecked(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
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
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb2
bb2:
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v21 = mul i32 %v20, 16
  %v22 = add i32 %v21, %v18
  %v23 = mul i32 %v19, 16
  %v24 = add i32 %v23, %v17
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v18 to i64
  %v27 = zext i32 %v16 to i64
  %v28 = zext i32 %v22 to i64
  %v29 = zext i32 %v24 to i64
  %v30 = mul i64 %v26, 16
  %v31 = add i64 %v30, %v25
  %v32 = extractvalue { ptr, i64 } %v13, 0
  %v33 = extractvalue { ptr, i64 } %v14, 0
  %v34 = udiv i64 %v27, 16
  br label %bb5
bb5:
  %v35 = phi float [ 0.0, %bb4 ], [ %v56, %bb15 ]
  %v36 = phi i64 [ 0, %bb4 ], [ %v73, %bb15 ]
  %v37 = icmp ult i64 %v36, %v34
  %v38 = xor i1 %v37, 1
  br i1 %v38, label %bb16, label %bb6
bb6:
  %v39 = mul i64 %v36, 16
  %v40 = add i64 %v39, %v25
  %v41 = mul i64 %v36, 16
  %v42 = add i64 %v41, %v26
  %v43 = mul i64 %v28, %v27
  %v44 = add i64 %v43, %v40
  %v45 = getelementptr inbounds float, ptr %v32, i64 %v44
  %v46 = load float, ptr %v45
  %v47 = mul i64 %v42, %v27
  %v48 = add i64 %v47, %v29
  %v49 = getelementptr inbounds float, ptr %v33, i64 %v48
  %v50 = load float, ptr %v49
  %v52 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_2, i64 %v31
  br label %bb7
bb7:
  store float %v46, ptr addrspace(3) %v52
  %v54 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_3, i64 %v31
  br label %bb8
bb8:
  store float %v50, ptr addrspace(3) %v54
  call void @llvm.nvvm.barrier0() #0
  br label %bb9
bb9:
  br label %bb10
bb10:
  %v56 = phi float [ %v35, %bb9 ], [ %v70, %bb13 ]
  %v57 = phi i64 [ 0, %bb9 ], [ %v71, %bb13 ]
  %v58 = icmp ult i64 %v57, 16
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb14, label %bb11
bb11:
  %v60 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v61 = add i64 %v30, %v57
  %v62 = getelementptr inbounds float, ptr addrspace(3) %v60, i64 %v61
  br label %bb12
bb12:
  %v63 = load float, ptr addrspace(3) %v62
  %v64 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v65 = mul i64 %v57, 16
  %v66 = add i64 %v65, %v25
  %v67 = getelementptr inbounds float, ptr addrspace(3) %v64, i64 %v66
  br label %bb13
bb13:
  %v68 = load float, ptr addrspace(3) %v67
  %v69 = fmul float %v63, %v68
  %v70 = fadd float %v56, %v69
  %v71 = add i64 %v57, 1
  br label %bb10
bb14:
  call void @llvm.nvvm.barrier0() #0
  br label %bb15
bb15:
  %v73 = add i64 %v36, 1
  br label %bb5
bb16:
  %v74 = extractvalue { ptr, i64 } %v15, 0
  %v75 = mul i64 %v28, %v27
  %v76 = add i64 %v75, %v29
  %v77 = getelementptr inbounds float, ptr %v74, i64 %v76
  store float %v35, ptr %v77
  ret void
}


attributes #0 = { convergent }
