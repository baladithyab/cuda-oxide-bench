; ModuleID = 'builtin.module'
source_filename = "oxide_matmul"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @matmul_unchecked(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
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
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb1
bb1:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  br label %bb2
bb2:
  %v19 = mul i32 %v17, %v18
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb3
bb3:
  %v21 = add i32 %v19, %v20
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb4
bb4:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb5
bb5:
  %v24 = mul i32 %v22, %v23
  %v25 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb6
bb6:
  %v26 = add i32 %v24, %v25
  %v27 = icmp uge i32 %v21, %v16
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb7, label %bb8
bb7:
  %v29 = icmp uge i32 %v26, %v16
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb9, label %bb8
bb8:
  br label %bb13
bb9:
  %v31 = zext i32 %v16 to i64
  %v32 = zext i32 %v21 to i64
  %v33 = zext i32 %v26 to i64
  %v34 = extractvalue { ptr, i64 } %v13, 0
  %v35 = extractvalue { ptr, i64 } %v14, 0
  br label %bb10
bb10:
  %v36 = phi float [ 0.0, %bb9 ], [ %v49, %bb11 ]
  %v37 = phi i64 [ 0, %bb9 ], [ %v50, %bb11 ]
  %v38 = icmp ult i64 %v37, %v31
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb12, label %bb11
bb11:
  %v40 = mul i64 %v32, %v31
  %v41 = add i64 %v40, %v37
  %v42 = getelementptr inbounds float, ptr %v34, i64 %v41
  %v43 = load float, ptr %v42
  %v44 = mul i64 %v37, %v31
  %v45 = add i64 %v44, %v33
  %v46 = getelementptr inbounds float, ptr %v35, i64 %v45
  %v47 = load float, ptr %v46
  %v48 = fmul float %v43, %v47
  %v49 = fadd float %v36, %v48
  %v50 = add i64 %v37, 1
  br label %bb10
bb12:
  %v51 = extractvalue { ptr, i64 } %v15, 0
  %v52 = mul i64 %v32, %v31
  %v53 = add i64 %v52, %v33
  %v54 = getelementptr inbounds float, ptr %v51, i64 %v53
  store float %v36, ptr %v54
  br label %bb13
bb13:
  ret void
}

define ptx_kernel void @matmul(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
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
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb1
bb1:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  br label %bb2
bb2:
  %v19 = mul i32 %v17, %v18
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb3
bb3:
  %v21 = add i32 %v19, %v20
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb4
bb4:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb5
bb5:
  %v24 = mul i32 %v22, %v23
  %v25 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb6
bb6:
  %v26 = add i32 %v24, %v25
  %v27 = icmp uge i32 %v21, %v16
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb7, label %bb8
bb7:
  %v29 = icmp uge i32 %v26, %v16
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb9, label %bb8
bb8:
  br label %bb15
bb9:
  %v31 = zext i32 %v16 to i64
  %v32 = zext i32 %v21 to i64
  %v33 = zext i32 %v26 to i64
  br label %bb10
bb10:
  %v34 = phi float [ 0.0, %bb9 ], [ %v53, %bb13 ]
  %v35 = phi i64 [ 0, %bb9 ], [ %v54, %bb13 ]
  %v36 = icmp ult i64 %v35, %v31
  %v37 = xor i1 %v36, 1
  br i1 %v37, label %bb14, label %bb11
bb11:
  %v38 = mul i64 %v32, %v31
  %v39 = add i64 %v38, %v35
  %v40 = extractvalue { ptr, i64 } %v13, 1
  %v41 = icmp ult i64 %v39, %v40
  br i1 %v41, label %bb12, label %bb16
bb12:
  %v42 = extractvalue { ptr, i64 } %v13, 0
  %v43 = getelementptr inbounds float, ptr %v42, i64 %v39
  %v44 = load float, ptr %v43
  %v45 = mul i64 %v35, %v31
  %v46 = add i64 %v45, %v33
  %v47 = extractvalue { ptr, i64 } %v14, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb13, label %bb17
bb13:
  %v49 = extractvalue { ptr, i64 } %v14, 0
  %v50 = getelementptr inbounds float, ptr %v49, i64 %v46
  %v51 = load float, ptr %v50
  %v52 = fmul float %v44, %v51
  %v53 = fadd float %v34, %v52
  %v54 = add i64 %v35, 1
  br label %bb10
bb14:
  %v55 = extractvalue { ptr, i64 } %v15, 0
  %v56 = mul i64 %v32, %v31
  %v57 = add i64 %v56, %v33
  %v58 = getelementptr inbounds float, ptr %v55, i64 %v57
  store float %v34, ptr %v58
  br label %bb15
bb15:
  ret void
bb16:
  unreachable
bb17:
  unreachable
}

