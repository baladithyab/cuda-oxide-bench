; ModuleID = 'builtin.module'
source_filename = "oxide_reduction"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_0 = addrspace(3) global [8 x float] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #0
declare void @llvm.nvvm.barrier0() #0

define ptx_kernel void @reduce_sum(ptr %v0, i64 %v1, ptr %v2, i64 %v3) {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v11 = zext i32 %v10 to i64
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
bb2:
  %v13 = zext i32 %v12 to i64
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb3
bb3:
  %v15 = zext i32 %v14 to i64
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  br label %bb4
bb4:
  %v17 = zext i32 %v16 to i64
  %v18 = extractvalue { ptr, i64 } %v8, 1
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  br label %bb5
bb5:
  %v20 = zext i32 %v19 to i64
  %v21 = zext i32 5 to i64
  %v22 = and i64 %v21, 63
  %v23 = lshr i64 %v11, %v22
  %v24 = mul i64 %v15, %v17
  %v25 = mul i64 %v13, %v15
  %v26 = add i64 %v25, %v11
  %v27 = extractvalue { ptr, i64 } %v8, 0
  br label %bb6
bb6:
  %v28 = phi float [ 0.0, %bb5 ], [ %v34, %bb7 ]
  %v29 = phi i64 [ %v26, %bb5 ], [ %v35, %bb7 ]
  %v30 = icmp ult i64 %v29, %v18
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb8, label %bb7
bb7:
  %v32 = getelementptr inbounds float, ptr %v27, i64 %v29
  %v33 = load float, ptr %v32
  %v34 = fadd float %v28, %v33
  %v35 = add i64 %v29, %v24
  br label %bb6
bb8:
  %v36 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v28, i32 16, i32 31) #0
  br label %bb22
bb9:
  %v38 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v23
  br label %bb10
bb10:
  store float %v60, ptr addrspace(3) %v38
  br label %bb11
bb11:
  call void @llvm.nvvm.barrier0() #0
  br label %bb12
bb12:
  %v40 = icmp eq i64 %v23, 0
  br i1 %v40, label %bb13, label %bb21
bb13:
  %v41 = icmp ult i64 %v20, 8
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb16, label %bb14
bb14:
  %v44 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v45 = getelementptr inbounds float, ptr addrspace(3) %v44, i64 %v20
  br label %bb15
bb15:
  %v46 = load float, ptr addrspace(3) %v45
  br label %bb17
bb16:
  br label %bb17
bb17:
  %v47 = phi float [ %v46, %bb15 ], [ 0.0, %bb16 ]
  %v48 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v47, i32 4, i32 31) #0
  br label %bb27
bb18:
  %v49 = extractvalue { ptr, i64 } %v9, 0
  %v50 = bitcast ptr %v49 to ptr
  %v51 = atomicrmw fadd ptr %v50, float %v67 syncscope("device") monotonic
  br label %bb19
bb19:
  br label %bb20
bb20:
  br label %bb21
bb21:
  ret void
bb22:
  %v52 = fadd float %v28, %v36
  %v53 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v52, i32 8, i32 31) #0
  br label %bb23
bb23:
  %v54 = fadd float %v52, %v53
  %v55 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v54, i32 4, i32 31) #0
  br label %bb24
bb24:
  %v56 = fadd float %v54, %v55
  %v57 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v56, i32 2, i32 31) #0
  br label %bb25
bb25:
  %v58 = fadd float %v56, %v57
  %v59 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v58, i32 1, i32 31) #0
  br label %bb26
bb26:
  %v60 = fadd float %v58, %v59
  %v61 = icmp eq i64 %v20, 0
  %v62 = icmp eq i64 %v20, 0
  br i1 %v62, label %bb9, label %bb11
bb27:
  %v63 = fadd float %v47, %v48
  %v64 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v63, i32 2, i32 31) #0
  br label %bb28
bb28:
  %v65 = fadd float %v63, %v64
  %v66 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v65, i32 1, i32 31) #0
  br label %bb29
bb29:
  %v67 = fadd float %v65, %v66
  %v68 = xor i1 %v61, 1
  br i1 %v68, label %bb20, label %bb18
}


attributes #0 = { convergent }
