; ModuleID = 'builtin.module'
source_filename = "oxide_tma_copy"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_3 = addrspace(3) global [1024 x float] zeroinitializer, align 128
@__shared_mem_2 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_1 = addrspace(3) global [4096 x float] zeroinitializer, align 128
@__shared_mem_0 = addrspace(3) global [1 x i64] zeroinitializer, align 8
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3), i32) #0
declare void @llvm.nvvm.barrier0() #0
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7), ptr addrspace(3), ptr, i32, i32, i16, i64, i1, i1, i32) #0
declare i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3)) #0
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define ptx_kernel void @tma_copy_2d_test(ptr %v0, ptr %v1, i64 %v2, i32 %v3, i32 %v4) {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v1, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v2, 1
  br label %bb0
bb0:
  %v7 = phi ptr [ %v0, %entry ]
  %v8 = phi { ptr, i64 } [ %v6, %entry ]
  %v9 = phi i32 [ %v3, %entry ]
  %v10 = phi i32 [ %v4, %entry ]
  %v11 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb24
bb3:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_0, i32 %v12) #0
  br label %bb4
bb4:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb5
bb5:
  call void @llvm.nvvm.barrier0() #0
  br label %bb6
bb6:
  %v17 = xor i1 %v52, 1
  br i1 %v17, label %bb9, label %bb7
bb7:
  %v19 = addrspacecast ptr addrspace(3) @__shared_mem_1 to ptr
  %v21 = addrspacecast ptr %v19 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v21, ptr addrspace(3) @__shared_mem_0, ptr %v7, i32 %v9, i32 %v10, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb8
bb8:
  br label %bb9
bb9:
  %v23 = xor i1 %v52, 1
  br i1 %v23, label %bb12, label %bb10
bb10:
  %v25 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v26 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v25, i32 16384) #0
  br label %bb11
bb11:
  br label %bb14
bb12:
  %v28 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v29 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v28) #0
  br label %bb13
bb13:
  br label %bb14
bb14:
  %v30 = phi i64 [ %v26, %bb11 ], [ %v29, %bb13 ], [ %v30, %bb17 ]
  %v32 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v33 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.shared.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,l,~{memory}"(ptr addrspace(3) %v32, i64 %v30) #0
  %v34 = trunc i32 %v33 to i1
  br label %bb15
bb15:
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb17, label %bb16
bb16:
  call void @llvm.nvvm.barrier0() #0
  br label %bb18
bb17:
  br label %bb14
bb18:
  %v37 = icmp ult i64 %v51, 4096
  %v38 = xor i1 %v37, 1
  br i1 %v38, label %bb23, label %bb19
bb19:
  %v40 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v41 = getelementptr inbounds float, ptr addrspace(3) %v40, i64 %v51
  br label %bb20
bb20:
  %v42 = load float, ptr addrspace(3) %v41
  %v43 = extractvalue { ptr, i64 } %v8, 1
  %v44 = icmp ult i64 %v51, %v43
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb28, label %bb27
bb21:
  %v46 = extractvalue { i8, ptr } %v59, 1
  store float %v42, ptr %v46
  br label %bb23
bb22:
  br label %bb23
bb23:
  ret void
bb24:
  %v47 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb25
bb25:
  %v48 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb26
bb26:
  %v49 = mul i32 %v47, %v48
  %v50 = add i32 %v49, %v13
  %v51 = zext i32 %v50 to i64
  %v52 = icmp eq i32 %v11, 0
  %v53 = icmp eq i32 %v11, 0
  br i1 %v53, label %bb3, label %bb5
bb27:
  %v54 = extractvalue { ptr, i64 } %v8, 0
  %v55 = getelementptr inbounds float, ptr %v54, i64 %v51
  %v56 = insertvalue { i8, ptr } undef, i8 1, 0
  %v57 = insertvalue { i8, ptr } %v56, ptr %v55, 1
  br label %bb29
bb28:
  %v58 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb29
bb29:
  %v59 = phi { i8, ptr } [ %v57, %bb27 ], [ %v58, %bb28 ]
  %v60 = extractvalue { i8, ptr } %v59, 0
  %v61 = zext i8 %v60 to i64
  %v62 = icmp eq i64 %v61, 1
  br i1 %v62, label %bb21, label %bb30
bb30:
  %v63 = icmp eq i64 %v61, 0
  br i1 %v63, label %bb22, label %bb31
bb31:
  unreachable
}

define ptx_kernel void @tma_pipeline_test(ptr %v0, ptr %v1, i64 %v2) {
entry:
  %v3 = insertvalue { ptr, i64 } undef, ptr %v1, 0
  %v4 = insertvalue { ptr, i64 } %v3, i64 %v2, 1
  br label %bb0
bb0:
  %v5 = phi ptr [ %v0, %entry ]
  %v6 = phi { ptr, i64 } [ %v4, %entry ]
  %v7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v8 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb22
bb3:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_2, i32 %v8) #0
  br label %bb4
bb4:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb5
bb5:
  call void @llvm.nvvm.barrier0() #0
  br label %bb6
bb6:
  %v13 = xor i1 %v42, 1
  br i1 %v13, label %bb9, label %bb7
bb7:
  %v15 = addrspacecast ptr addrspace(3) @__shared_mem_3 to ptr
  %v17 = addrspacecast ptr %v15 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v17, ptr addrspace(3) @__shared_mem_2, ptr %v5, i32 0, i32 0, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb8
bb8:
  br label %bb9
bb9:
  %v19 = xor i1 %v42, 1
  br i1 %v19, label %bb12, label %bb10
bb10:
  %v21 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v22 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v21, i32 4096) #0
  br label %bb11
bb11:
  br label %bb14
bb12:
  %v24 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v25 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v24) #0
  br label %bb13
bb13:
  br label %bb14
bb14:
  %v26 = phi i64 [ %v22, %bb11 ], [ %v25, %bb13 ], [ %v26, %bb17 ]
  %v28 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v29 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.shared.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,l,~{memory}"(ptr addrspace(3) %v28, i64 %v26) #0
  %v30 = trunc i32 %v29 to i1
  br label %bb15
bb15:
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb17, label %bb16
bb16:
  call void @llvm.nvvm.barrier0() #0
  br label %bb18
bb17:
  br label %bb14
bb18:
  %v33 = extractvalue { ptr, i64 } %v6, 1
  %v34 = icmp ult i64 %v41, %v33
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb26, label %bb25
bb19:
  %v36 = extractvalue { i8, ptr } %v49, 1
  store i32 1, ptr %v36
  br label %bb21
bb20:
  br label %bb21
bb21:
  ret void
bb22:
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb23
bb23:
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb24
bb24:
  %v39 = mul i32 %v37, %v38
  %v40 = add i32 %v39, %v9
  %v41 = zext i32 %v40 to i64
  %v42 = icmp eq i32 %v7, 0
  %v43 = icmp eq i32 %v7, 0
  br i1 %v43, label %bb3, label %bb5
bb25:
  %v44 = extractvalue { ptr, i64 } %v6, 0
  %v45 = getelementptr inbounds i32, ptr %v44, i64 %v41
  %v46 = insertvalue { i8, ptr } undef, i8 1, 0
  %v47 = insertvalue { i8, ptr } %v46, ptr %v45, 1
  br label %bb27
bb26:
  %v48 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb27
bb27:
  %v49 = phi { i8, ptr } [ %v47, %bb25 ], [ %v48, %bb26 ]
  %v50 = extractvalue { i8, ptr } %v49, 0
  %v51 = zext i8 %v50 to i64
  %v52 = icmp eq i64 %v51, 1
  br i1 %v52, label %bb19, label %bb28
bb28:
  %v53 = icmp eq i64 %v51, 0
  br i1 %v53, label %bb20, label %bb29
bb29:
  unreachable
}


attributes #0 = { convergent }
