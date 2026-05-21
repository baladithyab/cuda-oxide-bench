# Wave 21 -- mojo-matmul-bf16: hand-rolled bf16-in/f32-acc tiled matmul on sm_120
#
# Step 1 skeleton: GPU detection only. Real kernel arrives in Tasks 3-7.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return
    with DeviceContext() as ctx:
        print("GPU:", ctx.name())
        print("[mojo-matmul-bf16] Wave 21 Task 1 skeleton OK")
