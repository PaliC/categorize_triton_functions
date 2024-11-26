import triton
import triton.language as tl
import torch

@triton.jit
def test_pid_conds(conds):
    """Test if condition on pids are fulfilled
    E.g.:
        '=0'    checks that pid_0 == 0
        ',>1'   checks that pid_1 > 1
        '>1,=0' checks that pid_0 > 1 and pid_1 == 0
    """
    return _test_pid_conds(conds, tl.program_id(0).handle.data[0], tl.
        program_id(1).handle.data[0], tl.program_id(2).handle.data[0])
