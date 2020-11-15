import unittest
import torch
from quickvision.utils import (
    nested_tensor_from_tensor_list,
    NestedTensor,
    seed_everything,
    set_debug_apis,
    AverageMeter,
    ProgressMeter,
    EarlyStopping,
    print_size_of_model,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
class test_torch_utils():
    def test_set_debug_apis(self):
        set_debug_apis(False)
        return True

    def test_seed_everything():
        seed_everything(seed=42)
        return True


if __name__ == '__main__':
    unittest.main()
