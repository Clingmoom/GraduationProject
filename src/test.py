from unittest import TestCase


class TestPPO(TestCase):
    def test_make_experience(self):
        import torch

        # 构造两个token
        res = torch.tensor([220, 50256, 100, 220,50256, 200])
        target = [torch.tensor(220), torch.tensor(50256)]

        # zip滑窗组合连续token对：[(220, 50256), (50256, 100), (100, 200)]
        matches = [
                i for i, (a, b) in enumerate(zip(res, res[1:]))
                if a.item() == 220 and b.item() == 50256
            ]

        print("匹配索引:", matches)

    def test_int_and_tensor(self):
        import torch
        # 构造两个token
        int_v = 10
        tensor_v = [10]
        try:
            if int_v == tensor_v:
                print("is ok")
            else:
                print("different")
        except Exception as e:
            print("error")
