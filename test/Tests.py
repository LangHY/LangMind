import torch

from model.model import gqa

class TestConfig:
    hidden_size = 512
    num_attention_heads = 8
    num_kv_heads = 2
    dropout = 0.0


def test():
    config = TestConfig()
    test_model = gqa(config)

    x = torch.randn(
        2,
        16,
        512
    )

    y = test_model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
