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
        512,
        requires_grad=True # 记录梯度/表示后续要对 X 求梯度（反向传播）
    )

    y = test_model(x)

    assert y.shape == x.shape # assert:程序自动验证，程序认为这个条件必须为 True；如果不是，就立刻报错
    print(f"shape测试通过")
    assert torch.isfinite(y).all(), f"y_shape = {y.shape}"
    # assert torch.isnan(y).all(), f"Error:{y}" # 检测是否存在 NaN，如果存在报错
    print("数据完整性测试通过")

    loss = y.mean()
    loss.backward() # 模拟反向传播（根据损失去找哪个对损失影响大）
    
    assert x.grad is not None
    print("存在梯度")
    assert x.grad.shape == x.shape
    print("梯度形状测试通过")
    assert torch.isfinite(x.grad).all()
    print("梯度数据测试通过")



if __name__ == "__main__":
    test()
