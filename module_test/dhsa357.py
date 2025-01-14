import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
Conv2d = nn.Conv2d
'''
DHSA:Restoring Images in Adverse Weather Conditions via Histogram Transformer (ECCV 2024)

背景：这篇论文提出了一种名为 Histoformer 的新型 Transformer 架构，用于在恶劣天气（如雨雪、雾霾等）条件下恢复
图像的质量。研究人员基于观察发现，恶劣天气引发的降解模式（如亮度变化和遮挡）通常存在相似性，因此提出了直方图
自注意力（Histogram Self-Attention）机制来有效处理这些问题。

DHSA模块原理及作用：
一. 动态范围卷积
在进入自注意力机制之前，DHSA 对特征进行动态范围的卷积处理。这部分操作会先将特征图按像素强度水平和垂直排序，
再使用卷积。
1. 分支分离：输入特征图𝐹被沿通道维度分为两个分支。F1和F2
2. 排序操作：对𝐹1进行水平和垂直排序，重新排列像素，使得高强度和低强度的像素集中在矩阵的角落，从而增强卷积
在降解区域上的作用。
3. 卷积操作：将排序后的特征和原始特征𝐹2连接（concat）起来，通过一个深度可分离卷积，提取动态范围内的空间特征。
原因1：增大卷积感受野，捕捉长距离依赖关系
传统卷积通常具有固定的感受野，主要关注局部邻域的信息，而动态范围卷积通过对像素排序，使得卷积核可以覆盖具有相似强度
但在空间上距离较远的像素。
原因2：强化降解区域与背景区域的区分
在恶劣天气条件下，图像中的降解区域（如雨滴、雾霾等）和背景区域的像素强度分布往往不同。动态范围卷积先对像素进行排序，
使得降解区域和背景区域的像素分布更为集中，便于卷积核分别处理降解信息和保留背景信息。
二. 直方图自注意力
传统的 Transformer 自注意力通常是在固定空间范围或通道维度内计算，限制了对长距离特征的建模能力。而DHSA引入直
方图自注意力机制来解决这一问题。
1. 分组bin操作：DHSA 根据像素的强度对特征进行排序并分组为多个bin，每个bin包含具有相似强度的像素。
2. Bin-wise Histogram Reshaping（BHR）:将图像特征分配到不同 bin 中，每个 bin 覆盖更多像素，用于提取大尺度的全局信息。
3. Frequency-wise Histogram Reshaping（FHR）：将 bin 设为不同频率的像素分布，每个 bin 只包含少量像素，从而提取细粒
度的局部信息。
三. 自注意力计算
1. 查询-键-值排序：使用重组后的 bin 进行查询（Query）和键（Key）的排序，保证相同强度的像素被分配到相同的bin中。
2. 自注意力计算：在 BHR 和 FHR 的特征上分别计算注意力，并将两者的结果通过逐元素乘法融合，生成最终的注意力图。
这种方法确保了每个bin都能在适当的尺度上关注降解信息，使模型在天气降解模式中能够捕捉到具有相似强度的像素间的关联关系
，提升了长距离和局部特征的建模效果。

四、适用场景：图像恢复，图像去噪、雨、雪、雾，目标检测，图像增强等所有CV2二维任务通用。
'''
## Dynamic-range Histogram Self-Attention (DHSA)
# 动态范围直方图自注意力
# 传统的Transformer自注意力通常是在固定空间范围或通道维度内进行计算，限制了对长距离特征的建模能力。
class DHSA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True, scales=[3, 5, 7]):
        super(DHSA, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv357 = nn.ModuleList([
            Conv2d(dim * 5, dim * 5, kernel_size=scale, stride=1, padding=scale // 2, groups=dim * 5, bias=bias)
            for scale in scales
        ])
        self.fuse_dwconv357 = nn.Conv2d(dim * 5 * len(scales), dim * 5, kernel_size=1)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        #
        b, c, h, w = x.shape
        # 以 h 进行排序
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        # 以 w 进行排序
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        # 1×1 --> 3×3
        qkv_dwconv = [qkv_dwconv357(self.qkv(x)) for qkv_dwconv357 in self.qkv_dwconv357]
        qkv = self.fuse_dwconv357(torch.cat(qkv_dwconv, dim=1))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 使用重组后的bin进行查询和键的排序，保证相同强度的像素被分配到相同的bin中。
        # BHR:将图像特征分配到不同的bin中，每个bin覆盖更多的像素，用于提取大尺度的全局信息。
        out1 = self.reshape_attn(q1, k1, v, True)

        # FHR:将bin设为不同频率的像素分布，每个bin中只包含少量像素，从而提取细粒度的局部信息。
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        # 在BHR和FHR的特征上分别计算注意力，并将两者的结果通过逐元素乘法融合，生成最终的注意力图。
        # 这种方法确保了每个bin都能在适当的尺度上关注降解信息，使模型在天气降解模式中能够捕捉到具有相似强度的像素间的关联关系，提升了长距离和局部特征的建模效果。
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out

# 输入 B C H W,  输出B C H W
if __name__ == "__main__":
    # 创建DHSA模块的实例
    model = DHSA(64)
    input = torch.randn(1, 64, 128, 128)
    # 执行前向传播
    output = model(input)
    print('Input size:', input.size())
    print('Output size:', output.size())
