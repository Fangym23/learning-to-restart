"""
restart.py — 自定义 autograd 函数，用于对重启事件求导（隐函数定理）。

EventTimeFunction:  计算重启时刻 t* 对 V 的导数: dL/dV = -1/(V_now-V_prev)
EndpointMapFunction: 将时刻梯度传回状态 x: grad_L = sum((x_i-x_prev)*grad_out)

被 alpha_study / derivative_comparison / gradient_restart_comparison 调用。
"""

import torch


class EventTimeFunction(torch.autograd.Function):
    """重启时刻导数。输入: V_now(标量), V_prev(标量,detach), L(迭代号)。输出: L。"""
    @staticmethod
    def forward(ctx, V_now, V_prev, L):
        denom = V_now - V_prev
        ctx.save_for_backward(denom)
        L = L   # 这里只是示意
        return torch.as_tensor(L, dtype=V_now.dtype, device=V_now.device)

 
    @staticmethod
    def backward(ctx, grad_output):
        (denom,) = ctx.saved_tensors
        grad_V_now  = (-1.0 / denom) * grad_output
        return grad_V_now, None,None
    

class EndpointMapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, L_i, x_i,x_prev):
        denom = x_i - x_prev
        ctx.save_for_backward(denom)
        x_next = x_i.clone()
        return x_next

    @staticmethod
    def backward(ctx, grad_output):
        (denom,) = ctx.saved_tensors

        grad_L = (denom * grad_output).sum()

        # 对 x_i 的 Jacobian 是 I
        grad_x = grad_output

        return grad_L, grad_x, None
    
