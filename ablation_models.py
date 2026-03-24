from aegis import AEGIS

# ==========================================
# 极其纯净的消融模型配置层
# 所有变体统一使用 AEGIS 核心架构和计算逻辑，仅调整开关参数，
# 确保“实验控制变量”的绝对纯粹性和可比性。
# ==========================================

def AEGIS_WoMacro(**kwargs):
    """移除宏观拓扑流，验证 SAGEConv 捕捉高频异常连接的能力"""
    return AEGIS(use_macro=False, **kwargs)

def AEGIS_WoMicro(**kwargs):
    """移除微观边特征流，验证边交互在低频隐蔽渗透中的作用"""
    return AEGIS(use_micro=False, **kwargs)

def AEGIS_WoSpatialGating(**kwargs):
    """移除空间门控，使用 0.5 * (Macro + Micro) 公平直接相加"""
    return AEGIS(use_spatial_gating=False, **kwargs)

def AEGIS_WoEdgeAug(**kwargs):
    """移除边增强机制，退化为普通的 Graph Attention"""
    return AEGIS(use_edge_aug=False, **kwargs)

def AEGIS_FixedTemporal(**kwargs):
    """移除熵驱动的自适应感受野，将所有多尺度卷积核权重强行平均分配"""
    return AEGIS(temporal_mode="fixed", **kwargs)