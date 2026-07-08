# BioDynamics Agent v4 - 通路注册表（Pathway Registry）
# 定义 10 条核心信号通路的 KEGG / Reactome ID 映射与关键词集合。
# 关键词列表用于 P4 Pathway Planner 的通路识别（规则优先匹配），
# P1 阶段仅作为常量定义，不接入主流程。

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathwayEntry:
    """单条通路的标准注册信息。

    Attributes:
        pathway_class: 通路类别键（与 v4 架构一致，如 "EGFR_RTK"）
        kegg_id: KEGG pathway ID（如 "hsa01521"）
        reactome_id: Reactome pathway ID（如 "R-HSA-177929"）
        keywords: 通路识别关键词列表（中英文，≥8 个），供 Pathway Planner 规则匹配
        description: 通路简述（中文）
        biomodels_id: BioModels 模型 ID（如 "BIOMD0000000017"），用于 SBML grounding
            与参数检索；TD-043 (IB-068) 新增字段，默认空串以保持向后兼容
    """
    pathway_class: str
    kegg_id: str
    reactome_id: str
    keywords: tuple[str, ...]
    description: str
    # TD-043 (IB-068) 修复：新增 biomodels_id 字段，默认空串保持向后兼容
    biomodels_id: str = ""

    def matches(self, text: str) -> bool:
        """判断文本是否包含本通路的关键词（大小写不敏感，词边界匹配）。

        使用词边界（\\b）匹配，避免子字符串误命中（如 "unrelated" 命中 "Rel"）。
        对于含中文或特殊字符的关键词，回退到子串匹配。

        Args:
            text: 用户输入或机制描述文本

        Returns:
            True 若文本命中任一关键词
        """
        if not text:
            return False
        text_lower = text.lower()
        for kw in self.keywords:
            kw_lower = kw.lower()
            # 含非 ASCII（中文/希腊字母等）的关键词用子串匹配
            if not kw_lower.isascii():
                if kw_lower in text_lower:
                    return True
                continue
            # 纯 ASCII 关键词用词边界匹配
            if re.search(rf"\b{re.escape(kw_lower)}\b", text_lower):
                return True
        return False


# =============================================================================
# 10 条核心通路注册表
# 关键词每条通路至少 8 个，覆盖中英文别名、缩写、关键蛋白
# =============================================================================
PATHWAY_REGISTRY: dict[str, PathwayEntry] = {
    "EGFR_RTK": PathwayEntry(
        pathway_class="EGFR_RTK",
        kegg_id="hsa01521",
        reactome_id="R-HSA-177929",
        keywords=(
            "EGFR", "ERBB1", "HER1", "ErbB", "epidermal growth factor receptor",
            "表皮生长因子受体", "RTK", "receptor tyrosine kinase",
            "EGF", "TGF-alpha", "amphiregulin", "HB-EGF",
        ),
        description="EGF 受体酪氨酸激酶通路：配体结合→受体二聚化→自磷酸化→下游信号",
        biomodels_id="BIOMD0000000017",  # TD-043 (IB-068)
    ),
    "MAPK_ERK": PathwayEntry(
        pathway_class="MAPK_ERK",
        kegg_id="hsa04010",
        reactome_id="R-HSA-5684996",
        keywords=(
            "MAPK", "ERK", "MEK", "RAF", "RAS", "BRAF", "KRAS", "NRAS",
            "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "丝裂原活化蛋白激酶",
            "Ras-Raf-MEK-ERK", "级联放大", "cascade",
        ),
        description="MAPK/ERK 级联通路：Ras→Raf→MEK→ERK 三级激酶级联放大",
        biomodels_id="BIOMD0000000010",  # TD-043 (IB-068) Schoeberl MAPK model
    ),
    "PI3K_AKT_mTOR": PathwayEntry(
        pathway_class="PI3K_AKT_mTOR",
        kegg_id="hsa04151",
        reactome_id="R-HSA-199418",
        keywords=(
            "PI3K", "AKT", "mTOR", "PTEN", "PIP3", "PIP2", "PDK1",
            "phosphoinositide 3-kinase", "protein kinase B", "PKB",
            "磷脂酰肌醇3激酶", "TSC", "Rheb", "S6K", "4E-BP1",
        ),
        description="PI3K-AKT-mTOR 通路：生长因子→PI3K→PIP3→AKT→mTOR，调控代谢与存活",
        biomodels_id="BIOMD0000000054",  # TD-043 (IB-068)
    ),
    "p53": PathwayEntry(
        pathway_class="p53",
        kegg_id="hsa04115",
        reactome_id="R-HSA-5633007",
        keywords=(
            "p53", "TP53", "MDM2", "p21", "CDKN1A", "BAX", "PUMA", "NOXA",
            "肿瘤抑制蛋白p53", "tumor suppressor p53",
            "DNA damage", "DNA损伤", "细胞周期阻滞", "apoptosis",
        ),
        description="p53 通路：DNA 损伤→p53 磷酸化→MDM2 反馈环路→周期阻滞/凋亡",
        biomodels_id="BIOMD0000000012",  # TD-043 (IB-068)
    ),
    "APOPTOSIS": PathwayEntry(
        pathway_class="APOPTOSIS",
        kegg_id="hsa04210",
        reactome_id="R-HSA-109606",
        keywords=(
            "apoptosis", "细胞凋亡", "caspase", "Caspase-3", "Caspase-8", "Caspase-9",
            "Bcl-2", "Bax", "cytochrome c", "细胞色素c",
            "FasL", "Fas", "TNF", "TRAIL", "死亡受体", "线粒体途径",
        ),
        description="凋亡通路：外源性（死亡受体）+ 内源性（线粒体）+ Caspase 级联",
        biomodels_id="BIOMD0000000200",  # TD-043 (IB-068)
    ),
    "CELL_CYCLE": PathwayEntry(
        pathway_class="CELL_CYCLE",
        kegg_id="hsa04110",
        reactome_id="R-HSA-1640170",
        keywords=(
            "cell cycle", "细胞周期", "Cyclin", "CDK", "CDKN2A", "p16", "p21",
            "Cyclin D", "Cyclin E", "Cyclin B", "CDC", "Rb", "E2F",
            "G1/S", "G2/M", "checkpoint", "检查点",
        ),
        description="细胞周期通路：Cyclin-CDK 驱动 G1/S/G2/M 转换，Rb/E2F 调控",
        biomodels_id="BIOMD0000000005",  # TD-043 (IB-068)
    ),
    "JAK_STAT": PathwayEntry(
        pathway_class="JAK_STAT",
        kegg_id="hsa04630",
        reactome_id="R-HSA-1059684",
        keywords=(
            "JAK", "STAT", "JAK1", "JAK2", "JAK3", "TYK2",
            "STAT1", "STAT3", "STAT5", "STAT6",
            "cytokine", "干扰素", "interferon", "IL-2", "IL-6",
            "growth hormone", "生长激素", "SH2", "磷酸化",
        ),
        description="JAK-STAT 通路：细胞因子受体→JK 磷酸化→STAT 二聚体入核→转录",
        biomodels_id="BIOMD0000000214",  # TD-043 (IB-068)
    ),
    "NF_KB": PathwayEntry(
        pathway_class="NF_KB",
        kegg_id="hsa04064",
        reactome_id="R-HSA-5602497",
        keywords=(
            "NF-kB", "NF-κB", "NFKB", "IkB", "IκB", "IKK", "RelA", "p65", "p50",
            "TNF", "LPS", "TLR", "炎症", "inflammation",
            "免疫", "immune", "Rel", "IkBa", "IκBα", "proteasome",
        ),
        description="NF-κB 通路：TNF/TLR→IKK→IκBα 降解→NF-κB 入核，炎症响应核心",
        biomodels_id="BIOMD0000000208",  # TD-043 (IB-068)
    ),
    "WNT": PathwayEntry(
        pathway_class="WNT",
        kegg_id="hsa04310",
        reactome_id="R-HSA-195721",
        keywords=(
            "Wnt", "WNT", "β-catenin", "beta-catenin", "APC", "Axin", "GSK3", "GSK-3β",
            "destruction complex", "降解复合物", "LRP5", "LRP6", "Frizzled", "FZD",
            "TCF", "LEF", "canonical Wnt", "经典Wnt通路", "胚胎发育",
        ),
        description="Wnt 通路：Wnt→Frizzled/LRP→破坏复合体解离→β-catenin 累积入核",
        biomodels_id="BIOMD0000000026",  # TD-043 (IB-068) Lee Wnt model
    ),
    "TGF_BETA": PathwayEntry(
        pathway_class="TGF_BETA",
        kegg_id="hsa04350",
        reactome_id="R-HSA-170834",
        keywords=(
            "TGF-beta", "TGF-β", "TGFB", "TGFBR", "SMAD", "Smad2", "Smad3", "Smad4",
            "transforming growth factor beta", "转化生长因子β",
            "receptor serine kinase", "受体丝氨酸激酶",
            "EMT", "上皮间质转化", "R-SMAD", "Co-SMAD",
        ),
        description="TGF-β 通路：TGF-β→TβR→SMAD2/3 磷酸化→与 SMAD4 结并入核",
        biomodels_id="BIOMD0000000053",  # TD-043 (IB-068)
    ),
}


def lookup_pathway(text: str) -> str | None:
    """根据文本匹配返回通路类别键（rule-based，供 P4 Pathway Planner 复用）。

    匹配规则：遍历所有通路，返回第一个关键词命中的通路类别。
    多通路命中时按注册表顺序返回（优先 EGFR_RTK 等上游通路）。

    Args:
        text: 用户输入或机制描述文本

    Returns:
        通路类别键（如 "EGFR_RTK"），无命中返回 None
    """
    if not text:
        return None
    for entry in PATHWAY_REGISTRY.values():
        if entry.matches(text):
            return entry.pathway_class
    return None


def get_pathway_entry(pathway_class: str) -> PathwayEntry | None:
    """根据通路类别键获取完整注册信息。

    Args:
        pathway_class: 通路类别键（如 "EGFR_RTK"）

    Returns:
        PathwayEntry 实例，未知通路返回 None
    """
    return PATHWAY_REGISTRY.get(pathway_class)
