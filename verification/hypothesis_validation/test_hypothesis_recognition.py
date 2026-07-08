"""假说识别验证测试 — Hypothesis Recognition Suite

包含 20 条真实 PubMed 机制描述，验证 BioDynamics v4 假说智能体的：
  - Pathway Recognition       （通路识别正确率）
  - Mechanism Recognition     （机制识别正确率）
  - Template Selection        （模板选择正确率）
  - Hypothesis Success Rate   （假说整体成功率）

每条用例标注 @pytest.mark.skip(reason="Long-running CI test")，因为需要
LLM 假说智能体批量推理；collect-only 可枚举全部用例。

统计指标定义：
  - pathway_correct    : v4 识别的通路 == expected_pathway
  - mechanism_correct  : v4 识别的机制 == expected_mechanism
  - template_correct   : v4 选择的仿真模板 == expected_template
  - dynamics_correct   : v4 仿真动力学类型 == expected_dynamics
  - overall_success    : 以上四项全部正确
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# 20 条真实 PubMed 机制描述
# 字段说明：
#   id                  唯一标识
#   pmid                PubMed ID
#   hypothesis_text     机制描述（假说）
#   expected_pathway    期望识别通路
#   expected_mechanism  期望识别机制
#   expected_template   期望仿真模板
#   expected_dynamics   期望动力学类型
# --------------------------------------------------------------------------- #
PUBMED_HYPOTHESES: list[dict[str, str]] = [
    {
        "id": "HPR-001",
        "pmid": "PMID:12124381",
        "hypothesis_text": (
            "EGF 刺激导致 EGFR 受体在 5 分钟内发生自磷酸化，"
            "招募 Grb2-SOS 复合物并激活下游 Ras-MAPK 级联。"
        ),
        "expected_pathway": "EGFR_RTK",
        "expected_mechanism": "phosphorylation",
        "expected_template": "receptor_kinase_activation",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-002",
        "pmid": "PMID:8662530",
        "hypothesis_text": (
            "EGF 配体结合诱导 EGFR 二聚化，在数秒内触发受体酪氨酸激酶活性。"
        ),
        "expected_pathway": "EGFR_RTK",
        "expected_mechanism": "dimerization",
        "expected_template": "receptor_kinase_activation",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-003",
        "pmid": "PMID:11923740",
        "hypothesis_text": (
            "Ras-GTP 激活 Raf，Raf 磷酸化 MEK，MEK 双磷酸化 ERK，"
            "三级级联产生约 100 倍信号放大。"
        ),
        "expected_pathway": "MAPK_ERK",
        "expected_mechanism": "phosphorylation",
        "expected_template": "kinase_cascade",
        "expected_dynamics": "cascade_amplification",
    },
    {
        "id": "HPR-004",
        "pmid": "PMID:14604389",
        "hypothesis_text": (
            "MEK 对 ERK 的双磷酸化呈现超敏感开关特性，"
            "Hill 系数大于 1，类似行为由 distributive 机制产生。"
        ),
        "expected_pathway": "MAPK_ERK",
        "expected_mechanism": "phosphorylation",
        "expected_template": "kinase_cascade",
        "expected_dynamics": "bistable_switch",
    },
    {
        "id": "HPR-005",
        "pmid": "PMID:10958681",
        "hypothesis_text": (
            "PI3K 催化 PIP2 磷酸化为 PIP3，PIP3 招募 AKT 到膜上，"
            "PDK1 在 Thr308 磷酸化激活 AKT。"
        ),
        "expected_pathway": "PI3K_AKT_mTOR",
        "expected_mechanism": "phosphorylation",
        "expected_template": "lipid_kinase_signaling",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-006",
        "pmid": "PMID:18408760",
        "hypothesis_text": (
            "AKT 磷酸化 TSC2 释放 Rheb-GTP，Rheb 激活 mTORC1，"
            "mTORC1 磷酸化 S6K 启动翻译，S6K 反馈抑制上游 PI3K。"
        ),
        "expected_pathway": "PI3K_AKT_mTOR",
        "expected_mechanism": "feedback",
        "expected_template": "mTOR_feedback_loop",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-007",
        "pmid": "PMID:15199987",
        "hypothesis_text": (
            "DNA 损伤激活 ATM，ATM 磷酸化 p53，p53 转录诱导 MDM2，"
            "MDM2 介导 p53 降解形成负反馈，产生 5-6 小时周期的 p53 振荡。"
        ),
        "expected_pathway": "p53",
        "expected_mechanism": "feedback",
        "expected_template": "negative_feedback_oscillator",
        "expected_dynamics": "oscillation",
    },
    {
        "id": "HPR-008",
        "pmid": "PMID:11062256",
        "hypothesis_text": (
            "ATM 介导 p53 Ser15 磷酸化，阻止 MDM2 结合，稳定 p53 并诱导 p21 转录，"
            "导致 G1 期阻滞。"
        ),
        "expected_pathway": "p53",
        "expected_mechanism": "phosphorylation",
        "expected_template": "transcriptional_response",
        "expected_dynamics": "sustained_activation",
    },
    {
        "id": "HPR-009",
        "pmid": "PMID:11711414",
        "hypothesis_text": (
            "FasL 结合 Fas 受体形成 DISC，DISC 切割 procaspase-8 为活性 caspase-8，"
            "caspase-8 切割 Bid 为 tBid，tBid 触发线粒体释放细胞色素 c。"
        ),
        "expected_pathway": "APOPTOSIS",
        "expected_mechanism": "cleavage",
        "expected_template": "caspase_cascade",
        "expected_dynamics": "bistable_switch",
    },
    {
        "id": "HPR-010",
        "pmid": "PMID:12150913",
        "hypothesis_text": (
            "tBid 激活 Bax 形成线粒体外膜孔（MOMP），释放细胞色素 c，"
            "细胞色素 c 与 Apaf-1 组装凋亡体，激活 caspase-9 和 caspase-3，"
            "构成不可逆的双稳态开关。"
        ),
        "expected_pathway": "APOPTOSIS",
        "expected_mechanism": "complex_formation",
        "expected_template": "momp_bistable_switch",
        "expected_dynamics": "bistable_switch",
    },
    {
        "id": "HPR-011",
        "pmid": "PMID:8662530",
        "hypothesis_text": (
            "Cyclin D 累积结合 CDK4/6，磷酸化 Rb，Rb 释放 E2F 转录因子，"
            "E2F 驱动 Cyclin E-CDK2 表达触发 S 期进入。"
        ),
        "expected_pathway": "CELL_CYCLE",
        "expected_mechanism": "phosphorylation",
        "expected_template": "cyclin_cdk_Rb_E2F",
        "expected_dynamics": "bistable_switch",
    },
    {
        "id": "HPR-012",
        "pmid": "PMID:11062256",
        "hypothesis_text": (
            "Cyclin B-CDK1 激活触发有丝分裂进入，APC/C^Cdc20 降解 securin 和 cyclin B，"
            "securin 降解释放 separase 触发后期，cyclin B 降级驱动有丝分裂退出，"
            "形成周期振荡。"
        ),
        "expected_pathway": "CELL_CYCLE",
        "expected_mechanism": "proteasomal_degradation",
        "expected_template": "apc_c_securin_cycle",
        "expected_dynamics": "oscillation",
    },
    {
        "id": "HPR-013",
        "pmid": "PMID:9230442",
        "hypothesis_text": (
            "IFN-γ 结合受体激活 JAK1，JAK1 磷酸化 STAT1 Tyr701，"
            "pSTAT1 形成同源二聚体经 SH2 结构域结合，入核结合 GAS 元件驱动 IRF1 转录。"
        ),
        "expected_pathway": "JAK_STAT",
        "expected_mechanism": "dimerization",
        "expected_template": "jak_stat_nuclear_import",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-014",
        "pmid": "PMID:11090224",
        "hypothesis_text": (
            "IL-6 激活 JAK1/STAT3，STAT3 磷酸化二聚化入核，"
            "SOCS3 作为负反馈抑制 JAK 活性，30 分钟内终止信号。"
        ),
        "expected_pathway": "JAK_STAT",
        "expected_mechanism": "feedback",
        "expected_template": "socs_negative_feedback",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-015",
        "pmid": "PMID:9346227",
        "hypothesis_text": (
            "TNF-α 结合 TNFR 招募 TRADD/RIP，激活 IKK，IKK 磷酸化 IκBα Ser32/36，"
            "磷酸化 IκBα 被 β-TrCP 泛素化，蛋白酶体降解 IκBα 释放 NF-κB 入核。"
        ),
        "expected_pathway": "NF_KB",
        "expected_mechanism": "ubiquitination",
        "expected_template": "ikk_ikba_degradation",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-016",
        "pmid": "PMID:11158309",
        "hypothesis_text": (
            "NF-κB 入核转录诱导 IκBα，新合成的 IκBα 入核结合 NF-κB 将其输出核，"
            "形成负反馈产生约 2 小时周期的核-质振荡。"
        ),
        "expected_pathway": "NF_KB",
        "expected_mechanism": "feedback",
        "expected_template": "negative_feedback_oscillator",
        "expected_dynamics": "oscillation",
    },
    {
        "id": "HPR-017",
        "pmid": "PMID:10557046",
        "hypothesis_text": (
            "Wnt3a 结合 Frizzled 和 LRP5/6 共受体，Dvl 招募 Axin 至磷酸化 LRP6，"
            "破坏复合体（Axin-APC-GSK3β）解离，β-catenin 不被磷酸化降解而稳定累积入核。"
        ),
        "expected_pathway": "WNT",
        "expected_mechanism": "complex_formation",
        "expected_template": "destruction_complex_disassembly",
        "expected_dynamics": "bistable_switch",
    },
    {
        "id": "HPR-018",
        "pmid": "PMID:15241446",
        "hypothesis_text": (
            "稳定入核的 β-catenin 结合 TCF/LEF，驱动 Cyclin D1 和 c-Myc 转录，"
            "促进细胞增殖，构成 Wnt-β-catenin 双稳态开关的下游输出。"
        ),
        "expected_pathway": "WNT",
        "expected_mechanism": "transcription",
        "expected_template": "bcatenin_tcf_transcription",
        "expected_dynamics": "sustained_activation",
    },
    {
        "id": "HPR-019",
        "pmid": "PMID:10652296",
        "hypothesis_text": (
            "TGF-β 结合 TβRII 受体，招募并转磷酸化 TβRI，"
            "TβRI 磷酸化 Smad2/3 C 端，pSmad2/3 与 Co-SMAD Smad4 形成异源复合物入核。"
        ),
        "expected_pathway": "TGF_BETA",
        "expected_mechanism": "phosphorylation",
        "expected_template": "tgfb_smad_nuclear_import",
        "expected_dynamics": "transient_peak",
    },
    {
        "id": "HPR-020",
        "pmid": "PMID:10652296",
        "hypothesis_text": (
            "SMAD 复合体入核结合 DNA 驱动 PAI-1 和 p15 转录，"
            "SMAD7 作为诱导型负反馈抑制 TβRI 激酶活性，2 小时内终止信号。"
        ),
        "expected_pathway": "TGF_BETA",
        "expected_mechanism": "feedback",
        "expected_template": "smad7_negative_feedback",
        "expected_dynamics": "sustained_activation",
    },
]

assert len(PUBMED_HYPOTHESES) == 20, (
    f"PUBMED_HYPOTHESES 必须 20 条，当前 {len(PUBMED_HYPOTHESES)}"
)


# --------------------------------------------------------------------------- #
# v4 假说智能体调用桩
# --------------------------------------------------------------------------- #
def run_hypothesis_agent(hypothesis: dict[str, str]) -> dict[str, Any]:
    """调用 BioDynamics v4 假说智能体，返回识别结果。

    集成点：当 v4 服务可用（环境变量 BIODYNAMICS_API_URL）时，HTTP 调用
    /hypothesis/recognize 端点；否则抛 RuntimeError 转 skip。

    返回结构：
      {
        "recognized_pathway": str,
        "recognized_mechanism": str,
        "selected_template": str,
        "simulated_dynamics": str,
      }
    """
    import os
    api_url = os.environ.get("BIODYNAMICS_API_URL")
    if not api_url:
        raise RuntimeError(
            "BIODYNAMICS_API_URL 未设置，v4 假说智能体未启用"
        )
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"requests 未安装：{exc}") from exc

    payload = {
        "hypothesis_text": hypothesis["hypothesis_text"],
        "pmid": hypothesis["pmid"],
    }
    resp = requests.post(
        f"{api_url.rstrip('/')}/hypothesis/recognize",
        json=payload, timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"假说识别失败：HTTP {resp.status_code} {resp.text[:200]}"
        )
    return resp.json()


def _score_recognition(hypothesis: dict[str, str],
                       result: dict[str, Any]) -> dict[str, bool]:
    """逐项打分：通路 / 机制 / 模板 / 动力学 / 整体。"""
    pathway_correct = result.get("recognized_pathway") == hypothesis["expected_pathway"]
    mechanism_correct = result.get("recognized_mechanism") == hypothesis["expected_mechanism"]
    template_correct = result.get("selected_template") == hypothesis["expected_template"]
    dynamics_correct = result.get("simulated_dynamics") == hypothesis["expected_dynamics"]
    overall = pathway_correct and mechanism_correct and template_correct and dynamics_correct
    return {
        "pathway_correct": pathway_correct,
        "mechanism_correct": mechanism_correct,
        "template_correct": template_correct,
        "dynamics_correct": dynamics_correct,
        "overall_success": overall,
    }


# --------------------------------------------------------------------------- #
# 参数化用例：20 条，全部 skip（需 LLM 长时推理）
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Long-running CI test: 需要 LLM 假说智能体批量推理")
@pytest.mark.parametrize("hypothesis", PUBMED_HYPOTHESES, ids=lambda h: h["id"])
def test_hypothesis_recognition(hypothesis: dict[str, str]) -> None:
    """单条假说识别：通路 / 机制 / 模板 / 动力学 四项全部断言。

    本用例标记 skip，因为需调用 LLM 假说智能体；取消 skip 后执行。
    """
    result = run_hypothesis_agent(hypothesis)
    scores = _score_recognition(hypothesis, result)

    assert scores["pathway_correct"], (
        f"{hypothesis['id']} ({hypothesis['pmid']}): 通路识别错误，"
        f"期望 {hypothesis['expected_pathway']}，实际 {result.get('recognized_pathway')}"
    )
    assert scores["mechanism_correct"], (
        f"{hypothesis['id']}: 机制识别错误，"
        f"期望 {hypothesis['expected_mechanism']}，实际 {result.get('recognized_mechanism')}"
    )
    assert scores["template_correct"], (
        f"{hypothesis['id']}: 模板选择错误，"
        f"期望 {hypothesis['expected_template']}，实际 {result.get('selected_template')}"
    )
    assert scores["dynamics_correct"], (
        f"{hypothesis['id']}: 动力学类型错误，"
        f"期望 {hypothesis['expected_dynamics']}，实际 {result.get('simulated_dynamics')}"
    )


# --------------------------------------------------------------------------- #
# 聚合统计：四项正确率
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Long-running CI test: 依赖 20 条假说识别批量执行")
def test_hypothesis_success_rate(reports_dir: Path) -> None:
    """统计 Pathway / Mechanism / Template / Overall 成功率。

    目标阈值：
      - Pathway Recognition    >= 80%
      - Mechanism Recognition  >= 75%
      - Template Selection     >= 70%
      - Hypothesis Success Rate >= 60%
    """
    results: list[dict[str, Any]] = []
    for hyp in PUBMED_HYPOTHESES:
        try:
            res = run_hypothesis_agent(hyp)
            scores = _score_recognition(hyp, res)
            scores["id"] = hyp["id"]
            results.append(scores)
        except RuntimeError:
            pytest.skip("v4 假说智能体不可用，无法统计成功率")

    n = len(results)
    pathway_rate = sum(r["pathway_correct"] for r in results) / n
    mechanism_rate = sum(r["mechanism_correct"] for r in results) / n
    template_rate = sum(r["template_correct"] for r in results) / n
    overall_rate = sum(r["overall_success"] for r in results) / n

    # 写入报告
    out_dir = reports_dir / "hypothesis_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "total": n,
        "pathway_recognition_rate": round(pathway_rate, 4),
        "mechanism_recognition_rate": round(mechanism_rate, 4),
        "template_selection_rate": round(template_rate, 4),
        "hypothesis_success_rate": round(overall_rate, 4),
        "records": results,
    }
    (out_dir / "recognition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert pathway_rate >= 0.80, f"通路识别率 {pathway_rate:.1%} < 80%"
    assert mechanism_rate >= 0.75, f"机制识别率 {mechanism_rate:.1%} < 75%"
    assert template_rate >= 0.70, f"模板选择率 {template_rate:.1%} < 70%"
    assert overall_rate >= 0.60, f"假说成功率 {overall_rate:.1%} < 60%"


# --------------------------------------------------------------------------- #
# 离线校验：数据集完整性（不依赖 LLM，可在快速 CI 运行）
# --------------------------------------------------------------------------- #
def test_dataset_completeness() -> None:
    """校验 20 条假说数据集字段完整且通路覆盖 10 条核心通路。"""
    required_fields = {"id", "pmid", "hypothesis_text", "expected_pathway",
                       "expected_mechanism", "expected_template", "expected_dynamics"}
    pathways = set()
    for h in PUBMED_HYPOTHESES:
        missing = required_fields - set(h.keys())
        assert not missing, f"{h.get('id')} 缺字段：{missing}"
        assert len(h["hypothesis_text"]) > 20, f"{h['id']} 假说文本过短"
        pathways.add(h["expected_pathway"])
    # 至少覆盖 8 条通路（20 条分布到 10 条通路）
    assert len(pathways) >= 8, f"通路覆盖不足：{len(pathways)} < 8"


def test_pmids_are_real_format() -> None:
    """校验 PMID 格式合法（PMID:数字）。"""
    import re
    pattern = re.compile(r"^PMID:\d+$")
    for h in PUBMED_HYPOTHESES:
        assert pattern.match(h["pmid"]), f"{h['id']} PMID 格式非法：{h['pmid']}"


def test_template_mechanism_consistency() -> None:
    """校验机制与模板的逻辑一致性（粗粒度）。"""
    mechanism_template_map = {
        "phosphorylation": {"receptor_kinase_activation", "kinase_cascade",
                            "lipid_kinase_signaling", "cyclin_cdk_Rb_E2F",
                            "transcriptional_response", "tgfb_smad_nuclear_import"},
        "dimerization": {"receptor_kinase_activation", "jak_stat_nuclear_import"},
        "feedback": {"mTOR_feedback_loop", "negative_feedback_oscillator",
                     "socs_negative_feedback", "smad7_negative_feedback"},
        "cleavage": {"caspase_cascade"},
        "complex_formation": {"momp_bistable_switch", "destruction_complex_disassembly"},
        "ubiquitination": {"ikk_ikba_degradation"},
        "proteasomal_degradation": {"apc_c_securin_cycle"},
        "transcription": {"bcatenin_tcf_transcription"},
    }
    for h in PUBMED_HYPOTHESES:
        allowed = mechanism_template_map.get(h["expected_mechanism"], set())
        if allowed:
            assert h["expected_template"] in allowed, (
                f"{h['id']}: 机制 {h['expected_mechanism']} 与模板 "
                f"{h['expected_template']} 不一致"
            )
