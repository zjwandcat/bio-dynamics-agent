# BioDynamics Agent - 从 BioModels 下载经典 SBML 模型并解析为结构化 JSON
# 使用 requests + xml.etree.ElementTree，避免 biopython 在 Python 3.14 下的编译问题。

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# 将 backend 目录加入 Python 路径，以便复用 app 包中的配置与工具
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 默认下载的经典模型列表：MAPK 信号与肿瘤微环境相关
DEFAULT_MODEL_IDS = [
    "BIOMD0000000010",  # Kholodenko 1999 - MAPK cascade (ultrasensitivity)
    "BIOMD0000000012",  # Bhalla & Iyengar 1999 - MAPK signaling
    "BIOMD0000000056",  # Tyson & Novak 2001 - cell cycle regulation
    "BIOMD0000000205",  # Clarke et al. 2006 - TGF-beta/SMAD signaling
    "BIOMD0000000567",  # Wang et al. 2015 - tumor-immune microenvironment
]

BIOMODELS_DOWNLOAD_URL = "https://www.ebi.ac.uk/biomodels/model/download/{model_id}?filename={model_id}_url.xml"


def download_sbml(model_id: str, raw_dir: Path) -> Path:
    """从 BioModels 下载单个 SBML 文件，保存到 raw 目录。"""
    url = BIOMODELS_DOWNLOAD_URL.format(model_id=model_id)
    output_path = raw_dir / f"{model_id}.xml"

    logger.info("正在下载 %s ...", model_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_path.write_text(response.text, encoding="utf-8")
    logger.info("已保存原始 SBML: %s", output_path)
    return output_path


def _parse_value(text: str | None) -> float | None:
    """安全解析数值，失败返回 None。"""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_sbml_to_records(xml_path: Path) -> list[dict]:
    """解析 SBML XML，提取物种与全局参数为结构化参数记录。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # SBML 命名空间处理：使用通用匹配，避免硬编码命名空间
    ns_match = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"sbml": ns_match} if ns_match else {}

    model = root.find(".//sbml:model", ns) or root.find(".//model")
    if model is None:
        logger.warning("%s 中未找到 model 节点", xml_path.name)
        return []

    records: list[dict] = []

    # 1. 提取物种（视为初始浓度参数）
    species_list = model.find(".//sbml:listOfSpecies", ns) or model.find(".//listOfSpecies")
    if species_list is not None:
        for species in species_list:
            sid = species.get("id", "")
            name = species.get("name", sid)
            init_conc = _parse_value(species.get("initialConcentration"))
            init_amount = _parse_value(species.get("initialAmount"))
            value = init_conc if init_conc is not None else init_amount
            unit = species.get("substanceUnits", "nM")
            if value is None:
                continue
            records.append(
                {
                    "param_name": f"initial_concentration_{sid}",
                    "value": value,
                    "unit": unit if unit else "nM",
                    "species": "",
                    "cell_line": "",
                    "context": (
                        f"Species '{name}' (id={sid}) from SBML model {xml_path.stem}. "
                        f"Initial concentration/amount used in the model."
                    ),
                    "confidence": "HIGH",
                    "source_model": xml_path.stem,
                    "source_type": "species",
                }
            )

    # 2. 提取全局参数
    param_list = model.find(".//sbml:listOfParameters", ns) or model.find(".//listOfParameters")
    if param_list is not None:
        for param in param_list:
            pid = param.get("id", "")
            value = _parse_value(param.get("value"))
            unit = param.get("units", "")
            if value is None:
                continue
            records.append(
                {
                    "param_name": pid,
                    "value": value,
                    "unit": unit if unit else "dimensionless",
                    "species": "",
                    "cell_line": "",
                    "context": f"Global parameter '{pid}' from SBML model {xml_path.stem}.",
                    "confidence": "HIGH",
                    "source_model": xml_path.stem,
                    "source_type": "parameter",
                }
            )

    # 3. 提取反应中的局部参数（常见于 BioModels）
    reaction_list = model.find(".//sbml:listOfReactions", ns) or model.find(".//listOfReactions")
    if reaction_list is not None:
        for reaction in reaction_list:
            rid = reaction.get("id", "")
            kl = reaction.find(".//sbml:kineticLaw", ns) or reaction.find(".//kineticLaw")
            if kl is None:
                continue
            local_param_list = kl.find(".//sbml:listOfLocalParameters", ns) or kl.find(
                ".//listOfLocalParameters"
            )
            if local_param_list is None:
                # 兼容旧版 listOfParameters 位于 kineticLaw 下的写法
                local_param_list = kl.find(".//sbml:listOfParameters", ns) or kl.find(
                    ".//listOfParameters"
                )
            if local_param_list is None:
                continue
            for param in local_param_list:
                pid = param.get("id", "")
                value = _parse_value(param.get("value"))
                unit = param.get("units", "")
                if value is None:
                    continue
                records.append(
                    {
                        "param_name": pid,
                        "value": value,
                        "unit": unit if unit else "dimensionless",
                        "species": "",
                        "cell_line": "",
                        "context": (
                            f"Local parameter '{pid}' in reaction '{rid}' "
                            f"from SBML model {xml_path.stem}."
                        ),
                        "confidence": "HIGH",
                        "source_model": xml_path.stem,
                        "source_type": "local_parameter",
                    }
                )

    logger.info("%s 解析完成，共提取 %d 条记录", xml_path.name, len(records))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="从 BioModels 下载并解析 SBML 参数")
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=DEFAULT_MODEL_IDS,
        help="BioModels ID 列表，默认使用 MAPK/肿瘤微环境经典模型",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "raw",
        help="原始 SBML XML 保存目录",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "processed",
        help="解析后的 JSON 保存目录",
    )
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for model_id in args.model_ids:
        try:
            xml_path = download_sbml(model_id, args.raw_dir)
            records = parse_sbml_to_records(xml_path)
            model_json_path = args.processed_dir / f"{model_id}.json"
            model_json_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("已保存解析结果: %s", model_json_path)
            all_records.extend(records)
            # 礼貌性延迟，避免对 EBI 服务器造成压力
            time.sleep(0.5)
        except Exception as exc:
            logger.error("处理 %s 失败: %s", model_id, exc)

    combined_path = args.processed_dir / "all_params.json"
    combined_path.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "全部完成: %d 个模型，共 %d 条参数记录，合并文件: %s",
        len(args.model_ids),
        len(all_records),
        combined_path,
    )


if __name__ == "__main__":
    main()
