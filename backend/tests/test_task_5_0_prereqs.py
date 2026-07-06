"""Task 5.0 — P5 前置检查测试

验证：
- SubTask 5.0.1: P4 全部完成（通过 git log 确认，此处仅验证 P4 关键模块可导入）
- SubTask 5.0.2: 依赖隔离策略（try-import 模板）正确实现
"""
from __future__ import annotations


# =============================================================================
# SubTask 5.0.1: P4 完成验证（关键模块可导入）
# =============================================================================
class TestP4Completion:
    """验证 Phase 4 关键模块可导入（间接确认 P4 已完成）。"""

    def test_pathway_planner_importable(self):
        from app.pathways.pathway_planner import pathway_planner_hook_node
        assert callable(pathway_planner_hook_node)

    def test_specialist_registry_importable(self):
        from app.pathways.pathway_registry import SPECIALIST_REGISTRY
        assert isinstance(SPECIALIST_REGISTRY, dict)

    def test_crosstalk_coordinator_importable(self):
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        assert callable(crosstalk_coordinator_hook_node)

    def test_specialist_hook_importable(self):
        from app.pathways.specialist_hook import specialist_hook_node
        assert callable(specialist_hook_node)

    def test_p4_feature_flags_exist(self):
        from app.config import settings
        assert hasattr(settings, "V4_PATHWAY_PLANNER_ENABLED")
        assert hasattr(settings, "V4_PATHWAY_SPECIALIST_ENABLED")
        assert hasattr(settings, "V4_CROSSTALK_COORDINATOR_ENABLED")

    def test_p4_feature_flags_default_false(self):
        from app.config import settings
        # 默认值必须为 False（铁律：flag 默认 OFF）
        assert settings.V4_PATHWAY_PLANNER_ENABLED is False
        assert settings.V4_PATHWAY_SPECIALIST_ENABLED is False
        assert settings.V4_CROSSTALK_COORDINATOR_ENABLED is False


# =============================================================================
# SubTask 5.0.2: 依赖隔离策略验证
# =============================================================================
class TestDependencyIsolationStrategy:
    """验证 try-import 依赖隔离策略正确实现。"""

    def test_roadrunner_availability_flag(self):
        from app.config import ROADRUNNER_AVAILABLE, ROADRUNNER_VERSION
        assert isinstance(ROADRUNNER_AVAILABLE, bool)
        # 版本字符串或 None
        assert ROADRUNNER_VERSION is None or isinstance(ROADRUNNER_VERSION, str)

    def test_lmfit_availability_flag(self):
        from app.config import LMFIT_AVAILABLE, LMFIT_VERSION
        assert isinstance(LMFIT_AVAILABLE, bool)
        assert LMFIT_VERSION is None or isinstance(LMFIT_VERSION, str)

    def test_salib_availability_flag(self):
        from app.config import SALIB_AVAILABLE, SALIB_VERSION
        assert isinstance(SALIB_AVAILABLE, bool)
        assert SALIB_VERSION is None or isinstance(SALIB_VERSION, str)

    def test_lxml_availability_flag(self):
        from app.config import LXML_AVAILABLE, LXML_VERSION
        assert isinstance(LXML_AVAILABLE, bool)
        assert LXML_VERSION is None or isinstance(LXML_VERSION, str)

    def test_p5_feature_flags_exist(self):
        from app.config import settings
        assert hasattr(settings, "V4_SBML_GROUNDER_ENABLED")
        assert hasattr(settings, "V4_VALIDATION_PYRAMID_ENABLED")
        assert hasattr(settings, "V4_CALIBRATION_AGENT_ENABLED")

    def test_p5_feature_flags_default_false(self):
        from app.config import settings
        # 默认值必须为 False（铁律：flag 默认 OFF）
        assert settings.V4_SBML_GROUNDER_ENABLED is False
        assert settings.V4_VALIDATION_PYRAMID_ENABLED is False
        assert settings.V4_CALIBRATION_AGENT_ENABLED is False

    def test_try_import_does_not_raise(self):
        """验证 try-import 模板不会因依赖缺失而抛异常（铁律：失败降级不阻塞）。"""
        # 多次导入 config 不应抛异常
        import importlib
        import app.config
        importlib.reload(app.config)
        # 再次验证标志可读
        from app.config import (
            ROADRUNNER_AVAILABLE,
            LMFIT_AVAILABLE,
            SALIB_AVAILABLE,
            LXML_AVAILABLE,
        )
        assert isinstance(ROADRUNNER_AVAILABLE, bool)
        assert isinstance(LMFIT_AVAILABLE, bool)
        assert isinstance(SALIB_AVAILABLE, bool)
        assert isinstance(LXML_AVAILABLE, bool)
