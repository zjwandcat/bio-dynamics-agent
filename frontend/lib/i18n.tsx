"use client";

/**
 * 极简 I18n 上下文。
 *
 * 设计目标：
 *   1. 开源用户可在中英文之间切换界面标签。
 *   2. 不引入 next-intl / react-i18next 等重型依赖，保持 bundle 精简。
 *   3. 通过 Zustand 持久化语言偏好（localStorage），刷新后保持。
 *
 * 扩展方式：
 *   在 translations 字典中添加 key，组件中用 const { t } = useTranslation() 读取。
 */

import React, {
  createContext,
  useContext,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";

type Locale = "zh" | "en";

interface I18nContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
  t: (key: string, fallback?: string, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

export const DEFAULT_LOCALE: Locale = "zh";

function getInitialLocale(): Locale {
  if (typeof window === "undefined") return DEFAULT_LOCALE;
  const stored = window.localStorage.getItem("biodynamics-locale");
  if (stored === "zh" || stored === "en") return stored;
  // 与 SSR 默认值保持一致，避免 hydration mismatch。
  // 用户可通过右上角语言按钮显式切换。
  return DEFAULT_LOCALE;
}

const translations: Record<Locale, Record<string, string>> = {
  zh: {
    // Workbench header
    "app.name": "BioDynamics",
    "header.pathway": "通路",
    "header.pathway.placeholder": "— 选择通路 —",
    "header.run": "运行",
    "header.ai.toggle.open": "展开 AI 助手",
    "header.ai.toggle.close": "折叠 AI 助手",
    "header.ai": "AI",
    "header.updateKb": "更新知识库",
    "header.kbStatus": "知识库状态",
    "header.lang": "语言",

    // Pane titles
    "pane.project": "Project / Pathway",
    "pane.workspace": "Scientific Workspace",
    "pane.validation": "Validation",
    "pane.ai": "AI Assistant",

    // AI Assistant
    "ai.title": "AI Assistant",
    "ai.tab.chat": "对话",
    "ai.tab.suggestions": "建议",
    "ai.tab.logs": "日志",
    "ai.placeholder": "输入生物学假说，AI 将自动建模并运行仿真。",
    "ai.stop": "停止生成",
    "ai.clear": "清除当前对话",
    "ai.running": "运行中",
    "ai.workflow": "工作流",

    // Validation
    "validation.pending": "Validation pending",
    "validation.pending.hint": "运行仿真以开始验证",
    "validation.notRun": "Not run",
    "validation.levelsPassed": "{passed}/{total} levels passed",
    "validation.failed": "validation failed",
    "validation.clear": "validation clear",
    "validation.allPass": "All Pass",
    "validation.failedBadge": "Failed",
    "validation.level1": "Internal Consistency",
    "validation.level2": "SBML / BioModels",
    "validation.level3": "Cross-Pathway",
    "validation.level4": "Benchmark",
    "validation.level5": "Hypothesis",

    // Parameter explorer
    "params.slider": "滑块",
    "params.ic50": "IC50/EC50",
    "params.knockout": "敲除",
    "params.overexpr": "过表达",
    "params.mutation": "突变",
    "params.apply": "应用更改",
    "params.reset": "重置基线",
    "params.reSimulating": "重新仿真中…",
    "params.allBaseline": "所有参数均为基线",
    "params.modified": "已修改",
    "params.override": "覆盖",
    "params.clear": "清除",
    "params.activePreset": "当前预设",
    "params.previewHint": "预览 — 点击“应用更改”对当前通路执行参数扫描。",
    "params.knockoutHint": "将某物种初始浓度强制设为 0（敲除）。",
    "params.overexprHint": "过表达某物种（相对于基线的倍数）。1× = 无变化。",
    "params.mutationHint": "预设突变会直接修改动力学参数。",
    "params.customOverride": "自定义覆盖",
    "params.nonePreset": "无（野生型）",
    "params.required": "必须",

    // Simulation panel
    "sim.title": "仿真",
    "sim.run": "运行仿真",
    "sim.retry": "重试",
    "sim.empty": "运行仿真以查看结果",
    "sim.running": "仿真运行中…",
    "sim.error": "仿真失败",
    "sim.status.idle": "空闲",
    "sim.status.running": "运行中",
    "sim.status.complete": "完成",
    "sim.status.error": "错误",
    "sim.tab.timeSeries": "时序",
    "sim.tab.doseResponse": "剂量响应",
    "sim.tab.sensitivity": "敏感性",
    "sim.tab.phasePortrait": "相图",
    "sim.tab.steadyState": "稳态",
    "sim.tab.oscillation": "振荡",

    // Run controls
    "run.mode": "运行模式",
    "run.modules": "模块勾选",
    "run.autoFast": "自动快速",
    "run.autoFast.desc": "极简流程，单智能体直跑",
    "run.autoStandard": "自动标准",
    "run.autoStandard.desc": "默认模式，LLM 动态裁剪流程",
    "run.manual": "手动",
    "run.manual.desc": "用户勾选所需模块",
    "run.module.terminology": "术语标准化 (MCP)",
    "run.module.mechanism": "机制解析与图谱",
    "run.module.rag": "知识检索 (RAG)",
    "run.module.pkpd": "PK/PD 推断",
    "run.module.sandbox": "沙箱仿真执行",
    "run.module.dose": "剂量递增分析",
    "run.module.evidence": "实验与文献检索",
    "run.module.report": "预测报告生成",
    "run.required": "必须",
    "run.tooltip.deps": "已自动补充依赖项：沙箱仿真执行、机制解析与图谱",

    // Pathway tree
    "pathway.loading": "加载通路中…",
    "pathway.autoDetect": "自动检测",
    "pathway.autoDetect.hint": "让后端从假说中自动推断通路",
    "pathway.selection": "通路选择",
    "pathway.offline": "离线",
    "benchmarks.title": "基准测试",
    "benchmarks.runAll": "全部运行",
    "simHistory.title": "仿真历史",

    // Validation detail fallback
    "validation.noChecks": "该层级暂无可用检查项。",
  },
  en: {
    // Workbench header
    "app.name": "BioDynamics",
    "header.pathway": "Pathway",
    "header.pathway.placeholder": "— Select pathway —",
    "header.run": "Run",
    "header.ai.toggle.open": "Open AI Assistant",
    "header.ai.toggle.close": "Close AI Assistant",
    "header.ai": "AI",
    "header.updateKb": "Update KB",
    "header.kbStatus": "KB Status",
    "header.lang": "Lang",

    // Pane titles
    "pane.project": "Project / Pathway",
    "pane.workspace": "Scientific Workspace",
    "pane.validation": "Validation",
    "pane.ai": "AI Assistant",

    // AI Assistant
    "ai.title": "AI Assistant",
    "ai.tab.chat": "Chat",
    "ai.tab.suggestions": "Suggestions",
    "ai.tab.logs": "Logs",
    "ai.placeholder": "Type a biological hypothesis; AI will model and simulate it.",
    "ai.stop": "Stop generation",
    "ai.clear": "Clear chat",
    "ai.running": "running",
    "ai.workflow": "Workflow",

    // Validation
    "validation.pending": "Validation pending",
    "validation.pending.hint": "Run a simulation to validate",
    "validation.notRun": "Not run",
    "validation.levelsPassed": "{passed}/{total} levels passed",
    "validation.failed": "validation failed",
    "validation.clear": "validation clear",
    "validation.allPass": "All Pass",
    "validation.failedBadge": "Failed",
    "validation.level1": "Internal Consistency",
    "validation.level2": "SBML / BioModels",
    "validation.level3": "Cross-Pathway",
    "validation.level4": "Benchmark",
    "validation.level5": "Hypothesis",

    // Parameter explorer
    "params.slider": "Slider",
    "params.ic50": "IC50/EC50",
    "params.knockout": "Knockout",
    "params.overexpr": "Overexpr.",
    "params.mutation": "Mutation",
    "params.apply": "Apply Changes",
    "params.reset": "Reset to Baseline",
    "params.reSimulating": "re-simulating…",
    "params.allBaseline": "All parameters at baseline",
    "params.modified": "modified",
    "params.override": "override",
    "params.clear": "clear",
    "params.activePreset": "Active preset",
    "params.previewHint": "Preview only — Apply Changes runs a parameter sweep against the current pathway.",
    "params.knockoutHint": "Force a species concentration to 0 (knockout) at t=0.",
    "params.overexprHint": "Overexpress a species (fold change over baseline). 1× = no change.",
    "params.mutationHint": "Preset mutations alter kinetic parameters directly.",
    "params.customOverride": "Custom override",
    "params.nonePreset": "None (wild-type)",
    "params.required": "required",

    // Simulation panel
    "sim.title": "Simulation",
    "sim.run": "Run Simulation",
    "sim.retry": "Retry",
    "sim.empty": "Run a simulation to see results",
    "sim.running": "Running simulation…",
    "sim.error": "Simulation failed",
    "sim.status.idle": "idle",
    "sim.status.running": "running",
    "sim.status.complete": "complete",
    "sim.status.error": "error",
    "sim.tab.timeSeries": "Time Series",
    "sim.tab.doseResponse": "Dose Response",
    "sim.tab.sensitivity": "Sensitivity",
    "sim.tab.phasePortrait": "Phase Portrait",
    "sim.tab.steadyState": "Steady State",
    "sim.tab.oscillation": "Oscillation",

    // Run controls
    "run.mode": "Run Mode",
    "run.modules": "Modules",
    "run.autoFast": "Auto Fast",
    "run.autoFast.desc": "Minimal pipeline, single-agent run",
    "run.autoStandard": "Auto Standard",
    "run.autoStandard.desc": "Default mode, LLM dynamically trims flow",
    "run.manual": "Manual",
    "run.manual.desc": "User selects required modules",
    "run.module.terminology": "Terminology (MCP)",
    "run.module.mechanism": "Mechanism & Graph",
    "run.module.rag": "Knowledge Retrieval (RAG)",
    "run.module.pkpd": "PK/PD Inference",
    "run.module.sandbox": "Sandbox Execution",
    "run.module.dose": "Dose Escalation",
    "run.module.evidence": "Evidence Retrieval",
    "run.module.report": "Report Generation",
    "run.required": "required",
    "run.tooltip.deps": "Auto-added dependencies: sandbox execution, mechanism graph",

    // Pathway tree
    "pathway.loading": "Loading pathways…",
    "pathway.autoDetect": "Auto Detect",
    "pathway.autoDetect.hint": "Let backend infer pathway from hypothesis",
    "pathway.selection": "Pathway Selection",
    "pathway.offline": "offline",
    "benchmarks.title": "BENCHMARKS",
    "benchmarks.runAll": "Run All",
    "simHistory.title": "SIMULATION HISTORY",

    // Validation detail fallback
    "validation.noChecks": "No checks available for this level.",
  },
};

function interpolate(template: string, vars: Record<string, string | number>) {
  return template.replace(/\{(\w+)\}/g, (_, key) => String(vars[key] ?? `{${key}}`));
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = React.useState<Locale>(() => getInitialLocale());

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("biodynamics-locale", next);
      document.documentElement.lang = next === "zh" ? "zh-CN" : "en";
    }
  }, []);

  const toggleLocale = useCallback(() => {
    setLocale(locale === "zh" ? "en" : "zh");
  }, [locale, setLocale]);

  const t = useCallback(
    (key: string, fallback?: string, vars?: Record<string, string | number>) => {
      const text = translations[locale][key] ?? fallback ?? key;
      return vars ? interpolate(text, vars) : text;
    },
    [locale]
  );

  const value = useMemo(
    () => ({ locale, setLocale, toggleLocale, t }),
    [locale, setLocale, toggleLocale, t]
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useTranslation() {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useTranslation must be used within I18nProvider");
  }
  return ctx;
}
