import { WorkbenchShell } from "@/components/workspace/WorkbenchShell";

/**
 * /advanced — 归档保留的 4 栏 Scientific Modeling IDE。
 *
 * 极简 Auto-Chat 重构后，原 /workspace 的四栏工作台（Project / Scientific
 * Workspace / Validation / AI Assistant）不再是产品主入口，但作为高级
 * 能力保留在此路由，供需要细粒度调试 / 历史回看的用户使用。
 *
 * 主入口 `/` 已切换为对话式极简体验：One Prompt → One Simulation →
 * One Scientific Report。
 */
export default function AdvancedWorkbenchPage() {
  return <WorkbenchShell />;
}
