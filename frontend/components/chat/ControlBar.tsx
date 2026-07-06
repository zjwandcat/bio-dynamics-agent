"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ClarificationDialog,
  ClarificationRequest,
  ClarificationAnswer,
} from "./ClarificationDialog";
import { Zap, Settings2, Hand, Activity, Cpu, Search, FlaskConical, FileText, Beaker, Library, AlertCircle, CheckCircle2, Database, Server } from "lucide-react";

export type RunMode = "auto_fast" | "auto_standard" | "manual";

export interface ControlBarState {
  mode: RunMode;
  manualModules: string[];
}

// 大模型供应商/模型状态接口
export interface ModelStatusData {
  llm: {
    provider: string;
    model: string;
    base_url: string;
  };
  backup_llm: {
    provider: string;
    model: string;
    base_url: string;
  } | null;
  embedding: {
    provider: string;
    model: string;
  };
  rerank: {
    provider: string;
    selection_mode: string;
    provider_priority: string[];
    candidates: Array<{ provider: string; model: string; display_name: string }>;
  };
}

// RAG 知识库状态接口
export interface RagStatusData {
  databases: Array<{ name: string; type: string; collection?: string; files?: string[] }>;
  collections: Record<string, number>;
  online_fallback_enabled: boolean;
  online_fallback_threshold: number;
}

export interface ControlBarProps {
  state: ControlBarState;
  onChange: (state: ControlBarState) => void;
  currentNode?: string;
  tokenUsage?: number;
  isStreaming?: boolean;
  clarification?: ClarificationRequest | null;
  onClarificationSubmit?: (answer: ClarificationAnswer) => void;
  onStop?: () => void;
  ragStatus?: RagStatusData | null;
}

const MODES: { value: RunMode; label: string; icon: React.ReactNode; desc: string }[] = [
  { value: "auto_fast", label: "Auto Fast", icon: <Zap className="h-4 w-4" />, desc: "极简流程，单智能体直跑" },
  { value: "auto_standard", label: "Auto Standard", icon: <Settings2 className="h-4 w-4" />, desc: "默认模式，LLM 动态裁剪流程" },
  { value: "manual", label: "Manual", icon: <Hand className="h-4 w-4" />, desc: "用户勾选所需模块" },
];

const MODULES: { key: string; label: string; icon: React.ReactNode; required?: boolean }[] = [
  { key: "terminology_mcp", label: "术语标准化 (MCP)", icon: <Activity className="h-4 w-4" /> },
  { key: "mechanism_graph", label: "机制解析与图谱", icon: <Cpu className="h-4 w-4" /> },
  { key: "mechanism_parameter_rag", label: "知识检索 (RAG)", icon: <Search className="h-4 w-4" /> },
  { key: "pkpd_inference", label: "PK/PD 推断", icon: <FlaskConical className="h-4 w-4" /> },
  { key: "sandbox_execute", label: "沙箱仿真执行", icon: <Beaker className="h-4 w-4" />, required: true },
  { key: "dose_analysis", label: "剂量递增分析", icon: <Activity className="h-4 w-4" /> },
  { key: "experiment_evidence_rag", label: "实验与文献检索", icon: <Library className="h-4 w-4" /> },
  { key: "report_generation", label: "预测报告生成", icon: <FileText className="h-4 w-4" /> },
];

const MODULE_ORDER = MODULES.map((m) => m.key);
const API_BASE = "http://localhost:8000";

export function ControlBar({ state, onChange, currentNode, tokenUsage, isStreaming, clarification, onClarificationSubmit, onStop, ragStatus }: ControlBarProps) {
  const { mode, manualModules } = state;
  const [tooltip, setTooltip] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusData | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/models/status`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: ModelStatusData) => {
        if (!cancelled) setModelStatus(data);
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("[ControlBar] 获取模型状态失败:", err);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (mode !== "manual") return;
    const hasReport = manualModules.includes("report_generation");
    const hasSandbox = manualModules.includes("sandbox_execute");
    const hasMechanism = manualModules.includes("mechanism_graph");
    if (hasReport && (!hasSandbox || !hasMechanism)) {
      const added = [hasSandbox ? null : "sandbox_execute", hasMechanism ? null : "mechanism_graph"].filter(Boolean) as string[];
      const next = [...manualModules, ...added].sort((a, b) => MODULE_ORDER.indexOf(a) - MODULE_ORDER.indexOf(b));
      onChange({ ...state, manualModules: next });
      setTooltip("已自动补充依赖项：沙箱仿真执行、机制解析与图谱");
      const t = setTimeout(() => setTooltip(null), 3000);
      return () => clearTimeout(t);
    }
  }, [manualModules, mode, onChange, state]);

  const setMode = (m: RunMode) => onChange({ ...state, mode: m });

  const toggleModule = (key: string) => {
    if (mode !== "manual") return;
    if (MODULES.find((m) => m.key === key)?.required) return;
    const next = manualModules.includes(key) ? manualModules.filter((k) => k !== key) : [...manualModules, key];
    onChange({ ...state, manualModules: next });
  };

  const nodeLabel = currentNode ? currentNode.replace("worker_", "").replace("_", " ") : "待机";
  const modeLabel = MODES.find((m) => m.value === mode)?.label ?? mode;

  return (
    <aside className="flex h-full w-[30%] min-w-[360px] flex-col border-l border-zinc-800 bg-zinc-900/95">
      <ScrollArea className="flex-1">
        <div className="space-y-5 p-4">
          <Card className="border-zinc-800 bg-zinc-900">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold text-zinc-100">运行模式</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {MODES.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  disabled={isStreaming}
                  onClick={() => setMode(opt.value)}
                  className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors ${
                    mode === opt.value ? "border-blue-500/50 bg-blue-500/10 text-blue-100" : "border-zinc-800 bg-zinc-950/50 text-zinc-300 hover:bg-zinc-800"
                  } ${isStreaming ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  <span className={mode === opt.value ? "text-blue-400" : "text-zinc-500"}>{opt.icon}</span>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{opt.label}</div>
                    <div className="text-xs text-zinc-500">{opt.desc}</div>
                  </div>
                  {mode === opt.value && <CheckCircle2 className="h-4 w-4 text-blue-400" />}
                </button>
              ))}
            </CardContent>
          </Card>

          {mode === "manual" && (
            <Card className="border-zinc-800 bg-zinc-900">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold text-zinc-100">模块勾选</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {tooltip && (
                  <div className="flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1.5 text-xs text-amber-300">
                    <AlertCircle className="h-3.5 w-3.5" />
                    {tooltip}
                  </div>
                )}
                {MODULES.map((mod) => {
                  const checked = manualModules.includes(mod.key);
                  return (
                    <label
                      key={mod.key}
                      className={`flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-2 text-sm transition-colors ${
                        checked ? "border-blue-500/40 bg-blue-500/10 text-zinc-100" : "border-zinc-800 bg-zinc-950/50 text-zinc-400 hover:bg-zinc-800"
                      } ${mod.required ? "opacity-90" : ""}`}
                    >
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-blue-500"
                        checked={checked}
                        disabled={mod.required || isStreaming}
                        onChange={() => toggleModule(mod.key)}
                      />
                      <span className="text-zinc-500">{mod.icon}</span>
                      <span className="flex-1">{mod.label}</span>
                      {mod.required && <Badge variant="outline" className="h-5 text-[10px] border-zinc-700 text-zinc-500">必须</Badge>}
                    </label>
                  );
                })}
              </CardContent>
            </Card>
          )}

          <Card className="border-zinc-800 bg-zinc-900">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold text-zinc-100">当前执行状态</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">模式</span>
                <Badge className="bg-zinc-800 text-zinc-200 hover:bg-zinc-800">{modeLabel}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">当前节点</span>
                <span className="text-zinc-200">{nodeLabel}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">Token 消耗</span>
                <span className="text-zinc-200">{tokenUsage ?? 0}</span>
              </div>
              {isStreaming && (
                <div className="flex items-center gap-2 text-xs text-blue-400">
                  <Activity className="h-3.5 w-3.5 animate-pulse" />
                  运行中...
                </div>
              )}
            </CardContent>
          </Card>

          {/* 当前使用的大模型供应商/模型 */}
          {modelStatus && (
            <Card className="border-zinc-800 bg-zinc-900">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                  <Server className="h-4 w-4 text-zinc-400" />
                  模型供应商
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-xs">
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">主 LLM</span>
                    <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-300">
                      {modelStatus.llm.provider}
                    </Badge>
                  </div>
                  <div className="truncate text-zinc-300" title={modelStatus.llm.model}>
                    {modelStatus.llm.model}
                  </div>
                </div>

                {modelStatus.backup_llm && (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500">备用 LLM</span>
                      <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-300">
                        {modelStatus.backup_llm.provider}
                      </Badge>
                    </div>
                    <div className="truncate text-zinc-300" title={modelStatus.backup_llm.model}>
                      {modelStatus.backup_llm.model}
                    </div>
                  </div>
                )}

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Embedding</span>
                    <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-300">
                      {modelStatus.embedding.provider}
                    </Badge>
                  </div>
                  <div className="truncate text-zinc-300" title={modelStatus.embedding.model}>
                    {modelStatus.embedding.model}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Rerank</span>
                    <div className="flex items-center gap-1">
                      <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-300">
                        {modelStatus.rerank.provider}
                      </Badge>
                      {modelStatus.rerank.candidates.length > 0 && (
                        <Badge variant="outline" className="text-[10px] border-blue-700/50 text-blue-300">
                          {modelStatus.rerank.selection_mode}
                        </Badge>
                      )}
                    </div>
                  </div>
                  {modelStatus.rerank.candidates.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {modelStatus.rerank.candidates.map((c) => (
                        <Badge
                          key={c.display_name}
                          variant="outline"
                          className="text-[10px] border-zinc-700 text-zinc-400"
                          title={c.model}
                        >
                          {c.display_name}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <div className="text-zinc-500">未启用模型 rerank</div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* 知识库状态卡片 */}
          {ragStatus && (
            <Card className="border-zinc-800 bg-zinc-900">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                  <Database className="h-4 w-4 text-zinc-400" />
                  知识库状态
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {/* 数据库列表 */}
                <div className="flex flex-wrap gap-1">
                  {ragStatus.databases.map((db) => (
                    <Badge
                      key={db.name}
                      className={`text-[10px] ${
                        db.type === "online_api"
                          ? "bg-emerald-900/40 text-emerald-300 border-emerald-700/50 hover:bg-emerald-900/60"
                          : db.type === "local_file"
                            ? "bg-blue-900/40 text-blue-300 border-blue-700/50 hover:bg-blue-900/60"
                            : "bg-zinc-800/60 text-zinc-400 border-zinc-600/50 hover:bg-zinc-700/60"
                      }`}
                      variant="outline"
                    >
                      {db.name}
                    </Badge>
                  ))}
                </div>
                {/* Collection 文档数 */}
                {ragStatus.collections && (
                  <div className="space-y-1 text-xs">
                    {Object.entries(ragStatus.collections).map(([key, count]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-zinc-500">{key}</span>
                        <span className="text-zinc-300">{count} 条</span>
                      </div>
                    ))}
                  </div>
                )}
                {/* 在线补充状态 */}
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">在线补充</span>
                  <span className={ragStatus.online_fallback_enabled ? "text-emerald-400" : "text-zinc-500"}>
                    {ragStatus.online_fallback_enabled ? "已启用" : "已关闭"}
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          {clarification && onClarificationSubmit && onStop && (
            <ClarificationDialog
              request={clarification}
              onSubmit={onClarificationSubmit}
              onStop={onStop}
              disabled={!isStreaming}
            />
          )}
        </div>
      </ScrollArea>
    </aside>
  );
}
