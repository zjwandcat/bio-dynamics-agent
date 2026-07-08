"use client";

import React from "react";
import Link from "next/link";
import {
  Atom,
  Play,
  RefreshCw,
  PanelRightOpen,
  PanelRightClose,
  Database,
  Languages,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useWorkbenchStore } from "@/lib/store";
import { useTranslation } from "@/lib/i18n";
import type { PathwayClass } from "@/lib/api";

const PATHWAY_OPTIONS: { value: PathwayClass; label: string }[] = [
  { value: "egfr", label: "EGFR" },
  { value: "mapk", label: "MAPK" },
  { value: "pi3k_akt_mtor", label: "PI3K / AKT / mTOR" },
  { value: "jak_stat", label: "JAK-STAT" },
  { value: "nf_kappa_b", label: "NF-κB" },
  { value: "wnt", label: "Wnt / β-catenin" },
  { value: "tgf_beta", label: "TGF-β" },
  { value: "p53", label: "p53" },
  { value: "apoptosis", label: "Apoptosis" },
  { value: "cell_cycle", label: "Cell Cycle" },
];

/**
 * Top header bar of the Scientific Modeling IDE: brand, pathway selector,
 * run action, knowledge-base controls, and the AI Assistant collapse toggle.
 */
export function WorkbenchHeader() {
  const { t, locale, toggleLocale } = useTranslation();
  const currentPathway = useWorkbenchStore((s) => s.currentPathway);
  const setCurrentPathway = useWorkbenchStore((s) => s.setCurrentPathway);
  const aiOpen = useWorkbenchStore((s) => s.uiState.aiAssistantOpen);
  const toggleAIPanel = useWorkbenchStore((s) => s.toggleAIPanel);
  const setAIPanelOpen = useWorkbenchStore((s) => s.setAIPanelOpen);
  const handleUpdateVectorDb = useWorkbenchStore((s) => s.handleUpdateVectorDb);
  const isUpdatingDb = useWorkbenchStore((s) => s.isUpdatingDb);
  const ragStatus = useWorkbenchStore((s) => s.ragStatus);

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-zinc-800 bg-zinc-950 px-3">
      {/* Brand */}
      <div className="flex items-center gap-2">
        <Link href="/workspace" className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-blue-500 to-purple-600">
            <Atom className="h-4 w-4 text-white" />
          </span>
          <span className="text-sm font-semibold text-zinc-100">{t("app.name")}</span>
          <Badge
            variant="outline"
            className="border-blue-700/50 text-[10px] text-blue-300"
          >
            v4
          </Badge>
        </Link>
      </div>

      {/* Center: pathway selector + run */}
      <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-zinc-500">{t("header.pathway")}</span>
            <select
              value={currentPathway ?? ""}
              onChange={(e) =>
                setCurrentPathway(
                  (e.target.value || null) as PathwayClass | null
                )
              }
              className="h-7 rounded-md border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-100 outline-none focus:border-blue-500"
            >
              <option value="">{t("header.pathway.placeholder")}</option>
              {PATHWAY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <Button
            size="sm"
            onClick={() => setAIPanelOpen(true)}
            title={t("header.run")}
            className="gap-1 bg-blue-600 hover:bg-blue-700"
          >
            <Play className="h-3.5 w-3.5" />
            {t("header.run")}
          </Button>
        </div>

      {/* Right: KB status + update + AI toggle */}
      <div className="flex items-center gap-2">
        {ragStatus && ragStatus.databases.length > 0 && (
          <div className="hidden items-center gap-1 lg:flex">
            {ragStatus.databases.slice(0, 4).map((db) => (
              <span
                key={db.name}
                title={db.type === "online_api" ? "在线 API" : db.type === "local_file" ? "本地文件" : "用户导入"}
                className={
                  "inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium " +
                  (db.type === "online_api"
                    ? "bg-emerald-900/40 text-emerald-300 border border-emerald-700/50"
                    : db.type === "local_file"
                      ? "bg-blue-900/40 text-blue-300 border border-blue-700/50"
                      : "bg-zinc-800/60 text-zinc-400 border border-zinc-700/50")
                }
              >
                {db.name}
              </span>
            ))}
          </div>
        )}
        <Button
          variant="outline"
          size="icon-sm"
          onClick={handleUpdateVectorDb}
          disabled={isUpdatingDb}
          title={t("header.updateKb")}
          aria-label={t("header.updateKb")}
          className="border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
        >
          <RefreshCw className={isUpdatingDb ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          title={t("header.kbStatus")}
          aria-label={t("header.kbStatus")}
          className="border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
        >
          <Database className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={toggleLocale}
          title={t("header.lang")}
          aria-label={t("header.lang")}
          className="gap-1 border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-300 hover:bg-zinc-800"
        >
          <Languages className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">{locale === "zh" ? "中" : "EN"}</span>
        </Button>
        <Button
          variant={aiOpen ? "default" : "outline"}
          size="sm"
          onClick={toggleAIPanel}
          title={aiOpen ? t("header.ai.toggle.close") : t("header.ai.toggle.open")}
          aria-label={aiOpen ? t("header.ai.toggle.close") : t("header.ai.toggle.open")}
          className="gap-1"
        >
          {aiOpen ? (
            <PanelRightClose className="h-3.5 w-3.5" />
          ) : (
            <PanelRightOpen className="h-3.5 w-3.5" />
          )}
          <span className="hidden sm:inline">{t("header.ai")}</span>
        </Button>
      </div>
    </header>
  );
}
