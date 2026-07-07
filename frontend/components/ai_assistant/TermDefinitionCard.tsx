"use client";

import React from "react";
import { motion } from "framer-motion";
import { BookOpen, Network, Tag } from "lucide-react";
import { cn } from "@/lib/utils";

// 术语定义（与后端 TermDefinition.model_dump() 对应）
export interface TermDefinition {
  term: string;
  canonical_name: string;
  definition: string;
  synonyms: string[];
  category: string;
  related_pathway: string;
}

interface TermDefinitionCardProps {
  definitions: TermDefinition[];
  className?: string;
}

// 术语类别对应的颜色标签
const CATEGORY_COLORS: Record<string, string> = {
  细胞因子: "bg-purple-500/10 text-purple-300",
  受体: "bg-blue-500/10 text-blue-300",
  细胞类型: "bg-teal-500/10 text-teal-300",
  信号通路: "bg-amber-500/10 text-amber-300",
  生物过程: "bg-pink-500/10 text-pink-300",
  药物: "bg-red-500/10 text-red-300",
  疾病: "bg-orange-500/10 text-orange-300",
  其他: "bg-zinc-500/10 text-zinc-300",
};

/**
 * 术语定义卡片组件
 * 展示 MCP 工具获取的生物医学术语标准化定义，
 * 帮助用户理解专业术语背景，同时减少模型需要生成的上下文。
 */
export function TermDefinitionCard({
  definitions,
  className,
}: TermDefinitionCardProps) {
  if (!definitions || definitions.length === 0) return null;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center gap-1.5">
        <BookOpen className="h-3.5 w-3.5 text-indigo-400" />
        <span className="text-xs font-medium text-zinc-300">
          术语定义（MCP 标准化）
        </span>
      </div>
      <div className="grid gap-2">
        {definitions.map((def, idx) => (
          <TermCard key={idx} definition={def} index={idx} />
        ))}
      </div>
    </div>
  );
}

/**
 * 单个术语定义卡片
 */
function TermCard({
  definition,
  index,
}: {
  definition: TermDefinition;
  index: number;
}) {
  const categoryColor =
    CATEGORY_COLORS[definition.category] ?? CATEGORY_COLORS.其他;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      className="rounded-md border border-zinc-700/60 bg-zinc-800/50 p-2.5"
    >
      {/* 术语名与类别 */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-semibold text-zinc-100">
              {definition.term}
            </span>
            {definition.canonical_name &&
              definition.canonical_name.toLowerCase() !==
                definition.term.toLowerCase() && (
                <span className="text-[11px] italic text-zinc-500">
                  {definition.canonical_name}
                </span>
              )}
          </div>
        </div>
        {definition.category && (
          <span
            className={cn(
              "flex-shrink-0 rounded-full px-1.5 py-0.5 text-[9px]",
              categoryColor
            )}
          >
            {definition.category}
          </span>
        )}
      </div>

      {/* 定义文本 */}
      <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
        {definition.definition}
      </p>

      {/* 同义词标签 */}
      {definition.synonyms && definition.synonyms.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          <Tag className="h-2.5 w-2.5 text-zinc-600" />
          {definition.synonyms.slice(0, 4).map((syn, i) => (
            <span
              key={i}
              className="rounded bg-zinc-700/40 px-1.5 py-0.5 text-[9px] text-zinc-400"
            >
              {syn}
            </span>
          ))}
        </div>
      )}

      {/* 相关通路 */}
      {definition.related_pathway && (
        <div className="mt-1.5 flex items-center gap-1 text-[10px] text-amber-400/80">
          <Network className="h-2.5 w-2.5" />
          <span>通路：{definition.related_pathway}</span>
        </div>
      )}
    </motion.div>
  );
}
