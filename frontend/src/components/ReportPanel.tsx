// 右侧报告面板组件
// 展示 v4 工作流的 7 类产物：机制图 / ODE 代码 / 仿真曲线 / 验证报告 /
// 实验方案 / 最终报告 / 文献证据。支持数据首次到达时自动切换到对应 tab。
import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Network,
  Code,
  LineChart,
  ShieldCheck,
  FlaskConical,
  FileText,
  BookOpen,
} from 'lucide-react';
import type { ComponentPropsWithoutRef } from 'react';
import type {
  V4PathwayGraphData,
  V4SimulationResultData,
  V4ValidationReportData,
  KnowledgeGraphData,
  RagInsightsData,
  PathwayNode,
} from '../types/sse';

// ---------------------------------------------------------------------------
// Props 定义
// ---------------------------------------------------------------------------

interface ReportPanelProps {
  /** v4_pathway_graph 事件数据 */
  pathwayGraph?: V4PathwayGraphData;
  /** code_generated 事件（Python 代码字符串） */
  code?: string;
  /** image_ready 事件 */
  imageBase64?: string;
  /** v4_simulation_result 事件数据 */
  simulationResult?: V4SimulationResultData;
  /** v4_validation_report 事件数据 */
  validationReport?: V4ValidationReportData;
  /** experiment_protocols 事件数据 */
  experimentProtocols?: unknown[];
  /** report.data.markdown */
  markdown?: string;
  /** knowledge_graph 事件数据 */
  knowledgeGraph?: KnowledgeGraphData;
  /** rag_insights 事件数据 */
  ragInsights?: RagInsightsData;
  /** paper_evidence 事件数据 */
  paperEvidence?: unknown[];
  /** 外部指定的激活 tab（可选） */
  activeTab?: string;
}

// ---------------------------------------------------------------------------
// Tab 定义
// ---------------------------------------------------------------------------

const TABS = [
  { id: 'mechanism', label: '机制图', icon: Network },
  { id: 'code', label: 'ODE 代码', icon: Code },
  { id: 'curve', label: '仿真曲线', icon: LineChart },
  { id: 'validation', label: '验证报告', icon: ShieldCheck },
  { id: 'experiment', label: '实验方案', icon: FlaskConical },
  { id: 'report', label: '最终报告', icon: FileText },
  { id: 'evidence', label: '文献证据', icon: BookOpen },
] as const;

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

/** 根据 relation 文本返回对应的展示颜色 */
function relationColor(relation: string): string {
  const r = relation.toLowerCase();
  if (r.includes('activ')) return '#10b981'; // activation 绿
  if (r.includes('inhibit')) return '#f43f5e'; // inhibition 红
  if (r.includes('phosphor')) return '#3b82f6'; // phosphorylation 蓝
  if (r.includes('bind')) return '#f97316'; // binding 橙
  if (r.includes('cataly')) return '#a855f7'; // catalysis 紫
  return '#64748b'; // 默认灰
}

/** 空状态提示块 */
function EmptyState({ text }: { text: string }) {
  return (
    <div className="flex h-full min-h-[200px] items-center justify-center text-sm text-slate-500">
      {text}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 机制图 SVG 渲染
// ---------------------------------------------------------------------------

interface NodePos {
  x: number;
  y: number;
}

/** 计算节点的垂直布局位置 */
function useNodePositions(nodes: PathwayNode[]): Map<string, NodePos> {
  return useMemo(() => {
    const map = new Map<string, NodePos>();
    nodes.forEach((n, i) => {
      map.set(n.id, { x: 150, y: 40 + i * 60 });
    });
    return map;
  }, [nodes]);
}

function MechanismGraph({ data }: { data: V4PathwayGraphData }) {
  const positions = useNodePositions(data.nodes);
  // 收集所有出现过的颜色，生成对应的箭头 marker
  const markers = useMemo(() => {
    const set = new Set<string>();
    data.edges.forEach((e) => set.add(relationColor(e.relation)));
    return Array.from(set);
  }, [data.edges]);

  const svgHeight = Math.max(200, data.nodes.length * 60 + 40);

  return (
    <div className="space-y-3">
      <div className="text-sm font-semibold text-indigo-300">
        通路类型：{data.pathway_class}
      </div>
      {data.nodes.length === 0 ? (
        <EmptyState text="无节点数据" />
      ) : (
        <svg
          width="100%"
          height={svgHeight}
          viewBox={`0 0 300 ${svgHeight}`}
          className="rounded-lg bg-slate-950/60"
        >
          {/* 每种颜色一个箭头标记 */}
          <defs>
            {markers.map((c) => (
              <marker
                key={c}
                id={`arrow-${c.replace('#', '')}`}
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill={c} />
              </marker>
            ))}
          </defs>

          {/* 边（曲线 + 箭头） */}
          {data.edges.map((e, i) => {
            const s = positions.get(e.source);
            const t = positions.get(e.target);
            if (!s || !t) return null;
            const color = relationColor(e.relation);
            const midY = (s.y + t.y) / 2;
            // 控制点向右偏移，形成右凸贝塞尔曲线
            const path = `M ${s.x} ${s.y} C ${s.x + 90} ${midY}, ${t.x + 90} ${midY}, ${t.x} ${t.y}`;
            return (
              <g key={`edge-${i}`}>
                <path
                  d={path}
                  stroke={color}
                  strokeWidth={2}
                  fill="none"
                  markerEnd={`url(#arrow-${color.replace('#', '')})`}
                />
                <text
                  x={s.x + 95}
                  y={midY}
                  fill={color}
                  fontSize="9"
                  textAnchor="middle"
                >
                  {e.relation}
                </text>
              </g>
            );
          })}

          {/* 节点（圆圈 + 标签） */}
          {data.nodes.map((n) => {
            const p = positions.get(n.id);
            if (!p) return null;
            return (
              <g key={`node-${n.id}`}>
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={18}
                  fill="#1e293b"
                  stroke="#6366f1"
                  strokeWidth={2}
                />
                <text
                  x={p.x}
                  y={p.y + 4}
                  fill="#e2e8f0"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {n.label.slice(0, 4)}
                </text>
                <text
                  x={p.x - 90}
                  y={p.y + 4}
                  fill="#94a3b8"
                  fontSize="10"
                  textAnchor="end"
                >
                  {n.label}
                </text>
              </g>
            );
          })}
        </svg>
      )}

      {/* 节点 / 边列表兜底展示 */}
      <details className="text-xs text-slate-400">
        <summary className="cursor-pointer text-slate-500">
          查看节点与边列表
        </summary>
        <div className="mt-2 space-y-1">
          <div className="text-slate-500">节点：</div>
          {data.nodes.map((n) => (
            <div key={n.id} className="pl-2">
              · {n.label}（{n.species}，{n.node_type}）
            </div>
          ))}
          <div className="mt-2 text-slate-500">边：</div>
          {data.edges.map((e, i) => (
            <div key={i} className="pl-2">
              · {e.source} →{' '}
              <span style={{ color: relationColor(e.relation) }}>
                {e.relation}
              </span>{' '}
              → {e.target}
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 仿真曲线 tab 内容
// ---------------------------------------------------------------------------

function CurveTab({
  imageBase64,
  simulationResult,
}: {
  imageBase64?: string;
  simulationResult?: V4SimulationResultData;
}) {
  const img = imageBase64 ?? simulationResult?.image_base64 ?? null;
  if (img) {
    return (
      <img
        src={`data:image/png;base64,${img}`}
        alt="仿真曲线"
        className="w-full rounded-lg"
      />
    );
  }
  // 没有图片但有仿真数据：用简单 SVG 折线图渲染第一个物种
  if (simulationResult && simulationResult.time_points.length > 0) {
    const speciesKeys = Object.keys(simulationResult.species);
    if (speciesKeys.length === 0) {
      return <EmptyState text="等待仿真执行..." />;
    }
    const firstKey = speciesKeys[0];
    const values = simulationResult.species[firstKey];
    const tp = simulationResult.time_points;
    const width = 320;
    const height = 180;
    const pad = 24;
    const maxY = Math.max(...values, 1);
    const maxX = Math.max(...tp, 1);
    const pts = values
      .map((v, i) => {
        const x = pad + (tp[i] / maxX) * (width - pad * 2);
        const y = height - pad - (v / maxY) * (height - pad * 2);
        return `${x},${y}`;
      })
      .join(' ');
    return (
      <div className="space-y-2">
        <div className="text-xs text-slate-400">
          物种 {firstKey} 浓度随时间变化
        </div>
        <svg
          width="100%"
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          className="rounded-lg bg-slate-950/60"
        >
          <polyline
            points={pts}
            fill="none"
            stroke="#6366f1"
            strokeWidth={2}
          />
        </svg>
        <details className="text-xs text-slate-400">
          <summary className="cursor-pointer text-slate-500">
            查看数据表格
          </summary>
          <table className="mt-2 w-full text-left">
            <thead>
              <tr className="text-slate-500">
                <th className="pr-2">时间</th>
                <th>{firstKey}</th>
              </tr>
            </thead>
            <tbody>
              {tp.map((t, i) => (
                <tr key={i}>
                  <td className="pr-2">{t}</td>
                  <td>{values[i]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </div>
    );
  }
  return <EmptyState text="等待仿真执行..." />;
}

// ---------------------------------------------------------------------------
// 验证报告 tab 内容
// ---------------------------------------------------------------------------

const LEVEL_KEYS = ['level1', 'level2', 'level3', 'level4', 'level5'] as const;

function ValidationTab({ report }: { report: V4ValidationReportData }) {
  const confidence =
    typeof report.confidence === 'number'
      ? Math.round(report.confidence * 100)
      : null;
  return (
    <div className="space-y-3 text-sm">
      {/* 总体通过状态 */}
      <div className="flex items-center gap-2">
        <span className="text-slate-400">总体结果：</span>
        {report.overall_pass ? (
          <span className="text-emerald-400">✅ 通过</span>
        ) : (
          <span className="text-rose-400">❌ 未通过</span>
        )}
      </div>
      {/* 置信度 */}
      {confidence !== null && (
        <div className="flex items-center gap-2">
          <span className="text-slate-400">置信度：</span>
          <div className="h-2 w-32 overflow-hidden rounded-full bg-slate-700">
            <div
              className="h-full bg-indigo-500"
              style={{ width: `${confidence}%` }}
            />
          </div>
          <span className="text-slate-300">{confidence}%</span>
        </div>
      )}
      {/* 失败等级 */}
      {report.failed_levels && report.failed_levels.length > 0 && (
        <div>
          <span className="text-slate-400">失败等级：</span>
          <span className="text-rose-400">
            {report.failed_levels.join('、')}
          </span>
        </div>
      )}
      {/* 各等级明细 */}
      {LEVEL_KEYS.some((k) => report[k] != null) && (
        <div className="space-y-1 rounded-lg bg-slate-950/40 p-3">
          <div className="text-xs text-slate-500">五级验证明细</div>
          {LEVEL_KEYS.map((key) => {
            const lv = report[key];
            if (lv == null) return null;
            const obj = lv as { key?: string; name?: string; passed?: boolean };
            const name = obj.key ?? obj.name ?? key;
            const passed = obj.passed;
            return (
              <div
                key={key}
                className="flex items-center justify-between text-xs"
              >
                <span className="text-slate-300">{name}</span>
                <span>
                  {passed === undefined
                    ? '—'
                    : passed
                    ? '✅'
                    : '❌'}
                </span>
              </div>
            );
          })}
        </div>
      )}
      {/* Markdown 报告 */}
      {report.report_markdown && (
        <div className="prose prose-invert max-w-none rounded-lg bg-slate-950/40 p-3 text-xs">
          <ReactMarkdown>{report.report_markdown}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Markdown 渲染（带代码高亮）
// ---------------------------------------------------------------------------

function MarkdownBlock({ source }: { source: string }) {
  return (
    <div className="prose prose-invert max-w-none text-sm">
      <ReactMarkdown
        components={{
          code: ({
            className,
            children,
          }: ComponentPropsWithoutRef<'code'>) => {
            const match = /language-(\w+)/.exec(className || '');
            return match ? (
              <SyntaxHighlighter
                language={match[1]}
                style={oneDark}
                PreTag="div"
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className}>{children}</code>
            );
          },
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 主组件
// ---------------------------------------------------------------------------

export default function ReportPanel(props: ReportPanelProps) {
  const {
    pathwayGraph,
    code,
    imageBase64,
    simulationResult,
    validationReport,
    experimentProtocols,
    markdown,
    knowledgeGraph,
    ragInsights,
    paperEvidence,
    activeTab: externalTab,
  } = props;

  const [activeTab, setActiveTab] = useState<string>(externalTab ?? 'mechanism');

  // 外部指定 tab 时同步内部状态
  useEffect(() => {
    if (externalTab) setActiveTab(externalTab);
  }, [externalTab]);

  // 记录上次数据是否到位，用于首次到达时自动切换
  const prevRef = useRef({
    imageBase64: false,
    code: false,
    markdown: false,
    validationReport: false,
  });

  useEffect(() => {
    const prev = prevRef.current;
    const next = {
      imageBase64: !!imageBase64,
      code: !!code,
      markdown: !!markdown,
      validationReport: !!validationReport,
    };
    // 按优先级依次判断首次到达
    if (!prev.imageBase64 && next.imageBase64) setActiveTab('curve');
    else if (!prev.code && next.code) setActiveTab('code');
    else if (!prev.markdown && next.markdown) setActiveTab('report');
    else if (!prev.validationReport && next.validationReport)
      setActiveTab('validation');
    prevRef.current = next;
  }, [imageBase64, code, markdown, validationReport]);

  // 实验方案：优先用独立事件，回退到验证报告内嵌
  const experimentList =
    experimentProtocols ?? validationReport?.experiment_protocols ?? [];

  // 文献证据：优先用独立事件，回退到验证报告内嵌
  const evidenceList =
    paperEvidence ?? validationReport?.paper_evidence ?? [];

  return (
    <div className="flex w-96 flex-col border-l border-slate-800 bg-slate-900">
      {/* Tab 栏 */}
      <div className="flex overflow-x-auto border-b border-slate-800">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex shrink-0 items-center gap-1 px-3 py-3 text-xs transition-colors ${
                isActive
                  ? 'border-b-2 border-indigo-400 text-indigo-400'
                  : 'border-b-2 border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* 内容区 */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'mechanism' &&
          (pathwayGraph ? (
            <MechanismGraph data={pathwayGraph} />
          ) : (
            <EmptyState text="等待机制解析..." />
          ))}

        {activeTab === 'code' &&
          (code ? (
            <SyntaxHighlighter
              language="python"
              style={oneDark}
              showLineNumbers
              customStyle={{
                margin: 0,
                borderRadius: '0.5rem',
                fontSize: '0.75rem',
              }}
            >
              {code}
            </SyntaxHighlighter>
          ) : (
            <EmptyState text="等待 ODE 代码生成..." />
          ))}

        {activeTab === 'curve' && (
          <CurveTab
            imageBase64={imageBase64}
            simulationResult={simulationResult}
          />
        )}

        {activeTab === 'validation' &&
          (validationReport ? (
            <ValidationTab report={validationReport} />
          ) : (
            <EmptyState text="等待验证完成..." />
          ))}

        {activeTab === 'experiment' &&
          (experimentList.length > 0 ? (
            <div className="space-y-2">
              {experimentList.map((item, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-slate-950/40 p-3 text-xs text-slate-300"
                >
                  {typeof item === 'string' ? (
                    <ReactMarkdown>{item}</ReactMarkdown>
                  ) : (
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(item, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <EmptyState text="等待实验方案生成..." />
          ))}

        {activeTab === 'report' &&
          (markdown ? (
            <MarkdownBlock source={markdown} />
          ) : (
            <EmptyState text="等待报告生成..." />
          ))}

        {activeTab === 'evidence' &&
          (evidenceList.length === 0 && !ragInsights ? (
            <EmptyState text="等待文献检索..." />
          ) : (
            <div className="space-y-3">
              {/* RAG 洞察摘要 */}
              {ragInsights && (
                <div className="rounded-lg bg-slate-950/40 p-3 text-xs">
                  <div className="mb-2 font-semibold text-indigo-300">
                    RAG 洞察
                  </div>
                  <div className="space-y-1 text-slate-300">
                    <div>
                      <span className="text-slate-500">改写查询：</span>
                      {ragInsights.rewritten_query}
                    </div>
                    <div>
                      <span className="text-slate-500">命中率：</span>
                      {ragInsights.hit_rate}
                    </div>
                    <div>
                      <span className="text-slate-500">候选总数：</span>
                      {ragInsights.total_candidates}
                    </div>
                    {Object.keys(ragInsights.source_distribution).length > 0 && (
                      <div>
                        <span className="text-slate-500">来源分布：</span>
                        {Object.entries(ragInsights.source_distribution).map(
                          ([k, v]) => `${k}:${v}`,
                        ).join('，')}
                      </div>
                    )}
                  </div>
                </div>
              )}
              {/* 文献证据列表 */}
              {evidenceList.map((item, i) => {
                const obj = item as {
                  title?: string;
                  pmid?: string | number;
                  summary?: string;
                };
                const hasField =
                  obj.title || obj.pmid || obj.summary;
                return (
                  <div
                    key={i}
                    className="rounded-lg bg-slate-950/40 p-3 text-xs text-slate-300"
                  >
                    {hasField ? (
                      <div className="space-y-1">
                        {obj.title && (
                          <div className="font-medium text-slate-100">
                            {obj.title}
                          </div>
                        )}
                        {obj.pmid != null && (
                          <div className="text-slate-500">PMID: {obj.pmid}</div>
                        )}
                        {obj.summary && (
                          <div className="text-slate-400">{obj.summary}</div>
                        )}
                      </div>
                    ) : (
                      <pre className="whitespace-pre-wrap break-words">
                        {JSON.stringify(item, null, 2)}
                      </pre>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
      </div>

      {/* 知识图谱摘要（底部状态条，可选展示） */}
      {knowledgeGraph && (
        <div className="border-t border-slate-800 px-4 py-2 text-xs text-slate-500">
          知识图谱：{knowledgeGraph.node_count} 节点 /{' '}
          {knowledgeGraph.edge_count} 边 ·{' '}
          {knowledgeGraph.is_acyclic ? '无环' : '有环'}
        </div>
      )}
    </div>
  );
}
