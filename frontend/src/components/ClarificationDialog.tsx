// 人工干预对话框组件
// 当 Supervisor 节点发起 clarification_needed 事件时弹出，让用户选择选项或填写自定义方案。
// 使用 framer-motion 做淡入淡出 + 缩放动画，TailwindCSS 深色主题。
import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Check, HelpCircle, X } from 'lucide-react';
import type { ClarificationOption } from '../types/sse';

// ---------------------------------------------------------------------------
// Props 定义
// ---------------------------------------------------------------------------

interface ClarificationDialogProps {
  /** clarification_needed.data.question */
  question: string;
  /** clarification_needed.data.options */
  options: ClarificationOption[];
  /** clarification_needed.data.context */
  context: string;
  /** 确认回调，传入选中选项 id 与可选自定义文本 */
  onRespond: (selectedOption: string, freeText?: string) => void;
  /** 取消回调 */
  onStop: () => void;
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

/** 把下划线 context 转成简短中文标签 */
function contextLabel(context: string): string {
  const map: Record<string, string> = {
    parameter_missing: '参数缺失',
    ambiguous_mechanism: '机制歧义',
    model_selection: '模型选择',
    data_conflict: '数据冲突',
    hypothesis_clarify: '假设澄清',
  };
  return map[context] ?? context;
}

/** 判断是否为自定义选项（id='C' 或 label 含"自定义"） */
function isCustomOption(opt: ClarificationOption): boolean {
  return opt.id === 'C' || opt.label.includes('自定义');
}

// ---------------------------------------------------------------------------
// 主组件
// ---------------------------------------------------------------------------

export default function ClarificationDialog({
  question,
  options,
  context,
  onRespond,
  onStop,
}: ClarificationDialogProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [freeText, setFreeText] = useState<string>('');

  // 当前选中的选项对象
  const selected = options.find((o) => o.id === selectedOption) ?? null;
  // 是否选中了自定义选项
  const showFreeText = selected !== null && isCustomOption(selected);

  const handleConfirm = () => {
    if (!selectedOption) return;
    // 自定义选项需要填写文本
    if (showFreeText && !freeText.trim()) return;
    onRespond(
      selectedOption,
      showFreeText ? freeText.trim() : undefined,
    );
  };

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          className="mx-4 w-full max-w-lg rounded-2xl border border-slate-700 bg-slate-800 p-6"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
        >
          {/* 顶部标题 */}
          <div className="flex items-center gap-2">
            <HelpCircle className="h-5 w-5 text-indigo-400" />
            <h2 className="text-lg font-semibold text-slate-100">
              需要您的输入
            </h2>
          </div>

          {/* context 徽章 */}
          <div className="mt-2">
            <span className="rounded bg-indigo-500/20 px-2 py-1 text-xs text-indigo-300">
              {contextLabel(context)}
            </span>
          </div>

          {/* 问题文本 */}
          <p className="mt-3 text-sm text-slate-300">{question}</p>

          {/* 选项列表 */}
          <div className="mt-4 space-y-2">
            {options.map((opt) => {
              const isSelected = selectedOption === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setSelectedOption(opt.id)}
                  className={`w-full rounded-lg px-4 py-3 text-left text-sm text-slate-100 transition-colors ${
                    isSelected
                      ? 'bg-slate-600 ring-2 ring-indigo-500'
                      : 'bg-slate-700 hover:bg-slate-600'
                  }`}
                >
                  <span className="mr-2 font-medium text-indigo-300">
                    {opt.id}.
                  </span>
                  {opt.label}
                </button>
              );
            })}
          </div>

          {/* 自定义方案输入框 */}
          <AnimatePresence>
            {showFreeText && (
              <motion.div
                className="mt-3"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <textarea
                  value={freeText}
                  onChange={(e) => setFreeText(e.target.value)}
                  placeholder="请填写自定义方案..."
                  className="min-h-[80px] w-full resize-y rounded-lg border border-slate-600 bg-slate-900 p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* 底部按钮栏 */}
          <div className="mt-5 flex items-center justify-between">
            <button
              type="button"
              onClick={onStop}
              className="flex items-center gap-1 text-sm text-slate-400 transition-colors hover:text-rose-400"
            >
              <X className="h-4 w-4" />
              取消
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              disabled={
                !selectedOption || (showFreeText && !freeText.trim())
              }
              className="flex items-center gap-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-600 disabled:text-slate-400"
            >
              <Check className="h-4 w-4" />
              确认
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
