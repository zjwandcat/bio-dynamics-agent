// ChatInput: 底部输入区组件
// textarea + 发送按钮 + 清除会话按钮，支持 Enter 发送、Shift+Enter 换行。

import { useRef, useState } from 'react';
import { Trash2, Send, Loader2 } from 'lucide-react';

/** ChatInput 组件 props */
export interface ChatInputProps {
  onSend: (text: string) => void;
  /** isStreaming 时禁用输入与发送 */
  disabled: boolean;
  /** 清除会话回调 */
  onClear?: () => void;
  placeholder?: string;
}

/**
 * 底部输入区：
 * - 上方一行：右侧"清除会话"按钮（Trash2 图标）
 * - 下方一行：textarea + 发送按钮（Send 图标，禁用时显示 Loader2 旋转）
 * - Enter 发送，Shift+Enter 换行
 */
export default function ChatInput({
  onSend,
  disabled,
  onClear,
  placeholder,
}: ChatInputProps) {
  // 输入文本状态
  const [input, setInput] = useState<string>('');
  // textarea 引用
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 发送逻辑：trim 后非空且未禁用则回调，并清空输入
  const handleSend = () => {
    const text = input.trim();
    if (!text || disabled) return;
    onSend(text);
    setInput('');
  };

  // 键盘事件：Enter 发送，Shift+Enter 换行
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-slate-800 bg-slate-900 p-4">
      {/* 上方一行：右侧清除会话按钮 */}
      <div className="flex justify-end mb-2">
        <button
          type="button"
          onClick={onClear}
          className="flex items-center gap-1 text-slate-400 hover:text-rose-400 text-sm transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          <span>清除会话</span>
        </button>
      </div>

      {/* 下方一行：textarea + 发送按钮 */}
      <div className="flex gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? 'Agent 处理中...' : placeholder}
          disabled={disabled}
          rows={2}
          className="flex-1 bg-slate-800 text-slate-100 rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-slate-500 disabled:opacity-60"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg px-5 py-3 flex items-center gap-2 transition-colors"
        >
          {disabled ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
          <span>发送</span>
        </button>
      </div>
    </div>
  );
}
