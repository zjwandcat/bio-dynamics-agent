// MessageList: 消息流容器组件
// 渲染消息列表、自动滚动到底部、流式时的加载指示器、空状态提示。

import { useEffect, useRef } from 'react';
import { MessageSquare } from 'lucide-react';
import type { ChatMessage } from '../types/sse';
import { LoadingIndicator } from './LoadingIndicator';
import MessageBubble from './MessageBubble';

/** MessageList 组件 props */
export interface MessageListProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  currentActionText: string;
}

/**
 * 消息流容器：
 * - 自动滚动到底部（messages / isStreaming 变化时触发）
 * - 流式进行中显示 LoadingIndicator
 * - 消息为空且非流式时显示空状态提示
 */
export default function MessageList({
  messages,
  isStreaming,
  currentActionText,
}: MessageListProps) {
  // 容器 ref，用于触发自动滚动
  const scrollRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部：messages 变化或 isStreaming 切换时触发
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, isStreaming]);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto px-4 py-6 space-y-4 bg-slate-900"
    >
      {/* 空状态：无消息且非流式时提示用户开始对话 */}
      {messages.length === 0 && !isStreaming && (
        <div className="flex flex-col items-center justify-center h-full text-slate-500">
          <MessageSquare className="w-12 h-12 mb-3" />
          <p className="text-sm">开始对话：输入生物学假说或通路描述</p>
        </div>
      )}

      {/* 消息列表 */}
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {/* 流式加载指示器：底部显示当前阶段文案 */}
      {isStreaming && <LoadingIndicator text={currentActionText} />}
    </div>
  );
}
