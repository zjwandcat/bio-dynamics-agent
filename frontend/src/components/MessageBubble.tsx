// MessageBubble: 单条聊天消息气泡组件
// 根据角色（user / agent / system）渲染不同样式：
// - user: 右对齐，靛蓝气泡，User 图标位于气泡右上方
// - agent: 左对齐，深灰气泡，Sparkles 图标位于气泡左上方，支持 Markdown / 代码高亮 / 图片
// - system: 居中，斜体灰字，Info 图标 + 文字

import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { User, Sparkles, Info } from 'lucide-react';
import type { ChatMessage } from '../types/sse';

/** MessageBubble 组件 props */
export interface MessageBubbleProps {
  message: ChatMessage;
}

/**
 * 单条消息气泡：依据 message.role 渲染不同样式。
 * agent 消息支持 Markdown 渲染（含表格、列表、代码块），
 * 并在 message.code 存在时用 Prism 渲染 Python 代码，
 * 在 message.imageBase64 存在时渲染 base64 图片。
 */
export default function MessageBubble({ message }: MessageBubbleProps) {
  // system 消息：居中、斜体、灰字，前缀 Info 图标
  if (message.role === 'system') {
    return (
      <div className="flex justify-center">
        <div className="flex items-center gap-1.5 text-xs text-slate-500 italic">
          <Info className="w-3.5 h-3.5" />
          <span>{message.content}</span>
        </div>
      </div>
    );
  }

  // user 消息：右对齐，靛蓝气泡，User 图标在气泡右上方
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="flex flex-col items-end max-w-[80%]">
          <User className="w-4 h-4 text-slate-400 mb-1" />
          <div className="bg-indigo-600 text-white rounded-2xl rounded-br-sm px-4 py-2 whitespace-pre-wrap break-words">
            {message.content}
          </div>
        </div>
      </div>
    );
  }

  // agent 消息：左对齐，深灰气泡，Sparkles 图标在气泡左上方
  return (
    <div className="flex justify-start">
      <div className="flex flex-col items-start max-w-[80%]">
        <Sparkles className="w-4 h-4 text-indigo-400 mb-1" />
        <div className="bg-slate-800 text-slate-100 rounded-2xl rounded-bl-sm px-4 py-2">
          {/* Markdown 渲染：支持表格、列表、代码块；code 组件用 Prism 高亮 */}
          <div className="prose prose-invert prose-sm max-w-none break-words">
            <ReactMarkdown
              components={{
                // react-markdown v9 已不传 inline，这里以 any 兼容旧用法
                code({ inline, className, children, ...props }: any) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code
                      className="bg-slate-700 px-1.5 py-0.5 rounded text-sm"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>

          {/* 独立代码块：message.code 用 Prism 渲染（语言 python，深色主题 oneDark） */}
          {message.code && (
            <SyntaxHighlighter language="python" style={oneDark} PreTag="div">
              {message.code}
            </SyntaxHighlighter>
          )}

          {/* 图片：base64 直出 */}
          {message.imageBase64 && (
            <img
              src={`data:image/png;base64,${message.imageBase64}`}
              alt="仿真结果"
              className="rounded-lg max-w-full my-2"
            />
          )}
        </div>
      </div>
    </div>
  );
}
