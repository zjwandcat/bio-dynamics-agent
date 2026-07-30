import { motion } from 'framer-motion';

/** 打字指示器组件的 props */
interface LoadingIndicatorProps {
  /** 当前阶段文案，如 "正在沙箱中执行仿真..." */
  text?: string;
}

/**
 * 打字指示器组件：三点跳动动画 + 当前阶段文案。
 * 用于在 AI 思考或某个 Worker 节点执行期间提示用户。
 */
export function LoadingIndicator({ text = '正在思考...' }: LoadingIndicatorProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="w-2 h-2 bg-indigo-400 rounded-full"
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
          />
        ))}
      </div>
      <span className="text-sm text-slate-400">{text}</span>
    </div>
  );
}
