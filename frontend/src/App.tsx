// App.tsx: 应用主布局
// 整合 useChatStream hook、TopBar、StageTimeline、MessageList、ChatInput、
// ReportPanel、ClarificationDialog 等组件，构建三栏深色主题聊天界面。
//
// 布局结构：
// - 顶部：TopBar（LLM / Agent / 阶段 / 连接状态）
// - 主体：三栏
//   - 左：StageTimeline（执行阶段时间轴）
//   - 中：MessageList（消息流） + ChatInput（输入区）
//   - 右：ReportPanel（报告面板，7 个 tab）
// - 弹层：ClarificationDialog（人工干预对话框）

import { useChatStream } from './hooks/useChatStream';
import TopBar from './components/TopBar';
import StageTimeline from './components/StageTimeline';
import MessageList from './components/MessageList';
import ChatInput from './components/ChatInput';
import ReportPanel from './components/ReportPanel';
import ClarificationDialog from './components/ClarificationDialog';
import { NODE_STAGE_MAP } from './types/sse';

function App() {
  // 从 useChatStream hook 获取全部状态与控制方法
  const {
    messages,
    stages,
    currentLLM,
    currentAgent,
    currentStage,
    currentActionText,
    reports,
    clarification,
    connectionStatus,
    isStreaming,
    sendMessage,
    respondToClarification,
    stopStream,
    clearMemory,
  } = useChatStream();

  // 当前阶段的中文标签：从 NODE_STAGE_MAP 查询
  const currentStageLabel = currentStage
    ? NODE_STAGE_MAP[currentStage]
    : undefined;

  return (
    <div className="min-h-screen flex flex-col bg-slate-900 text-slate-100">
      {/* 顶部状态栏 */}
      <TopBar
        currentLLM={currentLLM}
        currentAgent={currentAgent}
        currentStage={currentStage}
        currentStageLabel={currentStageLabel}
        connectionStatus={connectionStatus}
      />

      {/* 主体三栏布局：左 StageTimeline / 中 MessageList+ChatInput / 右 ReportPanel */}
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧：阶段时间轴（窄屏可隐藏，这里保持显示） */}
        <StageTimeline
          stages={stages}
          currentStage={currentStage}
          currentActionText={currentActionText}
        />

        {/* 中间：消息流 + 输入区 */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <MessageList
            messages={messages}
            isStreaming={isStreaming}
            currentActionText={currentActionText}
          />
          <ChatInput
            onSend={sendMessage}
            disabled={isStreaming}
            onClear={clearMemory}
            placeholder="输入您的生物学问题或通路描述，例如：EGF 刺激后 pEGFR 与 ppERK 的时序响应..."
          />
        </div>

        {/* 右侧：报告面板 */}
        <ReportPanel
          pathwayGraph={reports.pathwayGraph}
          code={reports.code}
          imageBase64={reports.imageBase64}
          simulationResult={reports.simulationResult}
          validationReport={reports.validationReport}
          experimentProtocols={reports.experimentProtocols}
          markdown={reports.markdown}
          knowledgeGraph={reports.knowledgeGraph}
          ragInsights={reports.ragInsights}
          paperEvidence={reports.paperEvidence}
        />
      </div>

      {/* 人工干预对话框（收到 clarification_needed 时弹出） */}
      {clarification && (
        <ClarificationDialog
          question={clarification.question}
          options={clarification.options}
          context={clarification.context}
          onRespond={respondToClarification}
          onStop={stopStream}
        />
      )}
    </div>
  );
}

export default App;
