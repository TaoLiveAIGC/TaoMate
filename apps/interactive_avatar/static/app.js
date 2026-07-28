const form = document.getElementById("jobForm");
const sceneDescription = document.getElementById("sceneDescription");
const userText = document.getElementById("userText");
let player = document.getElementById("player");
let standbyPlayer = document.getElementById("preloadPlayer");
const primaryPlayer = player;
const secondaryPlayer = standbyPlayer;
const liveCanvas = document.getElementById("liveCanvas");
const liveCanvasCtx = liveCanvas?.getContext("2d", { alpha: false });
const emptyStage = document.getElementById("emptyStage");
const videoFrame = document.querySelector(".video-frame");
const playbackRail = document.getElementById("playbackRail");
const liveControls = document.getElementById("liveControls");
const livePlayPauseBtn = document.getElementById("livePlayPauseBtn");
const liveProgress = document.getElementById("liveProgress");
const liveClock = document.getElementById("liveClock");
const liveReplayBtn = document.getElementById("liveReplayBtn");
const liveReturnBtn = document.getElementById("liveReturnBtn");
const liveOverlay = document.getElementById("liveOverlay");
const liveOverlayText = document.getElementById("liveOverlayText");
const replyBox = document.getElementById("replyBox");
const segmentList = document.getElementById("segmentList");
const logBox = document.getElementById("logBox");
const previewBtn = document.getElementById("previewBtn");
const cancelBtn = document.getElementById("cancelBtn");
const templateGrid = document.getElementById("templateGrid");
const clearTemplateBtn = document.getElementById("clearTemplateBtn");
const resetSessionBtn = document.getElementById("resetSessionBtn");
const serviceStatus = document.getElementById("serviceStatus");
const stageTitle = document.getElementById("stageTitle");
const modeMetric = document.getElementById("modeMetric");
const chunkMetric = document.getElementById("chunkMetric");
const phaseMetric = document.getElementById("phaseMetric");
const gpuMetric = document.getElementById("gpuMetric");
const systemNotice = document.getElementById("systemNotice");
const demoPlayer = document.getElementById("demoPlayer");
const demoTitle = document.getElementById("demoTitle");
const demoCaption = document.getElementById("demoCaption");
const demoFps = document.getElementById("demoFps");
const demoMode = document.getElementById("demoMode");
const demoClock = document.getElementById("demoClock");
const demoDots = document.getElementById("demoDots");
const demoPlaceholder = document.getElementById("demoPlaceholder");
const demoLiveBtn = document.getElementById("demoLiveBtn");
const demoDrawer = document.getElementById("demoDrawer");
const historyList = document.getElementById("historyList");
const historyCount = document.getElementById("historyCount");
const languageButtons = Array.from(document.querySelectorAll("[data-ui-language]"));

const UI_LANGUAGE_STORAGE_KEY = "taomate_ui_language";
const I18N = {
  en: {
    "app.subtitle": "Real-time Digital Human",
    "language.label": "Interface language",
    "header.newSession": "New session",
    "controls.input": "Input controls",
    "role.label": "Role",
    "role.clear": "Clear",
    "settings.label": "Generation settings",
    "mode.label": "Mode",
    "mode.text": "Text",
    "mode.firstFrame": "First frame",
    "mode.firstFrameSoon": "First-frame input is coming soon",
    "aspect.label": "Aspect ratio",
    "aspect.landscape": "Landscape",
    "aspect.portrait": "Portrait",
    "scene.summary": "Scene description",
    "scene.label": "Character and scene",
    "scene.placeholder": "Describe the person, lighting, and background",
    "conversation.label": "Conversation",
    "conversation.placeholder": "Enter what you want to say",
    "actions.preview": "Preview prompt",
    "actions.start": "Start conversation",
    "actions.cancel": "Cancel",
    "stage.video": "Video playback",
    "stage.liveVideo": "Live video",
    "stage.waitingToStart": "Waiting to start",
    "stage.liveConversation": "Live conversation",
    "playback.progress": "Playback progress",
    "playback.play": "Play",
    "playback.pause": "Pause",
    "playback.replay": "Replay",
    "playback.returnLive": "Return to live",
    "demo.reference": "Reference demos",
    "demo.empty": "No demos yet",
    "demo.loading": "Loading demo",
    "demo.waitingTitle": "Waiting for demos",
    "demo.waitingLine": "Waiting for demo dialogue",
    "demo.preparingLine": "Demo dialogue is loading",
    "demo.emptyCaption": "Completed demos will appear here.",
    "demo.previewMain": "Preview on main stage",
    "demo.livePreview": "Demo live preview",
    "demo.view": "View",
    "demo.collapse": "Collapse",
    "inspector.label": "History and debug",
    "history.label": "History",
    "history.count": "{count} turns",
    "history.none": "No history yet",
    "history.untitled": "Untitled session",
    "history.realtime": "Live conversation",
    "history.archived": "Previous session",
    "history.currentPrefix": "Current",
    "history.meta": "{time} · {count} turns · {seconds}s",
    "history.review": "Review",
    "history.current": "Current",
    "history.continue": "Continue",
    "history.turn": "Turn {number}",
    "history.saved": "Saved",
    "debug.label": "Debug details",
    "debug.expand": "Expand",
    "debug.reply": "Model reply",
    "debug.segments": "Segments",
    "debug.log": "Runtime log",
    "debug.viewPrompt": "View generated prompt",
    "segment.title": "Segment {number}",
    "segment.waiting": "Waiting",
    "reply.waiting": "Waiting for reply",
    "reply.generating": "The dialogue model is preparing a short reply.",
    "status.ready": "READY",
    "status.onAir": "ON AIR",
    "status.starting": "STARTING",
    "status.busy": "BUSY",
    "status.error": "ERROR",
    "status.offline": "OFFLINE",
    "phase.idle": "Waiting for input",
    "phase.preview_ready": "Storyboard ready",
    "phase.preview_failed": "Preview failed",
    "phase.llm_replying": "Thinking",
    "phase.prompt_expanding": "Preparing response",
    "phase.queued_on_gpu": "Connecting compute",
    "phase.queued": "Connecting compute",
    "phase.accepted": "Connecting",
    "phase.worker_warming_model": "Warming up",
    "phase.worker_running": "Generating first frame",
    "phase.running": "Generating first frame",
    "phase.generating": "Generating first frame",
    "phase.streaming": "Live conversation",
    "phase.succeeded": "Conversation complete",
    "phase.failed": "Generation failed",
    "phase.canceled": "Canceled",
    "phase.expired": "Session expired",
    "phase.new_conversation": "New session ready",
    "phase.uploading": "Connecting",
    "phase.previewing_prompt": "Preparing storyboard",
    "phase.input_required": "First frame required",
    "live.firstFrameReady": "First frame ready · Connecting live stream",
    "live.firstFrameConnecting": "Connecting first frame",
    "live.newScene": "Preparing new scene · Waiting for live stream",
    "live.waitingConnect": "Waiting to connect",
    "live.idleBridge": "Live idle view · Connecting generation stream",
    "live.idle": "Live idle",
    "live.connecting": "Connecting generation stream",
    "live.continuationFrame": "Holding previous frame · Connecting new live stream",
    "live.continuation": "Connecting new live stream",
    "live.continuing": "Continuing",
    "live.waiting": "Waiting for live stream",
    "live.reviewFull": "Reviewing full video",
    "live.reviewBuffer": "Reviewing live buffer",
    "live.ready": "Live video ready · This turn {current}s · Total {total}s",
    "live.buffering": "Buffering live video · This turn {current}s · Total {total}s",
    "live.waitingFrame": "Live · Waiting for frames · Total {total}s",
    "live.preloaded": "Live · Total {total}s · Preloaded {buffered}s",
    "live.waitingMore": "Live · Total {total}s · Waiting for frames",
    "live.clickStart": "Click the video to start · Total {total}s",
    "live.clickSound": "Click the video to enable sound · Total {total}s",
    "live.ended": "Live ended · About {total}s",
    "live.finalReady": "Full video ready",
    "live.review": "Review",
    "live.complete": "Complete {total}s",
    "live.added": "Live +{total}s",
    "error.generationFailed": "Generation did not succeed. Check the scene and conversation, then retry. Details are in the debug log.",
    "error.interruptedRetry": "Generation stopped. The current frame is preserved, so you can retry this turn.",
    "error.interruptedRestart": "Generation stopped. Please retry, or choose another role and start again.",
    "error.submitFailed": "Submission failed. Check the scene and conversation, then retry.",
    "gpu.unknown": "Compute unknown",
    "gpu.unavailable": "GPU status unavailable: {error}",
    "gpu.online": "Compute online",
    "gpu.ready": "Compute ready",
    "template.home.title": "Home Explainer",
    "template.home.role": "Young man · Contemporary home office",
    "template.tech.title": "Technology Presenter",
    "template.tech.role": "Young man · Professional technology studio",
    "template.business.title": "Business Consultant",
    "template.business.role": "Mature woman · Executive meeting room",
    "template.education.title": "Course Instructor",
    "template.education.role": "Young woman · Bright classroom corner",
    "template.wellness.title": "Wellness Host",
    "template.wellness.role": "Older man · Contemporary wellness studio",
  },
  zh: {
    "app.subtitle": "实时数字人",
    "language.label": "界面语言",
    "header.newSession": "新会话",
    "controls.input": "输入控制",
    "role.label": "角色",
    "role.clear": "清除",
    "settings.label": "生成设置",
    "mode.label": "模式",
    "mode.text": "文本",
    "mode.firstFrame": "首帧",
    "mode.firstFrameSoon": "首帧功能正在路上",
    "aspect.label": "画幅",
    "aspect.landscape": "横屏",
    "aspect.portrait": "竖屏",
    "scene.summary": "场景描述",
    "scene.label": "角色与场景",
    "scene.placeholder": "描述人物、光线和背景",
    "conversation.label": "对话内容",
    "conversation.placeholder": "输入你想说的话",
    "actions.preview": "预览提示词",
    "actions.start": "开始通话",
    "actions.cancel": "取消",
    "stage.video": "视频播放",
    "stage.liveVideo": "直播画面",
    "stage.waitingToStart": "等待开始",
    "stage.liveConversation": "实时通话中",
    "playback.progress": "播放进度",
    "playback.play": "播放",
    "playback.pause": "暂停",
    "playback.replay": "重播",
    "playback.returnLive": "返回直播",
    "demo.reference": "参考样例",
    "demo.empty": "暂无样例",
    "demo.loading": "样例加载中",
    "demo.waitingTitle": "等待样例",
    "demo.waitingLine": "等待样例台词",
    "demo.preparingLine": "样例台词准备中",
    "demo.emptyCaption": "样例生成完成后会显示在这里。",
    "demo.previewMain": "在主画面试播",
    "demo.livePreview": "Demo 直播试播",
    "demo.view": "查看",
    "demo.collapse": "收起",
    "inspector.label": "历史和调试",
    "history.label": "历史记录",
    "history.count": "{count} 轮",
    "history.none": "暂无历史",
    "history.untitled": "未命名会话",
    "history.realtime": "实时对话",
    "history.archived": "历史会话",
    "history.currentPrefix": "当前",
    "history.meta": "{time} · {count} 轮 · {seconds}s",
    "history.review": "回看",
    "history.current": "当前",
    "history.continue": "继续",
    "history.turn": "第 {number} 轮",
    "history.saved": "已保存",
    "debug.label": "调试详情",
    "debug.expand": "展开",
    "debug.reply": "模型回复",
    "debug.segments": "分段",
    "debug.log": "运行日志",
    "debug.viewPrompt": "查看生成提示词",
    "segment.title": "分段 {number}",
    "segment.waiting": "等待",
    "reply.waiting": "等待回复",
    "reply.generating": "后台 LLM 正在生成短回复。",
    "status.ready": "就绪",
    "status.onAir": "通话中",
    "status.starting": "启动中",
    "status.busy": "繁忙",
    "status.error": "错误",
    "status.offline": "离线",
    "phase.idle": "等待输入",
    "phase.preview_ready": "分镜已准备",
    "phase.preview_failed": "预览失败",
    "phase.llm_replying": "正在思考",
    "phase.prompt_expanding": "整理表达",
    "phase.queued_on_gpu": "接入算力",
    "phase.queued": "接入算力",
    "phase.accepted": "开始连接",
    "phase.worker_warming_model": "预热画面",
    "phase.worker_running": "生成首帧",
    "phase.running": "生成首帧",
    "phase.generating": "生成首帧",
    "phase.streaming": "实时通话中",
    "phase.succeeded": "通话完成",
    "phase.failed": "生成失败",
    "phase.canceled": "已取消",
    "phase.expired": "会话已过期",
    "phase.new_conversation": "新会话已准备",
    "phase.uploading": "正在连接",
    "phase.previewing_prompt": "准备分镜",
    "phase.input_required": "需要首帧",
    "live.firstFrameReady": "首帧已就绪 · 正在接入直播流",
    "live.firstFrameConnecting": "首帧接入中",
    "live.newScene": "新场景准备中 · 等待直播流接入",
    "live.waitingConnect": "等待接入",
    "live.idleBridge": "实时待机画面 · 正在接入生成流",
    "live.idle": "实时待机",
    "live.connecting": "正在接入生成流",
    "live.continuationFrame": "保持上一帧 · 正在接入新的直播流",
    "live.continuation": "正在接入新的直播流",
    "live.continuing": "正在续写",
    "live.waiting": "等待直播流",
    "live.reviewFull": "回看完整视频",
    "live.reviewBuffer": "回看直播缓冲",
    "live.ready": "直播画面已就绪 · 本轮 {current}s · 累计 {total}s",
    "live.buffering": "正在缓冲直播画面 · 本轮 {current}s · 累计 {total}s",
    "live.waitingFrame": "直播中 · 等待画面 · 累计 {total}s",
    "live.preloaded": "直播中 · 累计 {total}s · 预载 {buffered}s",
    "live.waitingMore": "直播中 · 累计 {total}s · 等待画面",
    "live.clickStart": "点击画面开始直播 · 累计 {total}s",
    "live.clickSound": "点击画面开启声音 · 累计 {total}s",
    "live.ended": "直播结束 · 约 {total}s",
    "live.finalReady": "完整视频已就绪",
    "live.review": "回看",
    "live.complete": "完成 {total}s",
    "live.added": "直播 +{total}s",
    "error.generationFailed": "生成没有成功。请检查场景描述、对话内容或稍后重试；详细原因在右侧调试日志。",
    "error.interruptedRetry": "生成中断。已保留当前画面，可以直接重试本轮对话。",
    "error.interruptedRestart": "生成中断。请稍后重试，或切换角色后重新开始通话。",
    "error.submitFailed": "提交失败。请检查场景描述和对话内容，或稍后重试。",
    "gpu.unknown": "算力未知",
    "gpu.unavailable": "GPU 状态不可用：{error}",
    "gpu.online": "算力在线",
    "gpu.ready": "算力就绪",
    "template.home.title": "居家讲解",
    "template.home.role": "年轻男性 · 真实居家工作台",
    "template.tech.title": "科技讲解",
    "template.tech.role": "年轻男性 · 专业科技演播室",
    "template.business.title": "商务顾问",
    "template.business.role": "成熟女性 · 真实咨询办公室",
    "template.education.title": "课程讲师",
    "template.education.role": "年轻女性 · 明亮教室角落",
    "template.wellness.title": "健康讲解员",
    "template.wellness.role": "年长男性 · 现代健康空间",
  },
};

let uiLanguage = localStorage.getItem(UI_LANGUAGE_STORAGE_KEY) === "zh" ? "zh" : "en";
let currentPhase = "idle";

function t(key, values = {}) {
  const template = I18N[uiLanguage]?.[key] ?? I18N.en[key] ?? key;
  return String(template).replace(/\{(\w+)\}/g, (_, name) => (
    Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : `{${name}}`
  ));
}

function phaseLabel(phase) {
  const normalized = String(phase || "idle");
  const key = `phase.${normalized}`;
  return I18N[uiLanguage]?.[key] || I18N.en[key] || normalized.replaceAll("_", " ");
}

function modeLabel(mode) {
  return String(mode || "").toLowerCase() === "t2av" ? t("mode.text") : String(mode || "Text");
}

function setServiceStatus(key) {
  serviceStatus.dataset.statusKey = key;
  serviceStatus.textContent = t(key);
}

function setReplyState(key) {
  replyBox.dataset.emptyState = key;
  replyBox.textContent = t(key);
}

function setReplyContent(value) {
  delete replyBox.dataset.emptyState;
  replyBox.textContent = String(value || "");
}

function updateDemoDrawerAction() {
  const action = demoDrawer?.querySelector(".drawer-action");
  if (action) action.textContent = t(demoDrawer.open ? "demo.collapse" : "demo.view");
}

function applyUiLanguage(language, { persist = true } = {}) {
  uiLanguage = language === "zh" ? "zh" : "en";
  if (persist) localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, uiLanguage);
  document.documentElement.lang = uiLanguage === "zh" ? "zh-CN" : "en";

  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((element) => {
    element.title = t(element.dataset.i18nTitle);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });
  languageButtons.forEach((button) => {
    const active = button.dataset.uiLanguage === uiLanguage;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });

  if (replyBox.dataset.emptyState) {
    replyBox.textContent = t(replyBox.dataset.emptyState);
  }
  if (serviceStatus.dataset.statusKey) {
    serviceStatus.textContent = t(serviceStatus.dataset.statusKey);
  }
  renderTemplates();
  setAspect(selectedAspect);
  setPhase(currentPhase);
  renderHistoryPanel();
  renderLiveStatus();
  updateDemoDrawerAction();
  if (!demos.length) {
    demoTitle.textContent = t("demo.waitingTitle");
    demoCaption.textContent = t("demo.emptyCaption");
    demoPlaceholder.textContent = t("demo.empty");
  }
}

let currentJobId = null;
let eventSource = null;
let eventRecoveryTimer = null;
let eventRecoveryFailures = 0;
let videos = [];
let liveSegments = [];
let liveSeen = new Set();
let livePlayIndex = -1;
let liveFinalUrl = "";
let currentJobLiveStartIndex = 0;
let liveStarted = false;
let liveEnded = false;
let liveStreamComplete = true;
let liveReviewMode = false;
let liveOverlapMode = false;
let liveMutedAutoplay = false;
let liveAutoplayBlocked = false;
let liveCanvasRaf = null;
let mseLive = null;
const requestedMode = "t2av";
let lastGpuSummary = "";
let systemBlocked = false;
let interactionBusy = false;
let serviceReady = false;
let readinessLogKey = "";
const DEFAULT_TEMPLATE_ID = "live5_canvas_smoke";
let selectedTemplateId = DEFAULT_TEMPLATE_ID;
let selectedAspect = "landscape";
let demos = [];
let activeDemoIndex = -1;
let demoRaf = null;
let demoLiveSource = null;
const CONVERSATION_STORAGE_KEY = "taomate_conversation_id";
const VISUAL_ROOT_STORAGE_KEY = "taomate_visual_root_key";
const ACTIVE_SESSION_STORAGE_KEY = "taomate_active_session_v2";
const SESSION_HISTORY_STORAGE_KEY = "taomate_session_history_v2";
const MAX_HISTORY_SESSIONS = 12;
const INITIAL_LIVE_BUFFER_SEGMENTS = 2;
const FALLBACK_LIVE_BUFFER_SEGMENTS = 2;
const FALLBACK_CONTINUATION_BUFFER_SEGMENTS = 2;
const MSE_LIVE_MIME_CANDIDATES = [
  'video/mp4; codecs="avc1.64001E, mp4a.40.2"',
  'video/mp4; codecs="avc1.64001F, mp4a.40.2"',
  'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
];
function newConversationId() {
  if (window.crypto?.randomUUID) return `conv_${window.crypto.randomUUID()}`;
  return `conv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

let conversationId = localStorage.getItem(CONVERSATION_STORAGE_KEY) || newConversationId();
localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
let visualRootKey = localStorage.getItem(VISUAL_ROOT_STORAGE_KEY) || "";
let activeSession = null;
let archivedSessions = [];

const ASPECT_PRESETS = {
  landscape: { labelKey: "aspect.landscape", width: 864, height: 480 },
  portrait: { labelKey: "aspect.portrait", width: 480, height: 864 },
};

const REFERENCE_TEMPLATES = [
  {
    id: "live5_canvas_smoke",
    titleKey: "template.home.title",
    roleKey: "template.home.role",
    scene:
      "eye-level tight medium close-up shot. A young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a charcoal gray zip-up hoodie over a white T-shirt sits upright in a contemporary home office. Only a narrow walnut desk edge is visible at the bottom below his hands; behind him, two closed neutral books and one small broad-leaf plant rest on a single oak shelf against an off-white plaster wall. The gray upholstered chair visibly supports his body. The presenter fills about two thirds of the frame height from head to mid-torso, his shoulders span roughly half the frame width, headroom and side margins stay compact, and his hands remain available in the lower quarter. A large diffused window at camera left is the dominant key light, while a white wall at camera right provides gentle neutral fill and consistent soft shadows. Natural 50mm-equivalent perspective, straight background verticals, moderate depth of field, crisp eyes, natural skin and fabric texture, controlled highlights, and polished natural color. The presenter keeps direct eye contact and uses one small slow hand gesture near chest level.",
    accent: "#c084fc",
  },
  {
    id: "tech_anchor",
    titleKey: "template.tech.title",
    roleKey: "template.tech.role",
    scene:
      "eye-level tight medium close-up shot. A young male technology presenter with warm fair skin, neat short black hair, tidy eyebrows, and a navy casual blazer over a white shirt sits upright in a restrained professional technology studio. Only a narrow brushed-aluminum desk edge is visible at the bottom below his hands; behind him, one small closed silver equipment case rests inside a recessed oak shelf set into straight matte graphite acoustic panels. All equipment remains on the rear shelf, leaving the foreground and his torso unobstructed. A low-backed charcoal chair stays fully behind and visibly supports his body, and every visible panel is matte and unmarked. The presenter fills about two thirds of the frame height from head to mid-torso, his shoulders span roughly half the frame width, headroom and side margins stay compact, and his hands remain available in the lower quarter. A large frosted studio window outside the frame at camera left is the dominant key light, while a white wall at camera right provides gentle neutral fill and one consistent shadow direction. Natural 50mm-equivalent perspective, straight background verticals, moderate depth of field, crisp eyes, natural skin and fabric texture, controlled highlights, and refined neutral color. The presenter keeps direct eye contact and uses one small slow hand gesture near chest level.",
    accent: "#2563eb",
  },
  {
    id: "business_consultant",
    titleKey: "template.business.title",
    roleKey: "template.business.role",
    scene:
      "eye-level tight medium close-up shot. A mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked behind one ear, natural makeup, a light gray blazer, and a white inner layer sits upright in a refined executive meeting room. Only a narrow band of an uncluttered matte light-gray meeting table is visible at the bottom below her hands; behind her, one small ceramic vase rests on a recessed sideboard against a continuous walnut wall panel. The upholstered chair visibly supports her body. The consultant fills about two thirds of the frame height from head to mid-torso, her shoulders span roughly half the frame width, headroom and side margins stay compact, and her hands remain available in the lower quarter. A large diffused window at camera left is the dominant key light, while a white wall at camera right provides gentle neutral fill and consistent soft shadows. Natural 50mm-equivalent perspective, straight background verticals, moderate depth of field, crisp eyes, natural skin and fabric texture, controlled highlights, and understated premium color. The consultant keeps direct eye contact, gives a small reassuring nod, and uses one measured hand gesture above the tabletop.",
    accent: "#0f766e",
  },
  {
    id: "education_coach",
    titleKey: "template.education.title",
    roleKey: "template.education.role",
    scene:
      "eye-level tight medium close-up shot. A young female course instructor with soft fair skin, clear glasses, a low ponytail, a gentle expression, and a pale green cardigan over a white top sits upright in a bright modern classroom corner. Only a narrow pale gray table edge is visible at the bottom below her hands; behind her, a blank matte whiteboard is mounted flush to the wall, while three plain closed books with blank spines and one small plant rest on a low oak bookcase. The chair visibly supports her body, and every object is aligned with its supporting surface. The instructor fills about two thirds of the frame height from head to mid-torso, her shoulders span roughly half the frame width, headroom and side margins stay compact, and her hands remain available in the lower quarter. A large diffused window at camera left is the dominant key light, while a white wall at camera right provides gentle neutral fill and consistent soft shadows. Natural 50mm-equivalent perspective, straight background verticals, moderate depth of field, crisp eyes, natural skin and fabric texture, controlled highlights, and fresh natural color. The instructor keeps warm eye contact and uses one small slow explaining gesture.",
    accent: "#7c3aed",
  },
  {
    id: "wellness_host",
    titleKey: "template.wellness.title",
    roleKey: "template.wellness.role",
    scene:
      "eye-level tight medium close-up shot. An older male wellness host with warm tan skin, short silver hair, soft wrinkles around the eyes, a trustworthy smile, and an off-white knit top sits upright in a calm contemporary wellness studio. Behind him, one ceramic bowl and one neatly folded towel rest on an oak sideboard, a potted olive tree stands in the corner, and a floor-length linen curtain hangs straight against a pale mineral-plaster wall. The linen chair visibly supports his body, and every object has a clear physical support. The host fills about two thirds of the frame height from head to mid-torso, his shoulders span roughly half the frame width, headroom and side margins stay compact, and his hands remain available in the lower quarter. Diffused daylight through the curtain at camera left is the dominant key light, while a white wall at camera right provides gentle neutral fill and consistent soft shadows. Natural 50mm-equivalent perspective, straight background verticals, moderate depth of field, crisp eyes, natural skin and knit texture, controlled highlights, and calm natural color. The host keeps warm eye contact and uses one small slow reassuring gesture near chest level.",
    accent: "#b45309",
  },
];

document.querySelectorAll(".aspect-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    setAspect(btn.dataset.aspect || "landscape");
  });
});

function getSelectedTemplate() {
  return REFERENCE_TEMPLATES.find((template) => template.id === selectedTemplateId) || null;
}

function getAspectPreset() {
  return ASPECT_PRESETS[selectedAspect] || ASPECT_PRESETS.landscape;
}

function normalizeSceneKey(scene) {
  return String(scene || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function sceneNeedsRefinement(scene) {
  const template = getSelectedTemplate();
  return !template || normalizeSceneKey(scene) !== normalizeSceneKey(template.scene);
}

function visualRootKeyFor(scene, aspect, templateId) {
  return JSON.stringify({
    scene: normalizeSceneKey(scene),
    template: templateId || "",
    aspect: aspect || "landscape",
    mode: requestedMode,
  });
}

function rememberVisualRoot(key) {
  visualRootKey = key || "";
  if (visualRootKey) {
    localStorage.setItem(VISUAL_ROOT_STORAGE_KEY, visualRootKey);
  } else {
    localStorage.removeItem(VISUAL_ROOT_STORAGE_KEY);
  }
}

function setAspect(aspect) {
  selectedAspect = ASPECT_PRESETS[aspect] ? aspect : "landscape";
  document.querySelectorAll(".aspect-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.aspect === selectedAspect);
  });
  if (videoFrame) {
    videoFrame.dataset.aspect = selectedAspect;
  }
  modeMetric.textContent = `${modeLabel(requestedMode)} · ${getAspectPreset().width}x${getAspectPreset().height}`;
}

function selectTemplate(templateId, applyScene = true) {
  selectedTemplateId = templateId || "";
  const template = getSelectedTemplate();
  document.querySelectorAll(".template-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.templateId === selectedTemplateId);
  });
  if (template && applyScene) {
    sceneDescription.value = template.scene;
  }
}

function syncTemplateSelectionToScene() {
  const template = getSelectedTemplate();
  if (template && normalizeSceneKey(sceneDescription.value) !== normalizeSceneKey(template.scene)) {
    selectTemplate("", false);
  }
}

function renderTemplates() {
  if (!templateGrid) return;
  templateGrid.innerHTML = "";
  REFERENCE_TEMPLATES.forEach((template) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "template-card";
    card.dataset.templateId = template.id;
    card.style.setProperty("--template-accent", template.accent);
    card.classList.toggle("active", template.id === selectedTemplateId);

    const body = document.createElement("div");
    body.className = "template-body";
    const title = document.createElement("strong");
    title.textContent = t(template.titleKey);
    const role = document.createElement("span");
    role.textContent = t(template.roleKey);
    body.append(title, role);

    const check = document.createElement("span");
    check.className = "template-check";
    check.textContent = "✓";
    check.setAttribute("aria-hidden", "true");

    card.append(body, check);
    card.addEventListener("click", () => selectTemplate(template.id, true));
    templateGrid.appendChild(card);
  });
}

function formatClock(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const mm = Math.floor(safe / 60).toString().padStart(2, "0");
  const ss = Math.floor(safe % 60).toString().padStart(2, "0");
  return `${mm}:${ss}`;
}

function demoCaptionForTime(demo, time, duration) {
  const segments = Array.isArray(demo?.segments) ? demo.segments : [];
  const speeches = segments
    .map((segment) => String(segment.speech || "").trim())
    .filter(Boolean);
  const text = speeches.length ? speeches.join(" ") : String(demo?.caption || "").trim();
  if (!text) return t("demo.waitingLine");
  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 10;
  const ratio = Math.min(1, Math.max(0.12, time / safeDuration));
  const chars = Math.max(6, Math.ceil(text.length * ratio));
  return text.slice(0, chars);
}

function renderDemoDots() {
  if (!demoDots) return;
  demoDots.innerHTML = "";
  demos.forEach((demo, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "demo-dot";
    btn.classList.toggle("active", index === activeDemoIndex);
    btn.title = demo.title || `Demo ${index + 1}`;
    btn.addEventListener("click", () => playDemo(index));
    demoDots.appendChild(btn);
  });
}

function updateDemoOverlay(ts = 0) {
  if (!demoPlayer || !demos.length || activeDemoIndex < 0) return;
  if (demoDrawer && !demoDrawer.open) {
    demoRaf = null;
    return;
  }
  const demo = demos[activeDemoIndex];
  const duration = demoPlayer.duration;
  demoCaption.textContent = demoCaptionForTime(demo, demoPlayer.currentTime, duration);
  demoClock.textContent = `${formatClock(demoPlayer.currentTime)} / ${formatClock(duration)}`;
  demoRaf = requestAnimationFrame(updateDemoOverlay);
}

function playDemo(index) {
  if (!demoPlayer || !demos.length) return;
  activeDemoIndex = Math.max(0, Math.min(index, demos.length - 1));
  const demo = demos[activeDemoIndex];
  demoTitle.textContent = demo.title || `Demo ${activeDemoIndex + 1}`;
  demoCaption.textContent = demo.caption || t("demo.preparingLine");
  demoMode.textContent = `${modeLabel(demo.mode || "t2av")} · ${demo.video_width || "--"}x${demo.video_height || "--"}`;
  demoFps.textContent = "";
  demoClock.textContent = "00:00";
  if (demoLiveBtn) {
    demoLiveBtn.disabled = !demo.live_events_url;
  }
  demoPlaceholder.classList.add("hidden");
  demoPlayer.src = demo.video_url;
  if (demoDrawer?.open) {
    demoPlayer.play().catch(() => {});
  } else {
    demoPlayer.pause();
  }
  if (demoRaf) cancelAnimationFrame(demoRaf);
  demoRaf = demoDrawer?.open ? requestAnimationFrame(updateDemoOverlay) : null;
  renderDemoDots();
}

function closeDemoLiveSource() {
  if (demoLiveSource) {
    demoLiveSource.close();
    demoLiveSource = null;
  }
}

function startDemoLiveReference() {
  if (activeDemoIndex < 0 || !demos[activeDemoIndex]?.live_events_url) return;
  const demo = demos[activeDemoIndex];
  closeEvents();
  closeDemoLiveSource();
  resetLiveStage(true);
  logBox.textContent = "";
  currentJobId = null;
  setBusy(false);
  setPhase("streaming");
  stageTitle.textContent = demo.title || t("demo.livePreview");
  modeMetric.textContent = `${modeLabel(demo.mode || "demo")} · ${demo.video_width || "--"}x${demo.video_height || "--"}`;
  log(`demo reference live: ${demo.title || demo.task_id}`);
  demoLiveSource = new EventSource(`${demo.live_events_url}?initial_delay=2&interval=1`);
  demoLiveSource.onmessage = (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    if (event.event === "stage") {
      setPhase(event.phase || "streaming");
      log(`demo live buffering ${event.delay || 0}s`);
    } else if (event.event === "asset") {
      if (event.kind === "image") {
        showGeneratedPreviewFrame(event.url);
      } else {
        addVideo(event.url, "");
      }
      setPhase("streaming");
    } else if (event.event === "done") {
      setPhase("succeeded");
      log("demo reference live completed");
    } else if (event.event === "closed") {
      closeDemoLiveSource();
    }
  };
  demoLiveSource.onerror = () => {
    log("demo reference live disconnected");
    closeDemoLiveSource();
  };
}

async function refreshDemos() {
  try {
    const res = await fetch("/api/demos");
    if (!res.ok) return;
    const data = await res.json();
    demos = (data.demos || []).filter((demo) => demo.video_url);
    if (!demos.length) {
      demoTitle.textContent = t("demo.waitingTitle");
      demoCaption.textContent = t("demo.emptyCaption");
      demoMode.textContent = phaseLabel("idle");
      demoFps.textContent = "";
      demoClock.textContent = "00:00";
      demoPlaceholder.classList.remove("hidden");
      renderDemoDots();
      return;
    }
    if (activeDemoIndex < 0 || activeDemoIndex >= demos.length) {
      playDemo(0);
    } else {
      renderDemoDots();
    }
  } catch (err) {
    log(`demo load failed: ${err.message || err}`);
  }
}

function setBusy(isBusy) {
  interactionBusy = Boolean(isBusy);
  const jobActive = Boolean(currentJobId);
  const historyLocked = interactionBusy || jobActive;
  form.querySelector("button[type='submit']").disabled = isBusy || !serviceReady;
  previewBtn.disabled = isBusy || !serviceReady;
  cancelBtn.disabled = !jobActive;
  resetSessionBtn.disabled = historyLocked;
  setServiceStatus(
    historyLocked
      ? "status.onAir"
      : (!serviceReady ? "status.starting" : (systemBlocked ? "status.busy" : "status.ready")),
  );
  renderHistoryPanel();
}

function setPreviewBusy(isBusy) {
  previewBtn.disabled = isBusy || !serviceReady;
  if (!currentJobId) {
    form.querySelector("button[type='submit']").disabled = isBusy || !serviceReady;
  }
}

function log(line) {
  if (!line) return;
  const atBottom = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 20;
  logBox.textContent += `${line}\n`;
  if (atBottom) logBox.scrollTop = logBox.scrollHeight;
}

function setPhase(phase) {
  const normalized = phase || "idle";
  currentPhase = normalized;
  const label = phaseLabel(normalized);
  phaseMetric.textContent = label;
  stageTitle.textContent = liveStarted ? t("stage.liveConversation") : label;
}

function setSystemNotice(message, level = "info") {
  if (!systemNotice) return;
  if (!message) {
    systemNotice.className = "system-notice hidden";
    systemNotice.textContent = "";
    return;
  }
  systemNotice.className = `system-notice ${level}`;
  systemNotice.textContent = message;
}

function showUserFacingError(message) {
  setSystemNotice(
    message || t("error.generationFailed"),
    "warning",
  );
}

function renderSegments(segments) {
  segmentList.innerHTML = "";
  (segments || []).forEach((segment) => {
    const item = document.createElement("div");
    item.className = "segment-item";
    const suffix = segment.is_transition ? ` · ${t("segment.waiting")}` : "";
    const title = document.createElement("strong");
    title.textContent = `${t("segment.title", { number: Number(segment.segment_id) + 1 })}${suffix}`;
    const speech = document.createElement("p");
    speech.textContent = segment.speech || "";
    const meta = document.createElement("div");
    meta.className = "segment-meta";
    meta.textContent = `${segment.emotion || "calm"} · ${segment.action || "small gesture"}`;
    const details = document.createElement("details");
    details.className = "prompt-details";
    const summary = document.createElement("summary");
    summary.textContent = t("debug.viewPrompt");
    const prompt = document.createElement("pre");
    prompt.textContent = segment.prompt || "";
    details.append(summary, prompt);
    item.append(title, speech, meta, details);
    segmentList.appendChild(item);
  });
}

function resetTextPreview() {
  setReplyState("reply.generating");
  segmentList.innerHTML = "";
}

function startNewConversation(clearScene = false) {
  closeEvents();
  saveSessionToHistory(activeSession);
  currentJobId = null;
  conversationId = newConversationId();
  localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
  rememberVisualRoot("");
  activeSession = createSession(conversationId);
  resetLiveStage(true);
  resetTextPreview();
  logBox.textContent = "";
  if (clearScene) {
    userText.value = "";
    selectTemplate(DEFAULT_TEMPLATE_ID, true);
  }
  setBusy(false);
  setPhase("new_conversation");
  persistActiveSession();
  log(`new conversation ${conversationId}`);
}

async function previewSegments() {
  const scene = sceneDescription.value.trim();
  const text = userText.value.trim();
  if (!scene || !text) return;
  const refineScene = sceneNeedsRefinement(scene);
  setPreviewBusy(true);
  setPhase("previewing_prompt");
  resetTextPreview();
  try {
    const res = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene_description: scene,
        user_text: text,
        mode: requestedMode,
        has_first_frame: false,
        aspect_ratio: selectedAspect,
        template_id: selectedTemplateId,
        refine_scene: refineScene,
        conversation_id: conversationId,
      }),
    });
    if (!res.ok) {
      let message = await res.text();
      try {
        const parsed = JSON.parse(message);
        message = parsed.detail || message;
      } catch {}
      throw new Error(message);
    }
    const data = await res.json();
    setReplyContent(data.reply || "");
    renderSegments(data.segments || []);
    modeMetric.textContent = `${modeLabel(data.mode || requestedMode)} · ${data.video_width || getAspectPreset().width}x${data.video_height || getAspectPreset().height}`;
    setPhase("preview_ready");
    log(`preview ${data.mode || requestedMode}: ${(data.segments || []).length} segment(s), no GPU used`);
    const firstPrompt = (data.segments || [])[0]?.prompt;
    if (firstPrompt) {
      log(`prompt[0]: ${firstPrompt.slice(0, 360).replaceAll("\n", " ")}`);
    }
  } catch (err) {
    setPhase("preview_failed");
    log(`preview failed: ${err.message || err}`);
  } finally {
    setPreviewBusy(false);
  }
}

function videoSortKey(url) {
  const taskMatch = url.match(/\/media\/([^/]+)\//);
  const taskId = taskMatch ? taskMatch[1] : "";
  const file = url.split("/").pop() || "";
  const streamMatch = file.match(/_stream(\d+)\.mp4$/);
  if (streamMatch) return `${taskId}|0|${streamMatch[1].padStart(8, "0")}|${url}`;
  const chunkMatch = file.match(/_chunk(\d+)\.mp4$/);
  if (chunkMatch) return `${taskId}|1|${chunkMatch[1].padStart(8, "0")}|${url}`;
  return `${taskId}|2|99999999|${url}`;
}

function taskIdFromUrl(url) {
  const taskMatch = String(url || "").match(/\/media\/([^/]+)\//);
  return taskMatch ? taskMatch[1] : "";
}

function readJsonStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function uniqueSortedUrls(urls) {
  return Array.from(new Set((urls || []).filter(Boolean)));
}

function formatSessionTime(value) {
  const date = new Date(value || Date.now());
  if (Number.isNaN(date.getTime())) return "";
  return `${(date.getMonth() + 1).toString().padStart(2, "0")}/${date.getDate().toString().padStart(2, "0")} ${date.getHours().toString().padStart(2, "0")}:${date.getMinutes().toString().padStart(2, "0")}`;
}

function summarizeLine(text, fallback = "") {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  return compact ? compact.slice(0, 34) : (fallback || t("history.untitled"));
}

function normalizeTurn(turn) {
  const item = turn && typeof turn === "object" ? turn : {};
  return {
    taskId: String(item.taskId || item.task_id || ""),
    userText: String(item.userText || item.user_text || ""),
    reply: String(item.reply || ""),
    status: String(item.status || ""),
    phase: String(item.phase || ""),
    createdAt: Number(item.createdAt || Date.now()),
    updatedAt: Number(item.updatedAt || item.createdAt || Date.now()),
    segments: Array.isArray(item.segments) ? item.segments : [],
    videos: uniqueSortedUrls(item.videos || []),
  };
}

function normalizeSession(session, fallbackId = conversationId) {
  const source = session && typeof session === "object" ? session : {};
  const turns = Array.isArray(source.turns) ? source.turns.map(normalizeTurn) : [];
  const turnVideos = turns.flatMap((turn) => turn.videos || []);
  const id = String(source.id || source.conversationId || fallbackId || newConversationId());
  return {
    id,
    createdAt: Number(source.createdAt || Date.now()),
    updatedAt: Number(source.updatedAt || Date.now()),
    archivedAt: source.archivedAt ? Number(source.archivedAt) : null,
    title: String(source.title || ""),
    scene: String(source.scene || ""),
    templateId: String(source.templateId || ""),
    aspect: String(source.aspect || "landscape"),
    visualRootKey: String(source.visualRootKey || ""),
    mode: String(source.mode || "t2av"),
    turns,
    videos: uniqueSortedUrls([...(source.videos || []), ...turnVideos]),
  };
}

function createSession(id = newConversationId()) {
  return normalizeSession({
    id,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    aspect: selectedAspect,
    templateId: selectedTemplateId,
    scene: sceneDescription.value || "",
    visualRootKey,
    mode: requestedMode,
    turns: [],
    videos: [],
  }, id);
}

function sessionHasContent(session) {
  return Boolean(session && ((session.turns || []).length || (session.videos || []).length));
}

function loadStoredSessions() {
  archivedSessions = (readJsonStorage(SESSION_HISTORY_STORAGE_KEY, []) || [])
    .map((item) => normalizeSession(item))
    .filter(sessionHasContent);
  activeSession = normalizeSession(
    readJsonStorage(ACTIVE_SESSION_STORAGE_KEY, null),
    conversationId,
  );
  conversationId = activeSession.id || conversationId;
  localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
  visualRootKey = activeSession.visualRootKey || visualRootKey;
  if (visualRootKey) {
    localStorage.setItem(VISUAL_ROOT_STORAGE_KEY, visualRootKey);
  }
}

function writeHistoryStorage() {
  localStorage.setItem(
    SESSION_HISTORY_STORAGE_KEY,
    JSON.stringify(archivedSessions.slice(0, MAX_HISTORY_SESSIONS)),
  );
}

function saveSessionToHistory(session) {
  const normalized = normalizeSession(session);
  if (!sessionHasContent(normalized)) return;
  normalized.archivedAt = Date.now();
  archivedSessions = [
    normalized,
    ...archivedSessions.filter((item) => item.id !== normalized.id),
  ].slice(0, MAX_HISTORY_SESSIONS);
  writeHistoryStorage();
}

function beginFreshPageSession() {
  const previous = normalizeSession(activeSession);
  const preferences = {
    mode: "t2av",
    aspect: previous.aspect || "landscape",
    templateId: previous.scene ? previous.templateId : (previous.templateId || DEFAULT_TEMPLATE_ID),
    scene: previous.scene || "",
  };
  saveSessionToHistory(previous);

  conversationId = newConversationId();
  localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
  rememberVisualRoot("");
  activeSession = normalizeSession({
    id: conversationId,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    scene: preferences.scene,
    templateId: preferences.templateId,
    aspect: preferences.aspect,
    visualRootKey: "",
    mode: preferences.mode,
    turns: [],
    videos: [],
  }, conversationId);
  localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, JSON.stringify(activeSession));
  return preferences;
}

function persistActiveSession({ mirrorToHistory = false } = {}) {
  if (!activeSession) return;
  activeSession.id = conversationId;
  activeSession.updatedAt = Date.now();
  activeSession.scene = sceneDescription.value || activeSession.scene || "";
  activeSession.templateId = selectedTemplateId;
  activeSession.aspect = selectedAspect || activeSession.aspect || "landscape";
  activeSession.visualRootKey = visualRootKey || activeSession.visualRootKey || "";
  activeSession.mode = requestedMode;
  activeSession.videos = uniqueSortedUrls(activeSession.videos || []);
  localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, JSON.stringify(activeSession));
  localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
  if (mirrorToHistory) {
    saveSessionToHistory(activeSession);
  }
  renderHistoryPanel();
}

function startFreshActiveSession(id = newConversationId()) {
  conversationId = id;
  activeSession = createSession(conversationId);
  localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
  persistActiveSession();
}

function currentTurn(taskId = currentJobId) {
  if (!activeSession || !taskId) return null;
  return activeSession.turns.find((turn) => turn.taskId === taskId) || null;
}

function upsertTurn(taskId, patch = {}) {
  if (!activeSession || !taskId) return;
  let turn = currentTurn(taskId);
  if (!turn) {
    turn = normalizeTurn({ taskId, createdAt: Date.now() });
    activeSession.turns.push(turn);
  }
  Object.assign(turn, patch, { updatedAt: Date.now() });
  if (patch.videos) {
    turn.videos = uniqueSortedUrls(patch.videos);
  }
  activeSession.updatedAt = Date.now();
  if (!activeSession.title) {
    activeSession.title = summarizeLine(turn.userText || turn.reply, t("history.realtime"));
  }
  persistActiveSession();
}

function recordVideoForTask(url, taskId = currentJobId) {
  if (!activeSession || !url || !taskId) return;
  activeSession.videos = uniqueSortedUrls([...(activeSession.videos || []), url]);
  const turn = currentTurn(taskId);
  if (turn) {
    turn.videos = uniqueSortedUrls([...(turn.videos || []), url]);
    turn.updatedAt = Date.now();
  }
  persistActiveSession();
}

function sessionVideoUrls(session) {
  return uniqueSortedUrls([
    ...((session || {}).videos || []),
    ...(((session || {}).turns || []).flatMap((turn) => turn.videos || [])),
  ]);
}

function loadSessionTimeline(session, { activate = false, autoplay = false } = {}) {
  if (interactionBusy || currentJobId) return;
  const normalized = normalizeSession(session);
  if (activate) {
    activeSession = normalized;
    conversationId = normalized.id;
    localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
    rememberVisualRoot(normalized.visualRootKey || "");
    if (normalized.scene) sceneDescription.value = normalized.scene;
    selectTemplate(normalized.templateId || "", false);
    syncTemplateSelectionToScene();
    if (normalized.aspect) setAspect(normalized.aspect);
    persistActiveSession();
  }
  resetMseLive();
  videos = sessionVideoUrls(normalized);
  liveSegments = videos.filter(isPreviewVideo);
  liveSeen = new Set(liveSegments);
  liveFinalUrl = videos.filter((url) => !isPreviewVideo(url)).slice(-1)[0] || "";
  currentJobLiveStartIndex = liveSegments.length;
  livePlayIndex = liveSegments.length ? 0 : -1;
  liveStarted = false;
  liveEnded = false;
  liveStreamComplete = true;
  liveReviewMode = false;
  liveOverlapMode = false;
  liveAutoplayBlocked = false;
  if (liveSegments.length || liveFinalUrl) {
    enterReviewMode(autoplay);
  } else {
    resetLiveStage(true);
  }
  const lastTurn = normalized.turns[normalized.turns.length - 1];
  if (lastTurn) {
    if (lastTurn.reply) {
      setReplyContent(lastTurn.reply);
    } else {
      setReplyState("reply.generating");
    }
    renderSegments(lastTurn.segments || []);
  }
  renderHistoryPanel();
}

function renderHistoryPanel() {
  if (!historyList) return;
  const historyLocked = interactionBusy || Boolean(currentJobId);
  const rows = [];
  if (sessionHasContent(activeSession)) {
    rows.push({ session: normalizeSession(activeSession), active: true });
  }
  archivedSessions
    .filter((session) => !activeSession || session.id !== activeSession.id)
    .forEach((session) => rows.push({ session: normalizeSession(session), active: false }));
  const totalTurns = rows.reduce((sum, row) => sum + (row.session.turns || []).length, 0);
  if (historyCount) historyCount.textContent = t("history.count", { count: totalTurns });
  historyList.innerHTML = "";
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = t("history.none");
    historyList.appendChild(empty);
    return;
  }
  rows.forEach(({ session, active }) => {
    const details = document.createElement("details");
    details.className = "history-card";
    details.open = active;
    const summary = document.createElement("summary");
    summary.className = "history-summary";
    const titleWrap = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = active
      ? `${t("history.currentPrefix")} · ${summarizeLine(session.title || session.scene, t("history.realtime"))}`
      : summarizeLine(session.title || session.scene, t("history.archived"));
    const meta = document.createElement("span");
    meta.textContent = t("history.meta", {
      time: formatSessionTime(session.updatedAt),
      count: (session.turns || []).length,
      seconds: sessionVideoUrls(session).filter(isPreviewVideo).length,
    });
    titleWrap.append(title, meta);
    const actions = document.createElement("div");
    actions.className = "history-actions";
    const reviewBtn = document.createElement("button");
    reviewBtn.type = "button";
    reviewBtn.textContent = t("history.review");
    reviewBtn.disabled = historyLocked;
    reviewBtn.addEventListener("click", (event) => {
      event.preventDefault();
      loadSessionTimeline(session, { activate: false, autoplay: false });
    });
    const continueBtn = document.createElement("button");
    continueBtn.type = "button";
    continueBtn.textContent = active ? t("history.current") : t("history.continue");
    continueBtn.disabled = active || historyLocked;
    continueBtn.addEventListener("click", (event) => {
      event.preventDefault();
      loadSessionTimeline(session, { activate: true, autoplay: false });
      log(`restore conversation ${conversationId}`);
    });
    actions.append(reviewBtn, continueBtn);
    summary.append(titleWrap, actions);
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "history-body";
    (session.turns || []).forEach((turn, index) => {
      const turnEl = document.createElement("div");
      turnEl.className = "history-turn";
      const head = document.createElement("div");
      head.className = "history-turn-head";
      head.innerHTML = `<strong>${t("history.turn", { number: index + 1 })}</strong><span>${turn.status ? phaseLabel(turn.status) : t("history.saved")}</span>`;
      const user = document.createElement("p");
      user.className = "history-user";
      user.textContent = turn.userText || "";
      const reply = document.createElement("p");
      reply.className = "history-reply";
      reply.textContent = turn.reply || "";
      const strip = document.createElement("div");
      strip.className = "history-video-strip";
      (turn.videos || []).filter(isPreviewVideo).forEach((url, videoIndex) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "history-video-chip";
        item.textContent = `${videoIndex + 1}s`;
        item.addEventListener("click", () => {
          loadSessionTimeline(session, { activate: false, autoplay: false });
          const idx = liveSegments.indexOf(url);
          if (idx >= 0) enterReviewMode(true, idx);
        });
        strip.appendChild(item);
      });
      turnEl.append(head, user, reply);
      if (strip.children.length) turnEl.appendChild(strip);
      body.appendChild(turnEl);
    });
    details.appendChild(body);
    historyList.appendChild(details);
  });
}

function isPreviewVideo(url) {
  return /_(stream|chunk)\d+\.mp4$/.test(url || "");
}

function mediaSrc(el) {
  return el?.currentSrc || el?.src || "";
}

function clearMedia(el) {
  if (!el) return;
  el.pause();
  el.removeAttribute("src");
  el.load();
}

function canUseMseLive() {
  return Boolean(resolveMseLiveMime());
}

function liveStartBufferSegments() {
  if (canUseMseLive()) return INITIAL_LIVE_BUFFER_SEGMENTS;
  return liveOverlapMode
    ? FALLBACK_CONTINUATION_BUFFER_SEGMENTS
    : FALLBACK_LIVE_BUFFER_SEGMENTS;
}

function hasEnoughLiveBufferToStart() {
  const currentBuffered = Math.max(0, liveSegments.length - currentJobLiveStartIndex);
  return currentBuffered >= liveStartBufferSegments() || Boolean(liveFinalUrl);
}

function resolveMseLiveMime() {
  if (!window.MediaSource || typeof window.MediaSource.isTypeSupported !== "function") {
    return "";
  }
  return MSE_LIVE_MIME_CANDIDATES.find((mime) => window.MediaSource.isTypeSupported(mime)) || "";
}

function resetMseLive() {
  if (!mseLive) return;
  const state = mseLive;
  mseLive = null;
  try {
    if (state.sourceBuffer && state.mediaSource?.readyState === "open" && !state.sourceBuffer.updating) {
      state.mediaSource.endOfStream();
    }
  } catch {}
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
  }
}

function startMseLivePlayback() {
  if (!mseLive || liveStarted) return;
  const reviewMode = mseLive.mode === "review";
  const pendingSeek = Math.max(0, Number(mseLive.pendingSeek) || 0);
  liveReviewMode = reviewMode;
  liveOverlapMode = false;
  liveStarted = true;
  liveEnded = false;
  livePlayIndex = reviewMode
    ? Math.max(0, Math.min(liveSegments.length - 1, Math.floor(pendingSeek)))
    : Math.max(0, Number(mseLive.baseIndex) || 0);
  setActivePlayer(primaryPlayer);
  player.muted = liveMutedAutoplay;
  player.controls = false;
  emptyStage.classList.add("hidden");
  startLiveCanvasLoop();
  if (reviewMode && pendingSeek > 0) {
    seekActivePlayerTo(pendingSeek);
  }
  if (mseLive.autoplay !== false) {
    requestLivePlay();
  }
  renderLiveStatus();
}

function maybeEndMseLiveStream() {
  if (!mseLive || !mseLive.done || !mseLive.sourceBuffer || !mseLive.mediaSource) return;
  if (mseLive.queue.length > 0 || mseLive.appending || mseLive.sourceBuffer.updating) return;
  if (mseLive.mediaSource.readyState !== "open") return;
  try {
    mseLive.mediaSource.endOfStream();
  } catch {}
}

function shouldStartMseLivePlayback() {
  if (!mseLive || liveStarted || mseLive.appended <= 0) return false;
  if (mseLive.mode === "review") {
    return mseLive.appended >= Math.max(1, Number(mseLive.requiredStartSegments) || 1);
  }
  if (mseLive.done) return true;
  return mseLive.appended >= INITIAL_LIVE_BUFFER_SEGMENTS;
}

function fallbackFromMse(state, message) {
  if (message) log(message);
  const reviewMode = state?.mode === "review";
  const autoplay = state?.autoplay !== false;
  const fallbackIndex = Math.max(
    0,
    Math.min(liveSegments.length - 1, Math.floor(Number(state?.pendingSeek) || 0)),
  );
  resetMseLive();
  if (reviewMode && liveSegments.length) {
    liveReviewMode = true;
    liveStarted = false;
    livePlayIndex = fallbackIndex - 1;
    playNextLiveSegment({ review: true, autoplay });
  } else if (!liveStarted && liveSegments.length) {
    playNextLiveSegment();
  }
}

function pumpMseLiveQueue() {
  const state = mseLive;
  if (!state || !state.open || !state.sourceBuffer || state.appending) return;
  if (state.sourceBuffer.updating) return;
  if (state.queue.length === 0) {
    maybeEndMseLiveStream();
    return;
  }
  const item = state.queue[0];
  state.appending = true;
  fetch(item.url)
    .then((res) => {
      if (!res.ok) throw new Error(`fetch ${res.status}`);
      return res.arrayBuffer();
    })
    .then((buffer) => {
      if (mseLive !== state || !state.sourceBuffer) return;
      state.currentUrl = item.url;
      state.sourceBuffer.appendBuffer(buffer);
    })
    .catch((err) => {
      state.failed = true;
      state.appending = false;
      fallbackFromMse(state, `MSE ${state.mode} fallback: ${err.message || err}`);
    });
}

function ensureMseLive(options = {}) {
  if (mseLive?.active) return true;
  const mseMime = resolveMseLiveMime();
  if (!mseMime) return false;
  const mode = options.mode === "review" ? "review" : "live";
  const pendingSeek = Math.max(0, Number(options.pendingSeek) || 0);
  const mediaSource = new MediaSource();
  const objectUrl = URL.createObjectURL(mediaSource);
  mseLive = {
    active: true,
    mode,
    autoplay: options.autoplay !== false,
    mediaSource,
    objectUrl,
    sourceBuffer: null,
    queue: [],
    queuedUrls: new Set(),
    open: false,
    appending: false,
    appended: 0,
    currentUrl: "",
    baseIndex: mode === "review" ? 0 : Math.max(0, currentJobLiveStartIndex),
    pendingSeek,
    requiredStartSegments: Math.max(
      INITIAL_LIVE_BUFFER_SEGMENTS,
      Math.ceil(pendingSeek) + 1,
    ),
    done: false,
    failed: false,
    mime: mseMime,
  };
  setActivePlayer(primaryPlayer);
  clearMedia(secondaryPlayer);
  primaryPlayer.controls = false;
  primaryPlayer.src = objectUrl;
  primaryPlayer.load();
  mediaSource.addEventListener("sourceopen", () => {
    if (!mseLive || mseLive.mediaSource !== mediaSource) return;
    try {
      const sourceBuffer = mediaSource.addSourceBuffer(mseMime);
      sourceBuffer.mode = "sequence";
      sourceBuffer.addEventListener("updateend", () => {
        if (!mseLive || mseLive.mediaSource !== mediaSource) return;
        const appendedUrl = mseLive.currentUrl;
        if (appendedUrl && mseLive.queue[0]?.url === appendedUrl) {
          mseLive.queue.shift();
          mseLive.appended += 1;
          mseLive.currentUrl = "";
        }
        mseLive.appending = false;
        if (shouldStartMseLivePlayback()) {
          startMseLivePlayback();
        }
        renderLiveStatus();
        pumpMseLiveQueue();
      });
      sourceBuffer.addEventListener("error", () => {
        if (!mseLive || mseLive.mediaSource !== mediaSource) return;
        fallbackFromMse(mseLive, `MSE ${mseLive.mode} fallback: sourceBuffer error`);
      });
      mseLive.sourceBuffer = sourceBuffer;
      mseLive.open = true;
      pumpMseLiveQueue();
    } catch (err) {
      fallbackFromMse(mseLive, `MSE ${mode} unavailable: ${err.message || err}`);
    }
  }, { once: true });
  return true;
}

function appendMseLiveSegment(url) {
  if (!ensureMseLive() || !mseLive) return false;
  if (mseLive.queuedUrls.has(url)) return true;
  mseLive.queuedUrls.add(url);
  mseLive.queue.push({ url });
  pumpMseLiveQueue();
  return true;
}

function finishMseLiveStream() {
  liveStreamComplete = true;
  if (!mseLive) return;
  mseLive.done = true;
  if (shouldStartMseLivePlayback()) {
    startMseLivePlayback();
  }
  maybeEndMseLiveStream();
}

function startMseReviewPlayback(autoplay = false, startTime = 0) {
  if (!liveSegments.length || !canUseMseLive()) return false;
  resetMseLive();
  liveReviewMode = true;
  liveStarted = false;
  liveEnded = false;
  livePlayIndex = -1;
  const seekTime = Math.min(
    Math.max(0, Number(startTime) || 0),
    Math.max(0, liveSegments.length - 0.001),
  );
  if (!ensureMseLive({ mode: "review", autoplay, pendingSeek: seekTime })) {
    return false;
  }
  mseLive.requiredStartSegments = Math.min(
    liveSegments.length,
    Math.max(INITIAL_LIVE_BUFFER_SEGMENTS, Math.floor(seekTime) + 2),
  );
  liveSegments.forEach((url) => appendMseLiveSegment(url));
  mseLive.done = liveStreamComplete;
  maybeEndMseLiveStream();
  renderLiveStatus();
  return true;
}

function clearLiveCanvas() {
  if (!liveCanvas || !liveCanvasCtx) return;
  liveCanvasCtx.clearRect(0, 0, liveCanvas.width || 1, liveCanvas.height || 1);
  liveCanvas.classList.remove("has-frame");
}

function showGeneratedPreviewFrame(url) {
  if (!url || !liveCanvas || !liveCanvasCtx) return;
  const img = new Image();
  img.onload = () => {
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) return;
    if (liveCanvas.width !== width || liveCanvas.height !== height) {
      liveCanvas.width = width;
      liveCanvas.height = height;
    }
    liveCanvasCtx.drawImage(img, 0, 0, width, height);
    liveCanvas.classList.add("has-frame");
    emptyStage.classList.add("hidden");
    liveOverlay.classList.remove("hidden");
    liveOverlayText.textContent = t("live.firstFrameReady");
    chunkMetric.textContent = t("live.firstFrameConnecting");
  };
  img.src = url;
}

function drawLiveCanvasFrame() {
  liveCanvasRaf = null;
  if (!liveCanvas || !liveCanvasCtx) return;
  if (player && player.readyState >= 2 && player.videoWidth > 0 && player.videoHeight > 0) {
    if (liveCanvas.width !== player.videoWidth || liveCanvas.height !== player.videoHeight) {
      liveCanvas.width = player.videoWidth;
      liveCanvas.height = player.videoHeight;
    }
    liveCanvasCtx.drawImage(player, 0, 0, liveCanvas.width, liveCanvas.height);
    liveCanvas.classList.add("has-frame");
  }
  if (liveStarted || liveReviewMode || liveOverlapMode) {
    liveCanvasRaf = window.requestAnimationFrame(drawLiveCanvasFrame);
  }
}

function startLiveCanvasLoop() {
  if (!liveCanvas || liveCanvasRaf) return;
  liveCanvasRaf = window.requestAnimationFrame(drawLiveCanvasFrame);
}

function stopLiveCanvasLoop() {
  if (!liveCanvasRaf) return;
  window.cancelAnimationFrame(liveCanvasRaf);
  liveCanvasRaf = null;
}

function setActivePlayer(nextPlayer) {
  if (!nextPlayer || nextPlayer === player) return;
  const previous = player;
  previous.classList.remove("active");
  previous.classList.add("standby");
  nextPlayer.classList.remove("standby");
  nextPlayer.classList.add("active");
  player = nextPlayer;
  standbyPlayer = previous;
}

function prepareStandby(index) {
  if (!standbyPlayer || index < 0 || index >= liveSegments.length || liveReviewMode) return;
  const nextUrl = liveSegments[index];
  if (!nextUrl || mediaSrc(standbyPlayer).endsWith(nextUrl)) return;
  standbyPlayer.pause();
  standbyPlayer.muted = true;
  standbyPlayer.src = nextUrl;
  standbyPlayer.load();
}

function liveDurationSeconds() {
  const useFinalReview = shouldUseFinalVideoForReview();
  if (liveReviewMode && mseLive?.mode === "review") {
    const mediaDuration = Number.isFinite(player.duration) && player.duration > 0
      ? player.duration
      : 0;
    return Math.max(liveSegments.length, mediaDuration);
  }
  if (liveReviewMode && useFinalReview && Number.isFinite(player.duration) && player.duration > 0) {
    if (!liveSegments.length) {
      return Math.max(liveSegments.length, 0);
    }
    return player.duration;
  }
  return Math.max(liveSegments.length, 0);
}

function shouldUseFinalVideoForReview() {
  if (!liveFinalUrl) return false;
  if (!liveSegments.length) return true;
  const finalTaskId = taskIdFromUrl(liveFinalUrl);
  const taskIds = new Set(liveSegments.map(taskIdFromUrl).filter(Boolean));
  return Boolean(finalTaskId && taskIds.size === 1 && taskIds.has(finalTaskId));
}

function updateLiveControls() {
  if (!liveControls) return;
  const duration = liveDurationSeconds();
  const useFinalReview = shouldUseFinalVideoForReview();
  let current = 0;
  if (liveReviewMode && mseLive?.mode === "review" && Number.isFinite(player.currentTime)) {
    current = player.currentTime;
    livePlayIndex = Math.max(
      0,
      Math.min(liveSegments.length - 1, Math.floor(player.currentTime)),
    );
  } else if (liveReviewMode && useFinalReview && Number.isFinite(player.currentTime)) {
    current = player.currentTime;
  } else if (liveReviewMode && Number.isFinite(player.currentTime)) {
    const segDuration = Number.isFinite(player.duration) && player.duration > 0
      ? player.duration
      : 1;
    const segProgress = Math.min(1, Math.max(0, player.currentTime / segDuration));
    current = Math.max(0, livePlayIndex) + segProgress;
  } else if (mseLive?.active && liveStarted && Number.isFinite(player.currentTime)) {
    const baseIndex = Math.max(0, Number(mseLive.baseIndex) || 0);
    current = baseIndex + player.currentTime;
    livePlayIndex = Math.max(0, Math.min(liveSegments.length - 1, baseIndex + Math.floor(player.currentTime)));
  } else if (liveStarted) {
    const segProgress = Number.isFinite(player.duration) && player.duration > 0
      ? Math.min(1, Math.max(0, player.currentTime / player.duration))
      : 0;
    current = Math.max(0, livePlayIndex) + segProgress;
  }
  liveProgress.max = String(Math.max(duration, 0.01));
  liveProgress.value = String(Math.min(Math.max(current, 0), Math.max(duration, 0.01)));
  liveProgress.disabled = !liveReviewMode || (!useFinalReview && liveSegments.length === 0);
  liveReplayBtn.disabled = !liveFinalUrl && liveSegments.length === 0;
  liveReturnBtn.disabled = !liveReviewMode || liveStreamComplete;
  livePlayPauseBtn.textContent = t(player.paused ? "playback.play" : "playback.pause");
  liveClock.textContent = liveReviewMode
    ? `${formatClock(current)} / ${formatClock(duration)}`
    : `${formatClock(current)} / LIVE`;
}

function resetLiveStage(clearPlayer = true) {
  resetMseLive();
  videos = [];
  liveSegments = [];
  liveSeen = new Set();
  livePlayIndex = -1;
  liveFinalUrl = "";
  currentJobLiveStartIndex = 0;
  liveStarted = false;
  liveEnded = false;
  liveStreamComplete = true;
  liveReviewMode = false;
  liveOverlapMode = false;
  liveMutedAutoplay = false;
  liveAutoplayBlocked = false;
  if (clearPlayer) {
    stopLiveCanvasLoop();
    player = primaryPlayer;
    standbyPlayer = secondaryPlayer;
    primaryPlayer.classList.add("active");
    primaryPlayer.classList.remove("standby");
    secondaryPlayer.classList.add("standby");
    secondaryPlayer.classList.remove("active");
    primaryPlayer.muted = false;
    secondaryPlayer.muted = true;
    primaryPlayer.loop = false;
    secondaryPlayer.loop = false;
    primaryPlayer.controls = false;
    secondaryPlayer.controls = false;
    clearMedia(primaryPlayer);
    clearMedia(secondaryPlayer);
    clearLiveCanvas();
    emptyStage.classList.remove("hidden");
    liveOverlay.classList.add("hidden");
  }
  renderLiveStatus();
}

function freezeCurrentLiveFrame() {
  if (!liveCanvas || !liveCanvasCtx || !player) return false;
  if (player.readyState < 2 || !player.videoWidth || !player.videoHeight) {
    return liveCanvas.classList.contains("has-frame");
  }
  if (liveCanvas.width !== player.videoWidth || liveCanvas.height !== player.videoHeight) {
    liveCanvas.width = player.videoWidth;
    liveCanvas.height = player.videoHeight;
  }
  liveCanvasCtx.drawImage(player, 0, 0, liveCanvas.width, liveCanvas.height);
  liveCanvas.classList.add("has-frame");
  return true;
}

function prepareLiveNewRootStage() {
  resetMseLive();
  videos = [];
  liveSegments = [];
  liveSeen = new Set();
  livePlayIndex = -1;
  liveFinalUrl = "";
  currentJobLiveStartIndex = 0;
  liveStarted = false;
  liveEnded = false;
  liveStreamComplete = false;
  liveReviewMode = false;
  liveOverlapMode = false;
  liveMutedAutoplay = false;
  liveAutoplayBlocked = false;
  stopLiveCanvasLoop();
  player = primaryPlayer;
  standbyPlayer = secondaryPlayer;
  primaryPlayer.classList.add("active");
  primaryPlayer.classList.remove("standby");
  secondaryPlayer.classList.add("standby");
  secondaryPlayer.classList.remove("active");
  primaryPlayer.muted = false;
  secondaryPlayer.muted = true;
  primaryPlayer.loop = false;
  secondaryPlayer.loop = false;
  primaryPlayer.controls = false;
  secondaryPlayer.controls = false;
  clearMedia(primaryPlayer);
  clearMedia(secondaryPlayer);
  clearLiveCanvas();
  emptyStage.classList.remove("hidden");
  liveOverlay.classList.remove("hidden");
  liveOverlayText.textContent = t("live.newScene");
  chunkMetric.textContent = t("live.waitingConnect");
  updateLiveControls();
}

function prepareLiveIdleBridgeStage() {
  const demo = activeDemoIndex >= 0 ? demos[activeDemoIndex] : null;
  const bridgeUrl = demo?.video_url || mediaSrc(demoPlayer) || mediaSrc(player);
  resetMseLive();
  videos = [];
  liveSegments = [];
  liveSeen = new Set();
  livePlayIndex = -1;
  liveFinalUrl = "";
  currentJobLiveStartIndex = 0;
  liveStarted = false;
  liveEnded = false;
  liveStreamComplete = false;
  liveReviewMode = false;
  liveOverlapMode = Boolean(bridgeUrl);
  liveMutedAutoplay = false;
  liveAutoplayBlocked = false;
  stopLiveCanvasLoop();
  player = primaryPlayer;
  standbyPlayer = secondaryPlayer;
  primaryPlayer.classList.add("active");
  primaryPlayer.classList.remove("standby");
  secondaryPlayer.classList.add("standby");
  secondaryPlayer.classList.remove("active");
  primaryPlayer.controls = false;
  secondaryPlayer.controls = false;
  primaryPlayer.loop = Boolean(bridgeUrl);
  secondaryPlayer.loop = false;
  clearMedia(secondaryPlayer);
  if (bridgeUrl) {
    primaryPlayer.muted = true;
    if (!mediaSrc(primaryPlayer).endsWith(bridgeUrl)) {
      primaryPlayer.src = bridgeUrl;
      primaryPlayer.load();
    }
    const demoTime = Number.isFinite(demoPlayer?.currentTime) ? demoPlayer.currentTime : 0;
    primaryPlayer.onloadedmetadata = () => {
      primaryPlayer.onloadedmetadata = null;
      if (Number.isFinite(primaryPlayer.duration) && primaryPlayer.duration > 0) {
        primaryPlayer.currentTime = Math.min(Math.max(0, demoTime), Math.max(0, primaryPlayer.duration - 0.1));
      }
      primaryPlayer.play().catch(() => {});
    };
    primaryPlayer.play().catch(() => {});
    emptyStage.classList.add("hidden");
    liveCanvas?.classList.add("has-frame");
    liveOverlay.classList.remove("hidden");
    liveOverlayText.textContent = t("live.idleBridge");
    chunkMetric.textContent = t("live.idle");
    startLiveCanvasLoop();
  } else {
    clearMedia(primaryPlayer);
    clearLiveCanvas();
    emptyStage.classList.remove("hidden");
    liveOverlay.classList.remove("hidden");
    liveOverlayText.textContent = t("live.connecting");
    chunkMetric.textContent = t("phase.uploading");
  }
  updateLiveControls();
}

function prepareLiveContinuationStage() {
  const hasFrozenFrame = freezeCurrentLiveFrame();
  resetMseLive();
  currentJobLiveStartIndex = liveSegments.length;
  livePlayIndex = currentJobLiveStartIndex - 1;
  liveFinalUrl = "";
  liveStarted = false;
  liveEnded = false;
  liveStreamComplete = false;
  liveReviewMode = false;
  liveOverlapMode = false;
  liveAutoplayBlocked = false;
  stopLiveCanvasLoop();
  player = primaryPlayer;
  standbyPlayer = secondaryPlayer;
  primaryPlayer.classList.add("active");
  primaryPlayer.classList.remove("standby");
  secondaryPlayer.classList.add("standby");
  secondaryPlayer.classList.remove("active");
  primaryPlayer.controls = false;
  secondaryPlayer.controls = false;
  primaryPlayer.loop = false;
  secondaryPlayer.loop = false;
  clearMedia(primaryPlayer);
  clearMedia(secondaryPlayer);
  if (hasFrozenFrame) {
    emptyStage.classList.add("hidden");
    liveCanvas?.classList.add("has-frame");
  } else {
    emptyStage.classList.remove("hidden");
    clearLiveCanvas();
  }
  liveOverlay.classList.remove("hidden");
  liveOverlayText.textContent = hasFrozenFrame
    ? t("live.continuationFrame")
    : t("live.continuation");
  chunkMetric.textContent = t("live.continuing");
  updateLiveControls();
}

function renderLiveStatus() {
  const total = liveSegments.length;
  const currentIndex = mseLive?.active && Number.isFinite(player?.currentTime)
    ? (Math.max(0, Number(mseLive.baseIndex) || 0) + Math.floor(player.currentTime))
    : livePlayIndex;
  const buffered = Math.max(0, total - Math.max(0, currentIndex + 1));
  const currentBuffered = Math.max(0, total - currentJobLiveStartIndex);
  let label = t("live.waiting");
  if (liveReviewMode) {
    label = shouldUseFinalVideoForReview() ? t("live.reviewFull") : t("live.reviewBuffer");
  } else if (!liveStarted && total > 0) {
    const startBufferSegments = liveStartBufferSegments();
    label = currentBuffered >= startBufferSegments
      ? t("live.ready", { current: currentBuffered, total })
      : t("live.buffering", { current: currentBuffered, total });
  } else if (liveStarted && !liveEnded) {
    const waiting = player.ended && buffered === 0;
    label = waiting
      ? t("live.waitingFrame", { total })
      : buffered > 0
        ? t("live.preloaded", { total, buffered })
        : t("live.waitingMore", { total });
    if (liveAutoplayBlocked) {
      label = t("live.clickStart", { total });
    } else if (liveMutedAutoplay) {
      label = t("live.clickSound", { total });
    }
  } else if (liveEnded) {
    label = t("live.ended", { total });
  } else if (liveFinalUrl) {
    label = t("live.finalReady");
  }
  chunkMetric.textContent = liveReviewMode
    ? t("live.review")
    : liveEnded ? t("live.complete", { total }) : t("live.added", { total });
  liveOverlayText.textContent = label;
  liveOverlay.classList.toggle("hidden", !liveStarted && !liveEnded && total === 0 && !liveFinalUrl);

  const maxBars = Math.min(12, Math.max(1, total || 1));
  const bars = Array.from({ length: maxBars }, (_, index) => {
    const ratioIndex = total > maxBars ? Math.floor(index * total / maxBars) : index;
    const state = ratioIndex < currentIndex ? " done" : ratioIndex === currentIndex ? " active" : "";
    return `<span class="live-rail-bar${state}"></span>`;
  }).join("");
  playbackRail.innerHTML = `
    <div class="live-rail">
      <span class="live-rail-text">${label}</span>
      <span class="live-rail-bars">${bars}</span>
    </div>
  `;
  updateLiveControls();
}

function requestLivePlay() {
  if (!player) return;
  const playPromise = player.play();
  if (!playPromise || typeof playPromise.catch !== "function") return;
  playPromise
    .then(() => {
      liveAutoplayBlocked = false;
      renderLiveStatus();
    })
    .catch(() => {
      liveAutoplayBlocked = true;
      if (!player.muted) {
        player.muted = true;
        const mutedPromise = player.play();
        if (mutedPromise && typeof mutedPromise.then === "function") {
          mutedPromise
            .then(() => {
              liveAutoplayBlocked = false;
              liveMutedAutoplay = true;
              renderLiveStatus();
            })
            .catch(() => {
              renderLiveStatus();
            });
        } else {
          liveMutedAutoplay = true;
          liveAutoplayBlocked = false;
          renderLiveStatus();
        }
      } else {
        renderLiveStatus();
      }
    });
}

function seekActivePlayerTo(timeSeconds = 0) {
  const targetTime = Math.max(0, Number(timeSeconds) || 0);
  const apply = () => {
    try {
      const duration = Number.isFinite(player.duration) && player.duration > 0
        ? Math.max(0, player.duration - 0.05)
        : targetTime;
      player.currentTime = Math.min(targetTime, duration);
    } catch {}
    updateLiveControls();
  };
  if (player.readyState >= 1) {
    apply();
  } else {
    player.addEventListener("loadedmetadata", apply, { once: true });
  }
}

function returnToLiveEdge() {
  if (!liveReviewMode || liveStreamComplete) return false;
  if (mseLive?.mode === "review") {
    const buffered = player.buffered;
    const liveEdge = buffered.length
      ? buffered.end(buffered.length - 1)
      : (Number.isFinite(player.duration) ? player.duration : 0);
    mseLive.mode = "live";
    mseLive.autoplay = true;
    mseLive.done = false;
    mseLive.baseIndex = 0;
    mseLive.pendingSeek = 0;
    liveReviewMode = false;
    liveStarted = true;
    liveEnded = false;
    seekActivePlayerTo(Math.max(0, liveEdge - 0.05));
    requestLivePlay();
    renderLiveStatus();
    return true;
  }
  player.pause();
  liveReviewMode = false;
  return playLiveIndex(Math.max(0, liveSegments.length - 1));
}

function playLiveIndex(index, options = {}) {
  if (index < 0 || index >= liveSegments.length) return false;
  const review = Boolean(options.review);
  const autoplay = options.autoplay !== false;
  const startTime = Math.max(0, Number(options.startTime) || 0);
  const url = liveSegments[index];
  liveReviewMode = review;
  livePlayIndex = index;
  liveStarted = true;
  liveEnded = false;
  let target = player;
  if (standbyPlayer && mediaSrc(standbyPlayer).endsWith(url)) {
    target = standbyPlayer;
  } else if (!mediaSrc(player).endsWith(url)) {
    player.src = url;
    player.load();
  }
  if (target !== player) {
    setActivePlayer(target);
  }
  player.muted = liveMutedAutoplay;
  player.controls = false;
  emptyStage.classList.add("hidden");
  startLiveCanvasLoop();
  if (startTime > 0) {
    seekActivePlayerTo(startTime);
  }
  if (autoplay) {
    requestLivePlay();
  }
  if (!review) {
    prepareStandby(index + 1);
  } else {
    clearMedia(standbyPlayer);
  }
  renderLiveStatus();
  return true;
}

function playNextLiveSegment(options = {}) {
  const next = livePlayIndex + 1;
  if (next < liveSegments.length) {
    return playLiveIndex(next, options);
  }
  if (liveStarted) {
    renderLiveStatus();
  } else if (liveFinalUrl) {
    enterReviewMode(false);
  }
  return false;
}

function enterReviewMode(autoplay = false, startTime = 0) {
  if (!liveFinalUrl && !liveSegments.length) return false;
  const reviewStart = Math.max(0, Number(startTime) || 0);
  if (!shouldUseFinalVideoForReview() && startMseReviewPlayback(autoplay, reviewStart)) {
    return true;
  }
  resetMseLive();
  liveReviewMode = true;
  liveStarted = true;
  liveEnded = false;
  if (shouldUseFinalVideoForReview()) {
    if (!mediaSrc(player).endsWith(liveFinalUrl)) {
      player.src = liveFinalUrl;
      player.load();
    }
  } else {
    const index = Math.max(
      0,
      Math.min(liveSegments.length - 1, Math.floor(reviewStart)),
    );
    livePlayIndex = index - 1;
    return playNextLiveSegment({
      review: true,
      autoplay,
      startTime: reviewStart - index,
    });
  }
  standbyPlayer.pause();
  player.controls = false;
  emptyStage.classList.add("hidden");
  startLiveCanvasLoop();
  if (autoplay) {
    player.currentTime = reviewStart;
    requestLivePlay();
  } else if (reviewStart > 0) {
    seekActivePlayerTo(reviewStart);
  }
  renderLiveStatus();
  return true;
}

function addVideo(url, taskId = currentJobId) {
  if (!url || videos.includes(url)) return;
  videos.push(url);
  recordVideoForTask(url, taskId);

  if (!isPreviewVideo(url)) {
    liveFinalUrl = url;
    if (liveStarted && player.ended) {
      liveEnded = true;
    }
    if (liveEnded || (!liveSegments.length && !liveStarted)) {
      enterReviewMode(false);
    }
    renderLiveStatus();
    return;
  }

  if (!liveSeen.has(url)) {
    liveSeen.add(url);
    liveSegments.push(url);
  }
  if (appendMseLiveSegment(url)) {
    renderLiveStatus();
    return;
  }
  renderLiveStatus();
  if (!liveStarted) {
    prepareStandby(Math.max(0, currentJobLiveStartIndex));
    if (hasEnoughLiveBufferToStart()) {
      playNextLiveSegment();
    }
  } else if (player.ended) {
    playNextLiveSegment();
  } else if (livePlayIndex + 1 < liveSegments.length) {
    prepareStandby(livePlayIndex + 1);
  }
}

function handlePlayerEnded(el) {
  if (el !== player) return;
  if (mseLive?.active) {
    liveEnded = true;
    renderLiveStatus();
    return;
  }
  if (liveReviewMode) {
    if (!shouldUseFinalVideoForReview() && livePlayIndex + 1 < liveSegments.length) {
      playNextLiveSegment({ review: true, autoplay: true });
      return;
    }
    liveEnded = true;
    renderLiveStatus();
    return;
  }
  if (!playNextLiveSegment()) {
    liveEnded = Boolean(liveStarted);
    if (shouldUseFinalVideoForReview()) {
      enterReviewMode(false);
    } else {
      renderLiveStatus();
    }
  }
}

[primaryPlayer, secondaryPlayer].forEach((el) => {
  el.addEventListener("ended", () => handlePlayerEnded(el));
  el.addEventListener("timeupdate", updateLiveControls);
  el.addEventListener("loadedmetadata", updateLiveControls);
  el.addEventListener("canplay", updateLiveControls);
});

videoFrame.addEventListener("click", () => {
  if (!liveStarted && liveSegments.length) {
    if (hasEnoughLiveBufferToStart()) {
      playNextLiveSegment();
    } else {
      renderLiveStatus();
    }
    return;
  }
  if (!liveStarted) return;
  if (liveMutedAutoplay || player.muted) {
    player.muted = false;
    liveMutedAutoplay = false;
  }
  liveAutoplayBlocked = false;
  requestLivePlay();
  renderLiveStatus();
});

livePlayPauseBtn.addEventListener("click", () => {
  if (!liveStarted && liveSegments.length) {
    if (hasEnoughLiveBufferToStart()) {
      playNextLiveSegment();
    } else {
      renderLiveStatus();
    }
    return;
  }
  if (!liveStarted) return;
  if (player.paused) {
    requestLivePlay();
  } else {
    player.pause();
    updateLiveControls();
  }
});

liveReplayBtn.addEventListener("click", () => {
  if (liveFinalUrl) {
    enterReviewMode(true);
    return;
  }
  if (liveSegments.length) {
    enterReviewMode(true);
  }
});

liveReturnBtn.addEventListener("click", () => {
  returnToLiveEdge();
});

liveProgress.addEventListener("input", () => {
  if (!liveReviewMode) return;
  const value = Number(liveProgress.value);
  if (!Number.isFinite(value)) return;
  if (shouldUseFinalVideoForReview() || mseLive?.mode === "review") {
    player.currentTime = value;
    updateLiveControls();
    return;
  }
  if (!liveSegments.length) return;
  const wasPlaying = !player.paused;
  const clamped = Math.min(Math.max(value, 0), Math.max(0, liveSegments.length - 0.001));
  const index = Math.max(0, Math.min(liveSegments.length - 1, Math.floor(clamped)));
  const offset = Math.max(0, clamped - index);
  if (index === livePlayIndex && mediaSrc(player).endsWith(liveSegments[index])) {
    seekActivePlayerTo(offset);
    if (wasPlaying) requestLivePlay();
  } else {
    playLiveIndex(index, { review: true, autoplay: wasPlaying, startTime: offset });
  }
});

function closeEvents() {
  if (eventRecoveryTimer) {
    clearTimeout(eventRecoveryTimer);
    eventRecoveryTimer = null;
  }
  eventRecoveryFailures = 0;
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function scheduleEventRecovery(taskId, delayMs = 750) {
  if (!taskId || taskId !== currentJobId || eventRecoveryTimer) return;
  eventRecoveryTimer = setTimeout(() => {
    eventRecoveryTimer = null;
    recoverJobAfterEventLoss(taskId);
  }, Math.max(0, delayMs));
}

async function recoverJobAfterEventLoss(taskId) {
  if (!taskId || taskId !== currentJobId) return;
  try {
    const res = await fetch(`/api/jobs/${taskId}`);
    if (res.status === 404 || res.status === 410) {
      upsertTurn(taskId, { status: "expired" });
      liveStreamComplete = true;
      finishMseLiveStream();
      closeEvents();
      currentJobId = null;
      setBusy(false);
      setPhase("expired");
      return;
    }
    if (!res.ok) throw new Error(`job recovery ${res.status}`);
    const data = await res.json();
    (data.videos || [])
      .slice()
      .sort((a, b) => videoSortKey(a).localeCompare(videoSortKey(b)))
      .forEach((url) => addVideo(url, taskId));
    const status = String(data.status || data.phase || "running");
    const phase = String(data.phase || status);
    upsertTurn(taskId, { status, phase });
    eventRecoveryFailures = 0;
    if (["succeeded", "failed", "canceled"].includes(status)) {
      finishMseLiveStream();
      persistActiveSession({ mirrorToHistory: true });
      setPhase(status);
      if (status === "failed") {
        showUserFacingError(t("error.interruptedRetry"));
      }
      closeEvents();
      currentJobId = null;
      setBusy(false);
      if (status === "succeeded") refreshDemos();
      return;
    }
    scheduleEventRecovery(taskId, 1000);
  } catch (err) {
    eventRecoveryFailures += 1;
    if (eventRecoveryFailures === 1 || eventRecoveryFailures % 5 === 0) {
      log(`SSE recovery retry ${eventRecoveryFailures}: ${err.message || err}`);
    }
    scheduleEventRecovery(taskId, Math.min(5000, 750 * eventRecoveryFailures));
  }
}

function connectEvents(taskId) {
  closeEvents();
  eventSource = new EventSource(`/api/jobs/${taskId}/events`);
  eventSource.onmessage = (message) => {
    if (eventRecoveryTimer) {
      clearTimeout(eventRecoveryTimer);
      eventRecoveryTimer = null;
    }
    eventRecoveryFailures = 0;
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    if (event.event === "llm_reply") {
      setReplyContent(event.reply || "");
      upsertTurn(taskId, { reply: event.reply || "", status: "prompt_expanding" });
      setPhase("prompt_expanding");
    } else if (event.event === "prompt_ready") {
      renderSegments(event.segments || []);
      upsertTurn(taskId, { segments: event.segments || [], status: "queued_on_gpu" });
      setPhase("queued_on_gpu");
    } else if (event.event === "stage") {
      upsertTurn(taskId, { phase: event.phase || "", status: event.phase || "" });
      setPhase(event.phase);
    } else if (event.event === "runner_log") {
      setPhase(event.phase);
      log(event.line);
    } else if (event.event === "asset") {
      if (event.kind === "image") {
        showGeneratedPreviewFrame(event.url);
      } else {
        addVideo(event.url, event.task_id || taskId);
      }
      setPhase("streaming");
    } else if (event.event === "continuation_state") {
      log(`continuation state updated: ${event.conversation_id || conversationId}`);
      if (event.early) {
        setBusy(false);
        setPhase("streaming");
      }
    } else if (event.event === "done") {
      finishMseLiveStream();
      upsertTurn(taskId, { status: "succeeded" });
      persistActiveSession({ mirrorToHistory: true });
      setPhase("succeeded");
      setBusy(false);
      refreshVideos(taskId);
      refreshDemos();
    } else if (event.event === "error") {
      finishMseLiveStream();
      upsertTurn(taskId, { status: "failed" });
      persistActiveSession({ mirrorToHistory: true });
      setServiceStatus("status.error");
      setPhase("failed");
      setBusy(false);
      showUserFacingError(t("error.interruptedRestart"));
      log(`ERROR: ${event.message}`);
    } else if (event.event === "closed") {
      closeEvents();
      if (currentJobId === taskId) currentJobId = null;
      setBusy(false);
    }
  };
  eventSource.onerror = () => {
    if (!eventRecoveryTimer) {
      log("SSE disconnected; recovering job state in the background.");
    }
    scheduleEventRecovery(taskId);
  };
}

async function refreshVideos(taskId) {
  const res = await fetch(`/api/jobs/${taskId}/videos`);
  if (res.status === 404 || res.status === 410) {
    if (currentJobId === taskId) {
      closeEvents();
      currentJobId = null;
      setBusy(false);
      if (interactionBusy || currentJobId) {
        setServiceStatus(serviceReady ? "status.ready" : "status.starting");
      }
    }
    upsertTurn(taskId, { status: "expired" });
    return false;
  }
  if (!res.ok) return false;
  const data = await res.json();
  (data.videos || [])
    .slice()
    .sort((a, b) => videoSortKey(a).localeCompare(videoSortKey(b)))
    .forEach((url) => addVideo(url, taskId));
  return true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!serviceReady) {
    setServiceStatus("status.starting");
    log("Service is still starting. Wait until the status is READY.");
    return;
  }
  const scene = sceneDescription.value.trim();
  const text = userText.value.trim();
  if (!scene || !text) return;
  const refineScene = sceneNeedsRefinement(scene);
  const nextVisualRootKey = visualRootKeyFor(
    scene,
    selectedAspect,
    selectedTemplateId,
  );
  const startsNewVisualRoot = nextVisualRootKey !== visualRootKey;
  const continueConversation = liveSegments.length > 0 && !startsNewVisualRoot;
  closeEvents();
  currentJobId = null;
  if (startsNewVisualRoot) {
    saveSessionToHistory(activeSession);
    conversationId = newConversationId();
    localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
    rememberVisualRoot(nextVisualRootKey);
    activeSession = createSession(conversationId);
  } else {
    activeSession = activeSession || createSession(conversationId);
    activeSession.scene = scene;
    activeSession.templateId = selectedTemplateId;
    activeSession.aspect = selectedAspect;
    activeSession.visualRootKey = nextVisualRootKey;
    activeSession.mode = requestedMode;
    persistActiveSession();
  }
  if (!continueConversation) {
    prepareLiveNewRootStage();
    logBox.textContent = "";
  } else {
    prepareLiveContinuationStage();
    log(`continue conversation ${conversationId}`);
  }
  setReplyState("reply.generating");
  segmentList.innerHTML = "";
  setBusy(true);
  setPhase("uploading");

  try {
    const payload = new FormData();
    payload.append("scene_description", scene);
    payload.append("user_text", text);
    payload.append("mode", requestedMode);
    payload.append("aspect_ratio", selectedAspect);
    payload.append("template_id", selectedTemplateId);
    payload.append("refine_scene", String(refineScene));
    payload.append("conversation_id", conversationId);

    const res = await fetch("/api/jobs", { method: "POST", body: payload });
    if (!res.ok) {
      let message = await res.text();
      try {
        const parsed = JSON.parse(message);
        message = parsed.detail || message;
      } catch {}
      throw new Error(message);
    }
    const data = await res.json();
    currentJobId = data.task_id;
    setBusy(true);
    upsertTurn(currentJobId, {
      userText: text,
      status: data.status || data.phase || "accepted",
      createdAt: Date.now(),
    });
    modeMetric.textContent = `${modeLabel(data.mode || requestedMode)} · ${data.video_width || getAspectPreset().width}x${data.video_height || getAspectPreset().height}`;
    setPhase(data.phase);
    log(`accepted ${currentJobId}${data.is_continuation ? " · continuation" : " · new root"} · conversation=${data.conversation_id || conversationId}`);
    connectEvents(currentJobId);
  } catch (err) {
    finishMseLiveStream();
    setBusy(false);
    setServiceStatus("status.error");
    setPhase("failed");
    showUserFacingError(t("error.submitFailed"));
    log(`submit failed: ${err.message || err}`);
  }
});

previewBtn.addEventListener("click", previewSegments);
if (demoLiveBtn) {
  demoLiveBtn.addEventListener("click", startDemoLiveReference);
}
if (demoDrawer) {
  demoDrawer.addEventListener("toggle", () => {
    updateDemoDrawerAction();
    if (demoDrawer.open && activeDemoIndex >= 0) {
      demoPlayer.play().catch(() => {});
      if (!demoRaf) demoRaf = requestAnimationFrame(updateDemoOverlay);
      return;
    }
    demoPlayer.pause();
    if (demoRaf) cancelAnimationFrame(demoRaf);
    demoRaf = null;
  });
}
clearTemplateBtn.addEventListener("click", () => selectTemplate("", false));
sceneDescription.addEventListener("input", syncTemplateSelectionToScene);
resetSessionBtn.addEventListener("click", () => startNewConversation(true));
languageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    applyUiLanguage(button.dataset.uiLanguage || "en");
  });
});

cancelBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  await fetch(`/api/jobs/${currentJobId}`, { method: "DELETE" }).catch(() => {});
  closeEvents();
  finishMseLiveStream();
  upsertTurn(currentJobId, { status: "canceled" });
  persistActiveSession({ mirrorToHistory: true });
  setPhase("canceled");
  currentJobId = null;
  setBusy(false);
});

window.addEventListener("beforeunload", () => {
  persistActiveSession({ mirrorToHistory: true });
});

loadStoredSessions();
const startupPreferences = beginFreshPageSession();
renderTemplates();
if (startupPreferences.templateId) {
  selectTemplate(startupPreferences.templateId, !startupPreferences.scene);
} else if (!startupPreferences.scene) {
  selectTemplate(DEFAULT_TEMPLATE_ID, true);
} else {
  selectTemplate("", false);
}
if (startupPreferences.scene) {
  sceneDescription.value = startupPreferences.scene;
  syncTemplateSelectionToScene();
}
setAspect(startupPreferences.aspect || "landscape");
resetLiveStage(true);
persistActiveSession();
modeMetric.textContent = `${modeLabel(requestedMode)} · ${getAspectPreset().width}x${getAspectPreset().height}`;
applyUiLanguage(uiLanguage, { persist: false });
refreshDemos();
setInterval(refreshDemos, 60000);

async function refreshReadiness() {
  try {
    const res = await fetch("/readyz", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    serviceReady = Boolean(data.ready);
    const missing = Array.isArray(data.missing) ? data.missing : [];
    const logKey = serviceReady ? "ready" : `starting:${missing.join(",")}`;
    if (logKey !== readinessLogKey) {
      log(serviceReady
        ? "TaoMate service is ready."
        : `Service is starting: ${missing.join(", ") || "dependencies"}.`);
      readinessLogKey = logKey;
    }
    if (!currentJobId && !interactionBusy) {
      setServiceStatus(serviceReady ? "status.ready" : "status.starting");
    }
  } catch {
    serviceReady = false;
    if (readinessLogKey !== "offline") {
      log("TaoMate service is offline.");
      readinessLogKey = "offline";
    }
    if (!currentJobId && !interactionBusy) setServiceStatus("status.offline");
  }
  form.querySelector("button[type='submit']").disabled = interactionBusy || !serviceReady;
  previewBtn.disabled = interactionBusy || !serviceReady;
}

setBusy(false);
refreshReadiness();
setInterval(refreshReadiness, 2000);

async function refreshSystem() {
  try {
    const res = await fetch("/api/system");
    if (!res.ok) return;
    const data = await res.json();
    const gpu = data.gpu || {};
    if (!gpu.available) {
      systemBlocked = false;
      gpuMetric.textContent = t("gpu.unknown");
      setSystemNotice(gpu.error ? t("gpu.unavailable", { error: gpu.error }) : "", "warning");
      return;
    }
    const busy = (gpu.processes || []).filter((proc) => proc.sm > 10 || proc.mem > 10);
    systemBlocked = false;
    gpuMetric.textContent = busy.length ? t("gpu.online") : t("gpu.ready");
    if (!currentJobId && !interactionBusy) {
      setServiceStatus(serviceReady ? "status.ready" : "status.starting");
    }
    setSystemNotice("");
    if (busy.length) {
      const summary = `${busy.length} active process(es)`;
      if (summary !== lastGpuSummary) {
        lastGpuSummary = summary;
        log(`compute busy: ${summary}`);
      }
    } else {
      lastGpuSummary = "";
    }
  } catch {
    gpuMetric.textContent = t("gpu.unknown");
  }
}

refreshSystem();
setInterval(refreshSystem, 30000);
