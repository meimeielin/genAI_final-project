"""
MindBridge 的 Agent 們
  SafetyGuard   安全守門：偵測危機訊號，優先提供求助資源並中止模擬
  RouterAgent   意圖判斷：決定交給哪一個 Agent
  KnowledgeAgent  知識問答（RAG + 引用來源）
  PerspectiveAgent 視角轉換：一般話語 → 可能的扭曲解讀 + 認知扭曲分析
  ReverseAgent  反向翻譯：患者說的話 → 背後可能的感受 + 建議回應
  PersonaAgent  多回合對話演練中扮演角色
  CoachAgent    即時評分使用者的回應並給建議，結束時產生報告
"""
import llm
from rag import KnowledgeBase, format_context

DISORDERS = {
    "depression": "憂鬱症",
    "gad": "廣泛性焦慮症",
    "bipolar": "雙相情緒障礙症",
    "schizophrenia": "思覺失調症",
}

ETHICS = (
    "這是心理衛生教育用的共感模擬，目的是減少汙名、幫助一般人理解。"
    "規則：1) 不得誇大、獵奇或醜化患者；2) 不得描寫自傷、自殺方法或暴力細節；"
    "3) 內容需符合提供的參考資料，不足時說明不確定；4) 提醒每個人的經驗都不同。"
)

HELP = ("💛 如果你或身邊的人正感到很痛苦、有傷害自己的想法，請不要一個人承受：\n"
        "   1925 安心專線（24 小時）｜1995 生命線｜1980 張老師｜緊急狀況請撥 119")


class SafetyGuard:
    CRISIS = ["想死", "不想活", "自殺", "輕生", "結束生命", "活不下去", "傷害自己",
              "割腕", "跳樓", "消失就好", "去死"]

    def check(self, text: str) -> str | None:
        if any(k in text for k in self.CRISIS):
            return "⚠️ 偵測到可能與自身危機有關的內容，已暫停模擬。\n" + HELP
        return None


class RouterAgent:
    def route(self, text: str) -> str:
        try:
            r = llm.chat_json(
                "你是路由器，判斷使用者想做什麼，回傳 {\"intent\": ...}。"
                "knowledge=詢問精神疾病知識；perspective=想知道某句話在患者耳中聽起來如何；"
                "reverse=想理解患者說的話背後的意思或該怎麼回；roleplay=想練習對話。",
                text)
            return r.get("intent", "knowledge")
        except Exception:
            if any(k in text for k in ["練習", "演練", "對話"]):
                return "roleplay"
            if any(k in text for k in ["他說", "她說", "怎麼回", "什麼意思"]):
                return "reverse"
            if any(k in text for k in ["聽起來", "會怎麼想", "轉換"]):
                return "perspective"
            return "knowledge"


class BaseAgent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def retrieve(self, query, disorder=None, k=3):
        hits = self.kb.search(query, k=k, disorder=disorder)
        return hits, format_context(hits)

    @staticmethod
    def cites(hits) -> str:
        return "📚 依據：" + "；".join(f"[{i + 1}] {c.cite}" for i, (_, c) in enumerate(hits))


class KnowledgeAgent(BaseAgent):
    def run(self, question: str) -> str:
        hits, ctx = self.retrieve(question)
        try:
            ans = llm.chat(
                f"你是心理衛生知識助教。{ETHICS}只根據參考資料回答，並以 [編號] 標註出處；"
                "資料沒有的內容請說不知道，並建議查詢專業來源。",
                f"參考資料：\n{ctx}\n\n問題：{question}")
        except Exception:
            ans = "（示範模式：未設定 LLM，以下直接列出檢索到的段落）\n" + \
                  "\n".join(f"[{i + 1}] {c.text[:80]}…" for i, (_, c) in enumerate(hits))
        return f"{ans}\n{self.cites(hits)}"


class PerspectiveAgent(BaseAgent):
    def run(self, sentence: str, disorder: str) -> str:
        name = DISORDERS[disorder]
        hits, ctx = self.retrieve(f"{name} 語言接收的特徵 認知扭曲 {sentence}", disorder, k=4)
        try:
            r = llm.chat_json(
                f"你是共感模擬器。{ETHICS}根據參考資料，模擬處於{name}症狀影響下的人，"
                "可能如何「接收」這句話。回傳 JSON："
                '{"heard": "內心可能聽到的聲音（一句，用「」）", '
                '"distortion": "對應的認知扭曲或症狀名稱", '
                '"why": "用兩三句話解釋為什麼會這樣解讀，標註 [編號]", '
                '"better": "對方比較容易接收的說法"}',
                f"參考資料：\n{ctx}\n\n原本的話：「{sentence}」")
        except Exception:
            top = hits[0][1]
            r = {"heard": "（示範模式：設定 LLM 後會產生模擬的內心聲音）",
                 "distortion": "見參考資料",
                 "why": f"[1] {top.text[:90]}…",
                 "better": "（示範模式）"}
        return (f"🗣  你說的：「{sentence}」\n"
                f"👂 {name}視角可能聽到：{r['heard']}\n"
                f"🧠 背後機制：{r['distortion']}\n"
                f"💡 為什麼：{r['why']}\n"
                f"✅ 可以換成：{r['better']}\n"
                f"{self.cites(hits)}\n"
                "※ 模擬僅供理解，每位患者的經驗都不同。")


class ReverseAgent(BaseAgent):
    def run(self, sentence: str, disorder: str) -> str:
        name = DISORDERS[disorder]
        hits, ctx = self.retrieve(f"{name} 外在語言表現 溝通建議 {sentence}", disorder, k=4)
        try:
            r = llm.chat_json(
                f"你是溝通翻譯員。{ETHICS}有人（可能受{name}影響）說了一句話，"
                "請根據參考資料推測背後可能的感受與需要，並建議支持性回應。回傳 JSON："
                '{"feeling": "可能的感受", "need": "可能的需要", '
                '"avoid": "應避免的回應", "reply": "建議的回應（一兩句）", "why": "理由，標註 [編號]"}',
                f"參考資料：\n{ctx}\n\n對方說：「{sentence}」")
        except Exception:
            r = {"feeling": "（示範模式）", "need": "（示範模式）", "avoid": "（示範模式）",
                 "reply": "（示範模式）", "why": f"[1] {hits[0][1].text[:90]}…"}
        return (f"💬 對方說：「{sentence}」\n"
                f"❤️ 可能的感受：{r['feeling']}\n🙏 可能的需要：{r['need']}\n"
                f"🚫 避免說：{r['avoid']}\n✅ 可以回：{r['reply']}\n💡 理由：{r['why']}\n"
                f"{self.cites(hits)}")


class PersonaAgent(BaseAgent):
    def __init__(self, kb, disorder: str, scenario: str):
        super().__init__(kb)
        self.disorder, self.scenario, self.history = disorder, scenario, []
        _, self.ctx = self.retrieve(f"{DISORDERS[disorder]} 語言表現 語言接收", disorder, k=4)

    def reply(self, user_msg: str) -> str:
        self.history.append(("朋友", user_msg))
        convo = "\n".join(f"{who}：{m}" for who, m in self.history[-10:])
        try:
            msg = llm.chat(
                f"你在教育用對話演練中扮演一位受{DISORDERS[self.disorder]}影響的大學生。{ETHICS}"
                f"情境：{self.scenario}。依參考資料表現語言特徵，回應自然、簡短（1~2 句）。"
                "若對方的回應讓你感到被理解，可以慢慢放鬆；被否定時則更退縮或防衛。",
                f"參考資料：\n{self.ctx}\n\n對話紀錄：\n{convo}\n\n請以角色身分回覆：")
        except Exception:
            msg = "（示範模式）……嗯，沒事啦，不想麻煩你。"
        self.history.append(("角色", msg))
        return msg


class CoachAgent(BaseAgent):
    def __init__(self, kb):
        super().__init__(kb)
        self.scores = []
        _, self.ctx = self.retrieve("支持性溝通 傾聽 承認感受 溝通建議", k=4)

    def evaluate(self, persona_msg: str, user_msg: str, disorder: str) -> str:
        try:
            r = llm.chat_json(
                f"你是溝通教練，依參考資料評估使用者回應對受{DISORDERS[disorder]}影響者是否支持性。"
                '回傳 JSON：{"score": 1到5的整數, "good": "做得好的地方", "tip": "具體改進建議"}',
                f"參考資料：\n{self.ctx}\n\n角色剛說：「{persona_msg}」\n使用者回應：「{user_msg}」")
        except Exception:
            r = {"score": 3, "good": "（示範模式）", "tip": "（示範模式）先承認感受，再詢問需要。"}
        self.scores.append(int(r["score"]))
        return f"   🎯 教練：{'★' * int(r['score'])}{'☆' * (5 - int(r['score']))}｜👍 {r['good']}｜💡 {r['tip']}"

    def report(self) -> str:
        if not self.scores:
            return "沒有對話紀錄。"
        avg = sum(self.scores) / len(self.scores)
        trend = "進步中 📈" if len(self.scores) > 1 and self.scores[-1] > self.scores[0] else "持續練習 💪"
        return f"📊 演練報告：共 {len(self.scores)} 回合，平均 {avg:.1f} / 5 分，{trend}"
