"""
MindBridge 心橋 — 精神疾病語言共感 Agent（終端機雛形）
執行：python main.py
"""
import llm
from agents import (DISORDERS, HELP, CoachAgent, KnowledgeAgent, PerspectiveAgent,
                    PersonaAgent, ReverseAgent, RouterAgent, SafetyGuard)
from rag import KnowledgeBase

SCENARIOS = {
    "1": "期末報告分組，你這週一直沒回群組訊息，組員私訊你",
    "2": "室友發現你這幾天幾乎沒出房間，敲門關心你",
}


def pick_disorder() -> str:
    keys = list(DISORDERS)
    print("  " + "  ".join(f"{i + 1}.{DISORDERS[k]}" for i, k in enumerate(keys)))
    s = input("  選擇角色 > ").strip()
    return keys[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(keys) else "depression"


def roleplay(kb, guard):
    d = pick_disorder()
    print("  情境：" + "  ".join(f"{k}.{v}" for k, v in SCENARIOS.items()))
    scenario = SCENARIOS.get(input("  選擇情境 > ").strip(), SCENARIOS["1"])
    persona, coach = PersonaAgent(kb, d, scenario), CoachAgent(kb)
    print(f"\n🎭 演練開始（{DISORDERS[d]}｜{scenario}）輸入 q 結束\n")
    last = persona.reply("（你主動開口關心對方）")
    print(f"角色：{last}")
    while (msg := input("你  ：").strip()) not in ("q", ""):
        if (alert := guard.check(msg)):
            print(alert); break
        print(coach.evaluate(last, msg, d))
        last = persona.reply(msg)
        print(f"角色：{last}")
    print(coach.report())


def main():
    kb, guard = KnowledgeBase(), SafetyGuard()
    agents = {"knowledge": KnowledgeAgent(kb), "perspective": PerspectiveAgent(kb),
              "reverse": ReverseAgent(kb)}
    router = RouterAgent()
    print("=" * 56)
    print(" MindBridge 心橋｜Bridge the words, reach the mind")
    print(f" 知識庫：{len(kb.chunks)} 個段落｜LLM：{'示範模式' if llm.MOCK else llm.MODEL}")
    print("=" * 56)
    while True:
        print("\n1.知識問答  2.視角轉換  3.反向翻譯  4.對話演練  5.自由提問(自動分派)  0.離開")
        mode = input("> ").strip()
        if mode == "0":
            break
        if mode == "4":
            roleplay(kb, guard); continue
        text = input("  輸入內容 > ").strip()
        if not text:
            continue
        if (alert := guard.check(text)):
            print(alert); continue
        if mode == "5":
            intent = router.route(text)
            print(f"  🔀 Router：交給 {intent} agent")
            if intent == "roleplay":
                roleplay(kb, guard); continue
            mode = {"knowledge": "1", "perspective": "2", "reverse": "3"}.get(intent, "1")
        if mode == "1":
            print(agents["knowledge"].run(text))
        elif mode in ("2", "3"):
            d = pick_disorder()
            agent = agents["perspective" if mode == "2" else "reverse"]
            print(agent.run(text, d))
    print("謝謝使用。\n" + HELP)


if __name__ == "__main__":
    main()
