import argparse

from .orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Diabetes multi-agent assistant (educational demo)")
    parser.add_argument("--trace", action="store_true", help="show which agent did what")
    args = parser.parse_args()

    bot = Orchestrator()
    mode = "DEMO (offline fake LLM)" if type(bot.llm).__name__ == "FakeLLM" else "LIVE"
    print(f"Diabetes assistant [{mode}] - educational only, not medical advice. Type 'quit' to exit.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if not msg:
            continue
        r = bot.handle("cli", msg)
        print(f"\nBot: {r.answer}")
        if r.sources:
            print("Sources: " + "; ".join(r.sources))
        if r.ticket:
            print(f"[Escalated: ticket {r.ticket.id}, urgency={r.ticket.urgency}]")
        if args.trace:
            for e in r.trace:
                print(f"  - {e.agent:<10} {e.action:<13} {e.detail}")
        print()


if __name__ == "__main__":
    main()
