import re
import sys

def parse_backtest(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    rows = {}
    for l in text.splitlines():
        m = re.match(
            r"\s*(\S[\w_ ]+?)\s{2,}(\d+)\s+([\d.]+)\s+([+\-][\d.]+)\s+([+\-][\d.]+)\s+([\d.]+)\s+([\d.]+)",
            l,
        )
        if m:
            name = m.group(1).strip()
            rows[name] = {
                "N": int(m.group(2)),
                "Win": m.group(3),
                "Mean": m.group(4),
                "CAGR": m.group(5),
                "Sharpe": m.group(6),
                "MaxDD": m.group(7),
            }
    um = re.search(r"Universe: ([\d,]+) examples", text)
    universe = um.group(1) if um else "?"
    return rows, universe


prev_path = sys.argv[1] if len(sys.argv) > 1 else "logs/backtest_new_output.txt"
curr_path = sys.argv[2] if len(sys.argv) > 2 else "logs/backtest_15k.txt"

prev, u_prev = parse_backtest(prev_path)
curr, u_curr = parse_backtest(curr_path)

strategies = [
    "fear_contrarian_long",
    "extreme_greed_long",
    "greed_long_no_wsb",
    "baseline_all_longs",
    "patent_long",
    "baseline_high_conviction_long",
    "patent_high_conviction_long",
    "baseline_top10pct_conviction",
]

print(f"\nUniverse:  {u_prev} examples  -->  {u_curr} examples\n")
print(f"{'Strategy':<35}  {'N (prev)':>9}  {'N (curr)':>9}  {'Sharpe prev':>11}  {'Sharpe now':>10}  {'CAGR prev':>9}  {'CAGR now':>8}  {'MaxDD prev':>10}  {'MaxDD now':>9}")
print("-" * 120)
for s in strategies:
    p = prev.get(s, {})
    c = curr.get(s, {})
    pn = str(p.get("N", "-"))
    cn = str(c.get("N", "-"))
    ps = p.get("Sharpe", "-")
    cs = c.get("Sharpe", "-")
    pc = p.get("CAGR", "-")
    cc = c.get("CAGR", "-")
    pd = p.get("MaxDD", "-")
    cd = c.get("MaxDD", "-")
    # flag changes
    try:
        diff = float(cs) - float(ps)
        flag = f"  {'▲' if diff > 0 else '▼'}{abs(diff):.2f}"
    except Exception:
        flag = ""
    print(f"{s:<35}  {pn:>9}  {cn:>9}  {ps:>11}  {cs:>10}  {pc:>9}  {cc:>8}  {pd:>10}  {cd:>9}{flag}")
