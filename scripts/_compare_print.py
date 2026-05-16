prev = {
    "fear_contrarian_long":         (1085,  2.56,  6.1,  1.6),
    "extreme_greed_long":           (1761,  1.80, 20.6, 37.3),
    "greed_long_no_wsb":            (4650,  1.59, 26.1, 49.2),
    "baseline_all_longs":           (11190, 1.41, 37.7, 67.0),
    "patent_long":                  (1352,  1.29, 41.4, 60.1),
    "baseline_high_conviction_long":(1788,  0.69, 19.0, 94.0),
    "patent_high_conviction_long":  (209,   0.53,  5.2, 27.3),
    "baseline_top10pct_conviction": (633,   0.35, -6.5, 95.9),
}
curr = {
    "fear_contrarian_long":         (1231,  2.66,  7.9,  2.9),
    "extreme_greed_long":           (2195,  2.31, 26.7, 36.0),
    "greed_long_no_wsb":            (5766,  1.85, 30.7, 45.5),
    "baseline_all_longs":           (14244, 1.85, 52.0, 61.5),
    "patent_long":                  (1701,  1.56, 55.1, 58.2),
    "baseline_high_conviction_long":(1869,  0.77, 23.7, 93.3),
    "patent_high_conviction_long":  (277,   1.53, 20.2, 23.0),
    "baseline_top10pct_conviction": (635,   0.35, -6.5, 95.9),
}

print("Dataset:  12,300 examples (197 A-tickers)  -->  15,859 examples (~A-AB tickers)\n")
print(f"{'Strategy':<35}  {'N (12K)':>8}  {'N (15K)':>8}  {'Sh 12K':>7}  {'Sh 15K':>7}  {'Delta':>7}  {'CAGR 12K':>9}  {'CAGR 15K':>9}  {'MaxDD 12K':>10}  {'MaxDD 15K':>10}")
print("-" * 130)
for s, (pn, ps, pc, pd) in prev.items():
    cn, cs, cc, cd = curr[s]
    ds = cs - ps
    arrow = ("+" if ds >= 0 else "") + f"{ds:.2f}"
    print(f"{s:<35}  {pn:>8,}  {cn:>8,}  {ps:>7.2f}  {cs:>7.2f}  {arrow:>7}  {pc:>+9.1f}  {cc:>+9.1f}  {pd:>10.1f}  {cd:>10.1f}")
