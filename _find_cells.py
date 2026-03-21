import json
nb = json.load(open("4_dashboard_visuals.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "run_fetch_vip" in src or "run_build_network" in src:
        print(i, repr(src[:120]))
