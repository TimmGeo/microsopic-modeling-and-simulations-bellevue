import pandas as pd

# ------------------------------------------------------------
# Dateien / Einstellungen
# ------------------------------------------------------------

# Deine OD-Matrix
OD_CSV = "S05_OD_final.csv"          # bei dir lokal

# Deine Loop→Edge-Tabelle
LOOP_EDGE_MAPPING_CSV = "Loop_edge_keys.csv"

# Output für SUMO
OUTPUT_DEMAND_XML = "demand.xml"

# Zeitintervalle (normiert: 06:00 = 0 s)
TIME_SLOTS = {
    "T1": (0, 1800),       # 06:00–06:30
    "T2": (1800, 3600),    # 06:30–07:00
    "T3": (3600, 5400),    # 07:00–07:30
    "T4": (5400, 7200),    # 07:30–08:00
    "T5": (7200, 9000),    # 08:00–08:30
    "T6": (9000, 10800),   # 08:30–09:00
    "T7": (10800, 12600),  # 09:00–09:30
    "T8": (12600, 14400),  # 09:30–10:00
}

# Nachfrage runden (True: auf int, False: float)
ROUND_DEMAND = True

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def load_loop_to_edge_mapping(path):
    """
    Lädt Loop_edge_keys.csv und baut Dict: LOOP-ID -> EDGE-ID
    Erwartete Spalten: LOOP-ID, EDGE-ID, JUNCTION-ID (+ evtl. leere Unnamed-Spalten).
    """
    # sep=None + engine="python" → automatische Trennzeichenerkennung
    df = pd.read_csv(path, sep=None, engine="python")

    # BOM / komische Zeichen im Header entfernen
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # komplett leere Spalten droppen (z.B. Unnamed: 3,4,5,…)
    df = df.dropna(axis=1, how="all")

    # Sicherstellen, dass die wichtigen Spalten da sind
    required = ["LOOP-ID", "EDGE-ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {path}: {missing}")

    loop_to_edge = {}
    for _, row in df.iterrows():
        loop_id = str(row["LOOP-ID"]).strip()
        edge_id = str(row["EDGE-ID"]).strip()
        if loop_id and edge_id and edge_id.lower() != "nan":
            loop_to_edge[loop_id] = edge_id

    return loop_to_edge


def main():
    # Mapping laden
    loop_to_edge = load_loop_to_edge_mapping(LOOP_EDGE_MAPPING_CSV)
    print(f"Loaded {len(loop_to_edge)} loop→edge mappings.")

    # OD-Matrix laden
    df = pd.read_csv(OD_CSV)

    # Erste Spalte "index" als Index benutzen (da stehen T1_K... drin)
    if "index" in df.columns:
        df = df.set_index("index")

    # NaNs durch 0 ersetzen
    df = df.fillna(0.0)

    flows = []
    flow_counter = 1   # für IDs f_1, f_2, ...

    # Für jedes Zeitintervall eigenes Submatrix: Tn_* Zeilen, Tn_* Spalten
    for slot, (begin, end) in TIME_SLOTS.items():
        print(f"Processing time slot {slot} ({begin}–{end}s)")

        # Zeilen & Spalten, die mit "Tn_" beginnen
        row_mask = [str(idx).startswith(slot + "_") for idx in df.index]
        col_mask = [str(col).startswith(slot + "_") for col in df.columns]

        sub = df.loc[row_mask, df.columns[col_mask]]

        if sub.empty:
            print(f"  WARNING: keine Daten für {slot} gefunden, überspringe.")
            continue

        # Namen ohne Zeitpräfix extrahieren: "T1_K340D14_OUT" -> "K340D14_OUT"
        row_loops = [str(idx).split("_", 1)[1] for idx in sub.index]
        col_loops = [str(col).split("_", 1)[1] for col in sub.columns]

        # Schleife über alle Zellen der Submatrix
        for i, origin_loop in enumerate(row_loops):
            # Nur Origins mit "_IN"
            if not origin_loop.endswith("_IN"):
                continue

            for j, dest_loop in enumerate(col_loops):
                # Nur Destinations mit "_OUT"
                if not dest_loop.endswith("_OUT"):
                    continue

                val = sub.iat[i, j]
                if val <= 0:
                    continue

                demand = float(val)
                if ROUND_DEMAND:
                    demand = int(round(demand))

                if demand <= 0:
                    continue

                # Loop -> Edge Mapping
                if origin_loop not in loop_to_edge:
                    print(f"  WARNING: No edge mapping for origin loop {origin_loop}, skipping.")
                    continue
                if dest_loop not in loop_to_edge:
                    print(f"  WARNING: No edge mapping for dest loop {dest_loop}, skipping.")
                    continue

                from_edge = loop_to_edge[origin_loop]
                to_edge = loop_to_edge[dest_loop]

                # Kurze, laufende IDs
                flow_id = f"f_{flow_counter}"
                flow_counter += 1

                flows.append({
                    "id": flow_id,
                    "from": from_edge,
                    "to": to_edge,
                    "begin": begin,
                    "end": end,
                    "number": demand
                })

    print(f"Total flows created: {len(flows)}")

    # --------------------------------------------------------
    # demand.xml schreiben
    # --------------------------------------------------------
    with open(OUTPUT_DEMAND_XML, "w") as out:
        out.write('<routes>\n')

        for f in flows:
            out.write(
                f'  <flow id="{f["id"]}" from="{f["from"]}" to="{f["to"]}" '
                f'begin="{f["begin"]}" end="{f["end"]}" number="{f["number"]}"/>\n'
            )

        out.write('</routes>\n')

    print(f"Wrote {OUTPUT_DEMAND_XML}")


if __name__ == "__main__":
    main()
