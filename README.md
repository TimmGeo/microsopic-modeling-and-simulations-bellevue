README MICROSCOPIC

1. Data Preparation
UTD Loop Data
  - Raw UTD loop detector data are imported and processed using an R script.
  - Relevant detectors are identified spatially in QGIS, where loop locations are inspected and selected.
  - The filtered dataset is further processed in R to clean, aggregate, and prepare detector-level flow and occupancy data.

2. OD Matrix Construction
  - The OD matrix, including the IPF-based balancing procedure, is generated in R.
  - Both morning and evening peak periods are converted into time-expanded OD tables (eight 30-minute intervals each).
  - The resulting OD matrices are exported for use in the simulation.

Conversion to SUMO Demand
  - A Python script converts the OD matrices into a SUMO-compatible demand.xml file.
  - SUMO route definitions remain empty so that routing is computed automatically at the start of each simulation run.

3. Simulation
  - All simulations are executed manually in SUMO using .sumocfg files.
  - TRACI is not used; the workflow relies solely on SUMO’s internal configuration and routing.
  - Signal plans and detector definitions are included in the simulation to better represent operational conditions.
  - Scenarios include V0, V1, and V2 for both morning and evening periods.

4. Output Processing
  - SUMO produces two primary output types for each scenario:
      Output_LoopData_V0/1/2_morning/evening.xml
      Output_LaneData_V0/1/2_morning/evening.xml
  - These files are analysed using a Python script that extracts and visualises loop counts, lane metrics, speeds and scenario comparisons.
