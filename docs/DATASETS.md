# Datasets and prepared-case format

## UrbanNav

The natural multi-receiver and controlled-fault panels use the public
UrbanNav Hong Kong data:

- Project and download instructions: https://github.com/IPNL-POLYU/UrbanNavDataset
- Panels used as supporting evidence: HK Medium, HK Deep and HK Harsh
- The three-phone panel reuses the Harsh route and changes receiver hardware;
  it is not counted as an independent route.

The estimator consumes one `*.estimator_input.npz` file per registered
solution stream.  Ground truth and fault labels must be stored separately in
matching `*.hidden_labels.npz` files.  The estimator opens the former files;
the evaluation code opens the latter only after all candidate trajectories
have been built.

Use `build_receiver_panel_cases.py --help` to construct receiver panels once
the KISS-ICP trajectory and GNSS solution files have been prepared.  Use
`build_receiver_fault_cases.py --help` for the frozen step/ramp faults.

## LOCSP

External source-cardinality replication uses LOCSP CR1 and CR2:

- Dataset DOI: https://doi.org/10.57745/YCXRWF
- LiDAR: Hesai PandarXT-32
- Estimator sources: u-blox M8T and Septentrio
- Reference only: NovAtel INSPVA

The M8T NavPVT and NavSatFix messages encode the same physical solution and
therefore map to one evidence source in `PROVENANCE_REGISTRY.json`.

## Files intentionally excluded

ROS bags, raw point clouds, KISS-ICP cache directories, RTKLIB products and
prepared case arrays are not committed.  Do not bypass `.gitignore` to upload
them.  Preserve the original dataset citations and licenses when downloading
or redistributing any source data.
